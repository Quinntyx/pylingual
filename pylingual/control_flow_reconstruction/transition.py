"""
Transition control flow model: an encoder over the model input (model_view.py) and a decoder that writes the skeleton
in source order, one block grammar action at a time (block_grammar.py).

The encoder is a transformer over the input tokens. A reference <sN> is embedded as REF plus a sinusoid of N, so there
is no statement vocabulary, and each statement's vector is the encoder state at the reference after its `|`.
The decoder is an LSTM with attention over the encoder states, fed the parser state at each step (`features`): the
last action, the action that opened the innermost block, the first statement not placed yet and the innermost frame.
It scores statements by pointing (its state against each statement vector, plus a bias for how far past the first
unplaced statement one is) and other tokens from TOKENS, over only the actions the grammar allows, so every skeleton
it writes is accepted. Beam search keeps the k best.

Trained models store their input vocabulary; changing the lists below needs new models.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import transformers
from torch import nn

from .block_grammar import END, BlockGrammar, State
from .model_view import STMT_SEP, CflowRequest
from .skeleton import CLOSE, KEYWORD_TEXT, LOOP, OPEN, parse_ref, ref

TOKENS = [END, OPEN, CLOSE, LOOP, *KEYWORD_TEXT]  # actions other than statements
STMT, BOS, ROOT = range(len(TOKENS), len(TOKENS) + 3)  # action embeddings for a statement, the start, the root block
FRAME_KINDS = ["root", "if", "while", "for", "with", "try", "except", "except_bare", "except*", "else", "try_else", "finally", "match", "case", "case_bare"]
CHAINS = [None, "if", "loop", "try", "except", "except_bare", "except*", "try_else", "case", "case_bare", "deco", "match_empty"]
MAX_DEPTH = 16  # deeper blocks share a depth embedding
DISTANCES = 16  # pointing bias buckets: statements past the first unplaced one
ALIGN = 16  # inputs are padded to a multiple of this, for the memory efficient attention kernel
SPECIAL = ["<pad>", "<unk>", "<ref>"]
PAD, UNK, REF = range(3)


class Action(NamedTuple):
    token: int  # index in TOKENS, or STMT
    stmt: int = -1  # statement of a STMT action


class Parse(NamedTuple):
    """A skeleton prefix: the grammar state and what the decoder sees of it."""

    state: State
    openers: tuple[Action, ...] = (Action(ROOT),)  # action that opened each open block, root first
    prev: Action = Action(BOS)


def text(action: Action) -> str:
    return ref(action.stmt) if action.token == STMT else TOKENS[action.token]


def advance(grammar: BlockGrammar, parse: Parse, action: Action) -> Parse:
    """Parse after `action`. Raises ValueError if the grammar does not allow it."""
    token = text(action)
    openers = parse.openers + (parse.prev,) if token == OPEN else parse.openers[:-1] if token == CLOSE else parse.openers
    return Parse(grammar.step(parse.state, token), openers, action)


def features(grammar: BlockGrammar, parse: Parse) -> list[int]:
    """Decoder input at `parse`: the previous and opening actions, the first unplaced statement and the innermost frame."""
    state = parse.state
    top = state.stack[-1]
    left = grammar.all & ~state.placed
    flags = (state.pending is not None) | state.loop_next << 1 | top.loop << 2
    return [
        *parse.prev, *parse.openers[-1], (left & -left).bit_length() - 1,
        FRAME_KINDS.index(top.kind), CHAINS.index(top.chain), flags, min(len(state.stack) - 1, MAX_DEPTH - 1),
    ]  # fmt: skip


def allowed_row(grammar: BlockGrammar, state: State, n: int) -> np.ndarray:
    """Allowed actions at `state`: statements 0..n-1 (n >= grammar.n), then TOKENS."""
    stmts, tokens = grammar.allowed(state)
    row = np.zeros(n + len(TOKENS), dtype=bool)
    if stmts:
        row[: grammar.n] = np.unpackbits(np.frombuffer(stmts.to_bytes((grammar.n + 7) // 8, "little"), np.uint8), bitorder="little")[: grammar.n]
    for t in tokens:
        row[n + TOKENS.index(t)] = True
    return row


def gold_actions(target: list[str]) -> list[Action]:
    return [Action(STMT, i) if (i := parse_ref(t)) is not None else Action(TOKENS.index(t)) for t in [*target, END]]


def encode_input(tokens: list[str], vocab: dict[str, int]) -> tuple[list[int], list[int], list[int]]:
    """Token ids, the statement each token references (-1 for none), and the position of each statement's reference."""
    ids, refs, stmt_pos = [], [], []
    for p, t in enumerate(tokens):
        i = parse_ref(t)
        ids.append(REF if i is not None else vocab.get(t, UNK))
        refs.append(-1 if i is None else i)
        if i is not None and p and tokens[p - 1] == STMT_SEP:
            stmt_pos.append(p)
    return ids, refs, stmt_pos


def sinusoid(x: torch.Tensor, d: int) -> torch.Tensor:
    freqs = torch.exp(torch.arange(0, d, 2, device=x.device) * (-math.log(10000.0) / d))
    angles = x[..., None].float() * freqs
    return torch.cat([angles.sin(), angles.cos()], -1)


class TransitionConfig(transformers.PretrainedConfig):
    model_type = "cflow_transition"

    def __init__(self, vocab: list[str] | None = None, d_model=256, layers=4, heads=8, d_ff=1024, decoder_layers=2, dropout=0.1, max_stmts=512, max_input_tokens=2048, **kwargs):
        self.vocab = vocab or list(SPECIAL)
        self.d_model, self.layers, self.heads, self.d_ff = d_model, layers, heads, d_ff
        self.decoder_layers, self.dropout = decoder_layers, dropout
        self.max_stmts, self.max_input_tokens = max_stmts, max_input_tokens
        super().__init__(**kwargs)


class TransitionModel(transformers.PreTrainedModel):
    config_class = TransitionConfig
    main_input_name = "input_ids"

    def __init__(self, config: TransitionConfig):
        super().__init__(config)
        d = config.d_model
        self.vocab = {t: i for i, t in enumerate(config.vocab)}
        self.embed = nn.Embedding(len(config.vocab), d, padding_idx=PAD)
        self.ref_proj = nn.Linear(d, d)
        layer = nn.TransformerEncoderLayer(d, config.heads, config.d_ff, config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)

        self.actions = nn.Embedding(len(TOKENS) + 3, d)
        self.no_stmt = nn.Parameter(torch.zeros(d))
        self.stmt_proj = nn.Linear(d, d)
        self.kinds = nn.Embedding(len(FRAME_KINDS), d)
        self.chains = nn.Embedding(len(CHAINS), d)
        self.flags = nn.Embedding(8, d)
        self.depths = nn.Embedding(MAX_DEPTH, d)
        self.step_in = nn.Linear(4 * d, d)
        self.lstm = nn.LSTM(d, d, config.decoder_layers, batch_first=True, dropout=config.dropout if config.decoder_layers > 1 else 0.0)
        self.attention = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d)
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.distance = nn.Linear(d, DISTANCES)
        self.token_head = nn.Linear(d, len(TOKENS))
        self.post_init()

    def encode(self, input_ids, refs, attention_mask, stmt_pos):
        """Encoder states and statement vectors."""
        d = self.config.d_model
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        x = self.embed(input_ids) * math.sqrt(d) + sinusoid(positions, d)
        x = x + (refs >= 0)[..., None] * self.ref_proj(sinusoid(refs.clamp(min=0), d))
        memory = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        return memory, memory.gather(1, stmt_pos[..., None].expand(-1, -1, d))

    def step_inputs(self, stmts, feats):
        """Decoder inputs for `features` rows [B, T, 9], with statement vectors [B, N, d]."""
        table = torch.cat([self.no_stmt.expand(stmts.shape[0], 1, -1), self.stmt_proj(stmts)], 1)
        pick = lambda i: table.gather(1, (feats[..., i] + 1)[..., None].expand(-1, -1, table.shape[-1]))  # noqa: E731
        prev = self.actions(feats[..., 0]) + pick(1)
        opener = self.actions(feats[..., 2]) + pick(3)
        frame = self.kinds(feats[..., 5]) + self.chains(feats[..., 6]) + self.flags(feats[..., 7]) + self.depths(feats[..., 8])
        return self.step_in(torch.cat([prev, opener, pick(4), frame], -1))

    def scores(self, h, memory, attention_mask, stmts, next_stmt):
        """Action logits [B, T, N + len(TOKENS)] for decoder states `h`."""
        context, _ = self.attention(h, memory, memory, key_padding_mask=~attention_mask.bool(), need_weights=False)
        o = self.out_norm(h + context)
        pointed = self.query(o) @ self.key(stmts).transpose(1, 2) / math.sqrt(self.config.d_model)
        distance = (torch.arange(stmts.shape[1], device=h.device) - next_stmt[..., None]).clamp(0, DISTANCES - 1)
        pointed = pointed + self.distance(o).gather(-1, distance)
        return torch.cat([pointed, self.token_head(o)], -1).float()

    def forward(self, input_ids, refs, attention_mask, stmt_pos, features, allowed, labels=None):
        """Teacher forced: `features` and `allowed` of each step of the gold skeleton, whose actions are `labels`."""
        memory, stmts = self.encode(input_ids, refs, attention_mask, stmt_pos)
        h, _ = self.lstm(self.step_inputs(stmts, features))
        logits = self.scores(h, memory, attention_mask, stmts, features[..., 4])
        if labels is None:
            return {"logits": logits.masked_fill(~allowed, -math.inf)}
        logits = logits.masked_fill(~(allowed | (labels == -100)[..., None]), -math.inf)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        return {"loss": loss}

    def batch_inputs(self, requests: list[CflowRequest]) -> dict[str, torch.Tensor]:
        """Padded encoder inputs for `requests`."""
        encoded = [encode_input(r.tokens, self.vocab) for r in requests]
        length, n = -(-max(len(e[0]) for e in encoded) // ALIGN) * ALIGN, max(len(e[2]) for e in encoded)
        pad = lambda rows, width, value: torch.tensor([row + [value] * (width - len(row)) for row in rows], device=self.device)  # noqa: E731
        return {
            "input_ids": pad([e[0] for e in encoded], length, PAD),
            "refs": pad([e[1] for e in encoded], length, -1),
            "attention_mask": pad([[1] * len(e[0]) for e in encoded], length, 0),
            "stmt_pos": pad([e[2] for e in encoded], n, 0),
        }

    @torch.no_grad()
    def generate(self, requests: list[CflowRequest], num_beams: int = 1) -> list[list[list[str]]]:
        """Up to `num_beams` distinct skeletons (source order, best first) for each request, all accepted by its grammar."""
        k, inputs = num_beams, self.batch_inputs(requests)
        memory, stmts = self.encode(**inputs)
        n = stmts.shape[1]
        memory, stmts, mask = (x.repeat_interleave(k, 0) for x in (memory, stmts, inputs["attention_mask"]))
        grammars = [r.grammar() for r in requests]
        limits = [4 * r.n + 16 for r in requests]
        beams = [[(0.0, Parse(g.start()), [])] for g in grammars]  # (log prob, parse, tokens) in slots r * k + j
        finished: list[list[tuple[float, list[str]]]] = [[] for _ in requests]
        state = None
        while any(beams):
            feats = np.zeros((len(requests) * k, 9), dtype=np.int64)
            allowed = np.zeros((len(requests) * k, n + len(TOKENS)), dtype=bool)
            base = np.full(len(requests) * k, -math.inf, dtype=np.float32)
            for r, (grammar, beam) in enumerate(zip(grammars, beams)):
                for j, (score, parse, _) in enumerate(beam):
                    feats[r * k + j] = features(grammar, parse)
                    allowed[r * k + j] = allowed_row(grammar, parse.state, n)
                    base[r * k + j] = score
            allowed[base == -math.inf, n] = True  # empty slots score -inf, not NaN
            feats_t = torch.from_numpy(feats).to(self.device)[:, None]
            h, state = self.lstm(self.step_inputs(stmts, feats_t), state)
            logits = self.scores(h, memory, mask, stmts, feats_t[..., 4])[:, 0]
            logits = logits.masked_fill(~torch.from_numpy(allowed).to(self.device), -math.inf).log_softmax(-1)
            total = (logits + torch.from_numpy(base).to(self.device)[:, None]).view(len(requests), -1)
            top_scores, top = (x.tolist() for x in total.topk(min(2 * k, total.shape[1]), -1))
            parents = list(range(len(requests) * k))
            for r, grammar in enumerate(grammars):
                if not beams[r]:
                    continue
                new = []
                for score, flat in zip(top_scores[r], top[r]):
                    if score == -math.inf or len(new) == k:
                        break
                    j, a = divmod(flat, n + len(TOKENS))
                    _, parse, tokens = beams[r][j]
                    action = Action(STMT, a) if a < n else Action(a - n)
                    if action.token == TOKENS.index(END):
                        finished[r].append((score, tokens))
                        continue
                    parse, tokens = advance(grammar, parse, action), tokens + [text(action)]
                    if len(tokens) >= limits[r]:  # the model would run on: finish the skeleton by the grammar
                        finished[r].append((score, tokens + grammar.complete(parse.state)))
                    else:
                        parents[r * k + len(new)] = r * k + j
                        new.append((score, parse, tokens))
                best = sorted(finished[r], key=lambda f: -f[0])[:k]
                beams[r] = [] if len(best) == k and (not new or new[0][0] <= best[-1][0]) else new
            index = torch.tensor(parents, device=self.device)
            state = tuple(s.index_select(1, index) for s in state)
        return [[tokens for _, tokens in sorted(f, key=lambda f: -f[0])[:k]] for f in finished]
