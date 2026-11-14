"""
Output language of the control flow model and the grammar that keeps its decoding valid.

A skeleton is the nesting of a code object's statements: `<s0> { <s1> } else { <s2> }`. The model writes it in model
order (see `to_model_order`); the grammar makes sure braces balance, every statement is placed exactly once, keyword
blocks are never empty, and keywords the model never saw in training are never generated.

Trained models store their vocabulary, so changing a token spelling needs new models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from pylingual.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch
    import transformers
else:
    lazy_import("torch")

OPEN, CLOSE = "{", "}"  # body of the statement or keyword before OPEN
LOOP = "loop"  # the next statement is a `while` header
PLACE = "="  # model order only: where the referenced statement goes (see `to_model_order`)
# keywords for lines without bytecode of their own; dict order is the training vocabulary order
KEYWORD_TEXT = {
    "else": "else:",
    "try": "try:",
    "finally": "finally:",
    "except": "except:",
    "while_true": "while True:",
    "break": "break",
    "continue": "continue",
    "case_": "case _:",
}
BLOCK_KEYWORDS = ("else", "try", "finally", "except", "while_true", "case_")  # followed by a body
SIMPLE_KEYWORDS = ("break", "continue")  # statements on their own


def ref(i: int) -> str:
    """Reference to statement i."""
    return f"<s{i}>"


def parse_ref(token: str) -> int | None:
    """Statement index of a reference, None for other tokens."""
    if token.startswith("<s") and token.endswith(">") and token[2:-1].isdigit():
        return int(token[2:-1])
    return None


def to_model_order(tokens: list[str]) -> list[str]:
    """
    Reorder a skeleton so each statement reference comes first, followed by the tokens before it and PLACE:
    `} else { loop <s7> {` becomes `<s7> } else { loop = {`. The model then decides each structure token after it has
    chosen the statement. PLACE is left out when nothing comes before the statement.
    """
    segments: list[tuple[str, list[str], bool]] = []  # (reference, tokens before it, opens a block)
    before: list[str] = []
    i = 0
    while i < len(tokens):
        if parse_ref(tokens[i]) is not None:
            opens = i + 1 < len(tokens) and tokens[i + 1] == OPEN
            segments.append((tokens[i], before, opens))
            before = []
            i += opens
        else:
            before.append(tokens[i])
        i += 1
    out: list[str] = []
    for n, (reference, pre, opens) in enumerate(segments):
        out.append(reference)
        is_last = n == len(segments) - 1
        if pre or (not opens and is_last and before):
            out += [*pre, PLACE]
        if opens:
            out.append(OPEN)
    return out + before


def from_model_order(tokens: list[str]) -> list[str]:
    """Inverse of `to_model_order`."""
    out: list[str] = []
    pending: str | None = None  # reference waiting for its PLACE
    pre: list[str] = []
    for token in tokens:
        if parse_ref(token) is not None:
            if pending is not None:
                out += [*pre, pending]
                pre = []
            pending = token
        elif pending is not None and (token == PLACE or (token == OPEN and not pre)):
            out += [*pre, pending]
            pending, pre = None, []
            if token == OPEN:
                out.append(token)
        elif pending is not None:
            pre.append(token)
        else:
            out.append(token)
    if pending is not None:
        out += [*pre, pending]
    return out


class State(NamedTuple):
    used: int  # bitmask of placed statements
    depth: int  # open blocks
    phase: str  # "start", "ref" (right after a reference), "pre" (tokens before its PLACE) or "post" (placed)
    last: str  # kind of the last token: ref, place, open, close, loop, block_kw, stmt
    keyword_run: int  # block keywords since the last statement
    empty_block: bool  # a keyword block is open and holds no statement yet


START = State(0, 0, "start", "start", 0, False)


class SkeletonGrammar:
    """
    Rules: a block keyword is followed by OPEN, LOOP by PLACE; braces balance; a block opened after a keyword holds a
    statement before it closes; at most MAX_KEYWORD_RUN keywords in a row; every statement is placed once, then EOS.
    """

    MAX_KEYWORD_RUN = 2

    def __init__(self, tokenizer: transformers.PreTrainedTokenizerFast, max_stmts: int, keywords: list[str]):
        """`keywords`: the keywords seen in training."""
        vocab = tokenizer.get_vocab()
        self.eos = tokenizer.eos_token_id
        self.open, self.close, self.loop, self.place = vocab[OPEN], vocab[CLOSE], vocab[LOOP], vocab[PLACE]
        self.block_kw = [vocab[k] for k in BLOCK_KEYWORDS if k in keywords]
        self.simple_kw = [vocab[k] for k in SIMPLE_KEYWORDS if k in keywords]
        self.refs = np.array([vocab[ref(i)] for i in range(max_stmts)])  # token id of statement i
        self.statement_of = {int(t): i for i, t in enumerate(self.refs)}
        self.cache: dict[tuple, State] = {}

    def step(self, s: State, token: int) -> State:
        # a statement is placed at PLACE, or when its reference is followed by another reference or OPEN
        if token in self.statement_of:
            return State(s.used | (1 << self.statement_of[token]), s.depth, "ref", "ref", 0, s.empty_block and s.phase != "ref")
        if token == self.place:
            return State(s.used, s.depth, "post", "place", s.keyword_run, False)
        if token == self.open:
            if s.phase == "ref":
                return State(s.used, s.depth + 1, "post", "open", s.keyword_run, False)
            return State(s.used, s.depth + 1, s.phase, "open", s.keyword_run, s.empty_block or s.last == "block_kw")
        phase = "pre" if s.phase == "ref" else s.phase
        if token == self.close:
            return State(s.used, max(s.depth - 1, 0), phase, "close", s.keyword_run, s.empty_block)
        if token == self.loop:
            return State(s.used, s.depth, phase, "loop", s.keyword_run, s.empty_block)
        if token in self.block_kw:
            return State(s.used, s.depth, phase, "block_kw", s.keyword_run + 1, s.empty_block)
        return State(s.used, s.depth, phase, "stmt", 0, False)  # break / continue count as statements

    def state(self, prefix: tuple) -> State:
        """State after `prefix` = (decoder start token, *generated tokens)."""
        if prefix not in self.cache:
            self.cache[prefix] = START if len(prefix) <= 1 else self.step(self.state(prefix[:-1]), prefix[-1])
        return self.cache[prefix]

    def unused(self, used: int, n: int) -> np.ndarray:
        """Token ids of the statements not placed yet, out of n."""
        bits = np.unpackbits(np.frombuffer(used.to_bytes((n + 7) // 8 or 1, "little"), dtype=np.uint8), bitorder="little")[:n]
        return self.refs[:n][bits == 0]

    def fill(self, row: np.ndarray, s: State, n: int):
        """Set the allowed next tokens to True in `row` (one entry per vocabulary id)."""
        if s.last == "block_kw":
            row[self.open] = True
            return
        if s.last == "loop":
            row[self.place] = True
            return
        unused = self.unused(s.used, n)
        if s.phase == "start":
            row[unused if len(unused) else self.eos] = True
            return
        # a keyword block needs a statement for its body: the one being placed, or one not placed yet
        if s.keyword_run < self.MAX_KEYWORD_RUN and (s.phase != "post" or len(unused)):
            row[self.block_kw] = True
        row[self.simple_kw] = True
        if s.depth and not s.empty_block:
            row[self.close] = True
        if s.phase in ("ref", "pre"):
            row[self.place] = row[self.loop] = True
            if s.phase == "pre":
                return
        row[unused] = True
        if s.last in ("place", "ref"):
            row[self.open] = True
        if not s.depth and not len(unused):
            # every statement must be placed: dropping a real statement is worse than keeping a duplicate
            row[self.eos] = True

    def finish_placing(self, row: np.ndarray, s: State, n: int, best: int):
        """
        If the model wants to stop (`best` is its top token) while statements are left, only allow placing them (or
        closing the open block). Otherwise it keeps writing keyword blocks around them, which never compiles.
        """
        if best != self.eos or row[self.eos] or s.phase == "pre" or s.last in ("block_kw", "loop"):
            return
        unused = self.unused(s.used, n)
        if len(unused):
            close = row[self.close]
            row[:] = False
            row[unused] = True
            row[self.close] = close


class GrammarLogitsProcessor:
    """Applies the grammar to `generate()` (beam search). Inputs are in batch order, `num_beams` rows each."""

    def __init__(self, grammar: SkeletonGrammar, n_stmts: list[int], num_beams: int):
        self.grammar = grammar
        self.n_stmts = n_stmts
        self.num_beams = num_beams

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = np.zeros(scores.shape, dtype=bool)
        best = scores.argmax(-1).tolist()
        for row, ids in enumerate(input_ids.tolist()):
            s = self.grammar.state(tuple(ids))
            n = self.n_stmts[row // self.num_beams]
            self.grammar.fill(mask[row], s, n)
            self.grammar.finish_placing(mask[row], s, n, best[row])
        return scores.masked_fill(~torch.from_numpy(mask).to(scores.device), float("-inf"))
