"""
Control flow model with a chart parser (ARCH chart): a T5 encoder over the model input (model_view.py), whose statement
vectors score the source order of the statements and the items of a skeleton in that order (chart.Scores). Decoding
searches the skeletons the block grammar accepts exactly (chart.Chart).

The heads, over statement vectors (a statement's `|` token and the mean of its tokens):
    order       which statement comes next in source order (biaffine), decoded greedily
    header_end  the gap where each header's block ends (a pointer over the gaps after it)
    keyword_end the gap where a keyword block ends, by keyword
    insert      the keywords at each gap
    loop        whether a cond header is a while
Gaps are between consecutive statements of an order: the source order in training, the predicted one in decoding.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5Config, T5EncoderModel

from .chart import KEYWORDS, NONE, Chart, Scores, decode_order, order_beams, order_score
from .model_view import EDGE_TOKENS, EDGES, CflowRequest
from .skeleton import BLOCK_KEYWORDS

EDGE_KINDS = list(EDGE_TOKENS.values())
EDGE_FEATURES = 2 * len(EDGE_KINDS)  # each kind, landing at the start of the statement or inside it
MASKED = -1e9  # a score for what cannot be, finite so that a row of them has no NaN


def mlp(d: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))


class ChartParser(T5EncoderModel):
    """Its config holds `max_stmts`, `max_input_tokens`, `skeleton_keywords` (seen in training) and `statement_token_id` (of `|`)."""

    def __init__(self, config: T5Config):
        super().__init__(config)
        d = config.d_model
        self.statement = nn.Linear(2 * d, d)
        self.order_ends = nn.Parameter(torch.randn(2, d) * 0.02)  # before the first statement, after the last
        self.before, self.after = mlp(d), mlp(d)
        self.gap_ends = nn.Parameter(torch.randn(2, d) * 0.02)
        self.gap = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.opener = nn.Parameter(torch.randn(1 + len(BLOCK_KEYWORDS), d) * 0.02)  # a header's block, then each keyword's
        self.open, self.close = mlp(d), mlp(d)
        self.insert = nn.Linear(d, len(KEYWORDS))
        self.loop = nn.Linear(d, 1)
        # a successor score by how far the successor is in statement order (most code keeps it)
        self.order_bias = nn.Embedding(config.order_buckets, 1) if getattr(config, "order_buckets", 0) else None
        if self.order_bias is not None:
            nn.init.zeros_(self.order_bias.weight)
        # scores of the edges between two statements in the input (EDGE_KINDS, also landing inside it), for: the
        # successor (from the statement, to it), a header's block end (from the header to the statement after the end,
        # from the statement before the end to the header) and each keyword's block end (from the statement before
        # the keyword to the statement after the end)
        self.edge_weights = nn.Parameter(torch.zeros(4 + len(BLOCK_KEYWORDS), EDGE_FEATURES)) if getattr(config, "edge_features", False) else None

    def edges(self, input_ids: torch.Tensor, statement: torch.Tensor, n: int) -> torch.Tensor | None:
        """(batch, statements, statements, EDGE_FEATURES): whether a statement has an edge of each kind to another."""
        if self.edge_weights is None:
            return None
        c = self.config
        kind = input_ids.new_full((c.vocab_size,), -1)
        for k, ids in enumerate(c.edge_token_ids):
            kind[ids] = k
        previous = kind[torch.cat([input_ids.new_zeros(len(input_ids), 1), input_ids[:, :-1]], 1)]
        inside = torch.cat([input_ids[:, 1:], input_ids.new_zeros(len(input_ids), 1)], 1) == c.mid_token_id
        target = input_ids - c.ref_token_id
        rows, cols = ((previous >= 0) & (target >= 0) & (target < n)).nonzero(as_tuple=True)
        out = torch.zeros(len(input_ids), n, n, EDGE_FEATURES, device=input_ids.device)
        out[rows, statement[rows, cols], target[rows, cols], previous[rows, cols] + len(EDGE_KINDS) * inside[rows, cols]] = 1
        return out

    @staticmethod
    def padded(edges: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        """Edges between the positions of `order` (batch, statements), with no edges for positions -1 and n: index x + 1 is position x."""
        rows = torch.arange(len(order), device=order.device)[:, None]
        ordered = edges[rows, order.clamp(min=0)]
        ordered = ordered[rows[..., None], torch.arange(order.shape[1], device=order.device)[None, :, None], order.clamp(min=0)[:, None, :]]
        return F.pad(ordered, (0, 0, 1, 1, 1, 1))

    def header_edges(self, padded: torch.Tensor) -> torch.Tensor:
        """(batch, n, n + 1) edge scores of the header at each position ending at each gap."""
        n = padded.shape[1] - 2
        w = self.edge_weights.to(padded.dtype)
        return padded[:, 1 : n + 1, 1:] @ w[2] + (padded[:, : n + 1, 1 : n + 1] @ w[3]).transpose(1, 2)

    def offset_buckets(self, offsets: torch.Tensor) -> torch.Tensor:
        """Buckets of statement offsets like T5's relative positions: one each up to 8 away, then logarithmic up to 256."""
        half, exact = self.config.order_buckets // 2, 8
        size = offsets.abs()
        far = exact + (torch.log(size.clamp(min=1).float() / exact) / math.log(256 / exact) * (half - exact)).long()
        return torch.where(size < exact, size, far.clamp(max=half - 1)) + half * (offsets > 0)

    def statements(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Statement vectors (batch, statements, d), statement counts (batch) and edges between statements (see `edges`)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        starts = input_ids == self.config.statement_token_id
        counts = starts.sum(1)
        index = (starts.cumsum(1) - 1).clamp(min=0)
        weight = (attention_mask.bool() & (starts.cumsum(1) > 0)).to(hidden.dtype)[..., None]
        shape = (hidden.shape[0], max(int(counts.max()), 1), hidden.shape[-1])
        sums = hidden.new_zeros(shape).scatter_add(1, index[..., None].expand_as(hidden), hidden * weight)
        sizes = hidden.new_zeros(shape[:2]).scatter_add(1, index, weight[..., 0])
        heads = hidden.new_zeros(shape)
        rows, cols = starts.nonzero(as_tuple=True)
        heads[rows, index[rows, cols]] = hidden[rows, cols]
        return self.statement(torch.cat([heads, sums / sizes.clamp(min=1)[..., None]], -1)), counts, self.edges(input_ids, index, shape[1])

    def successors(self, vectors: torch.Tensor, counts: torch.Tensor, edges: torch.Tensor | None = None) -> torch.Tensor:
        """
        (batch, 1 + statements, statements + 1) scores of what comes next in source order: rows are the start and each
        statement, columns each statement and the end (the last column).
        """
        b, n, _ = vectors.shape
        start, end = self.order_ends[:1].expand(b, 1, -1), self.order_ends[1:].expand(b, 1, -1)
        scores = self.before(torch.cat([start, vectors], 1)) @ self.after(torch.cat([vectors, end], 1)).transpose(1, 2)
        column = torch.arange(n + 1, device=vectors.device)
        if self.order_bias is not None:
            # rows: the start (-1) and each statement; columns: each statement and the end (at the statement count)
            to = torch.where(column[None] == n, counts[:, None], column[None])
            scores = scores + self.order_bias(self.offset_buckets(to[:, None, :] - (column - 1)[None, :, None]))[..., 0].to(scores.dtype)
        if edges is not None:
            w = self.edge_weights.to(scores.dtype)
            pair = edges.to(scores.dtype) @ w[0] + (edges.to(scores.dtype) @ w[1]).transpose(1, 2)
            scores = scores + F.pad(pair, (0, 1, 1, 0))
        mask = (column[None, None, : n + 1] < counts[:, None, None]) | (column == n)[None, None]
        mask = mask & (column[None, None] != column[None, :, None] - 1)  # not the statement itself
        return scores.masked_fill(~mask, MASKED)

    def gaps(self, ordered: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """(batch, statements + 1, d) gap vectors from statement vectors in order: gap g is before statement g."""
        b, n, d = ordered.shape
        last = torch.arange(n + 1, device=ordered.device)[None, :, None] == counts[:, None, None]
        left = torch.cat([self.gap_ends[:1].expand(b, 1, d), ordered], 1)
        right = torch.where(last, self.gap_ends[1], torch.cat([ordered, ordered.new_zeros(b, 1, d)], 1))
        return self.gap(torch.cat([left, right], -1))

    def forward(self, input_ids, attention_mask, order, header_end, keyword_end, insert, loop, **_):
        """Training loss. Labels (see train_chart.py) are in source order: `order` holds the statements in source order."""
        vectors, counts, edges = self.statements(input_ids, attention_mask)
        n = vectors.shape[1]
        order, header_end, loop = order[:, :n], header_end[:, :n], loop[:, :n]
        keyword_end, insert = keyword_end[:, : n + 1], insert[:, : n + 1]
        rows = torch.arange(len(counts), device=counts.device)[:, None]
        # successor of the start and of each statement in source order
        successor = torch.full((len(counts), n + 1), -100, device=counts.device)
        successor[:, 0] = order[:, 0]
        previous = torch.cat([order[:, :1], order[:, :-1]], 1)
        position = torch.arange(n, device=counts.device)[None]
        has_previous = (position > 0) & (position < counts[:, None])
        successor[rows.expand_as(order)[has_previous], previous[has_previous] + 1] = order[has_previous]
        is_last = position == counts[:, None] - 1
        successor[rows.expand_as(order)[is_last], order[is_last] + 1] = n
        loss = F.cross_entropy(self.successors(vectors, counts, edges).flatten(0, 1), successor.flatten(), ignore_index=-100)
        padded = None if edges is None else self.padded(edges, order)

        gap = self.gaps(vectors[rows, order.clamp(min=0)], counts)
        close = self.close(gap)
        valid_end = torch.arange(n + 1, device=counts.device)[None] <= counts[:, None]
        # header blocks: gap p + 1 holds the header at p
        scores = self.open(gap[:, 1:] + self.opener[0]) @ close.transpose(1, 2)
        if padded is not None:
            scores = scores + self.header_edges(padded.to(scores.dtype))
        after = torch.arange(n + 1, device=counts.device)[None] > torch.arange(n, device=counts.device)[:, None]
        scores = scores.masked_fill(~(after[None] & valid_end[:, None]), MASKED)
        if (header_end != -100).any():  # a batch without headers has no header loss (not NaN)
            loss = loss + F.cross_entropy(scores.flatten(0, 1), header_end.flatten(), ignore_index=-100)
        # keyword blocks, only where there is one
        b, g, k = (keyword_end != -100).nonzero(as_tuple=True)
        if len(b):
            scores = (self.open(gap[b, g] + self.opener[1 + k])[:, None] * close[b]).sum(-1)
            if padded is not None:
                scores = scores + (padded[b, g, 1:].to(scores.dtype) * self.edge_weights[4 + k][:, None].to(scores.dtype)).sum(-1)
            gaps = torch.arange(n + 1, device=counts.device)[None]
            scores = scores.masked_fill(~((gaps > g[:, None]) & valid_end[b]), MASKED)
            loss = loss + F.cross_entropy(scores, keyword_end[b, g, k])
        logits = self.insert(gap)
        loss = loss + F.binary_cross_entropy_with_logits(logits[valid_end], insert[valid_end].to(logits.dtype))
        logits = self.loop(vectors[rows, order.clamp(min=0)])[..., 0]
        if (loop != -100).any():
            loss = loss + F.binary_cross_entropy_with_logits(logits[loop != -100], loop[loop != -100].to(logits.dtype))
        return {"loss": loss}

    def scores(self, vectors: torch.Tensor, edges: torch.Tensor | None, order: list[int]) -> Scores:
        """Item scores of one code object's statement vectors (statements, d) and edges in `order`."""
        n = len(order)
        gap = self.gaps(vectors[order][None], torch.tensor([n], device=vectors.device))[0]
        close = self.close(gap)
        gaps = torch.arange(n + 1, device=vectors.device)
        header = self.open(gap[1:] + self.opener[0]) @ close.T
        if edges is not None:
            padded = self.padded(edges[None], torch.tensor([order], device=vectors.device))
            header = header + self.header_edges(padded)[0]
        header = header.masked_fill(gaps[None] <= gaps[:n, None], float("-inf")).log_softmax(-1)
        keyword = torch.einsum("gkd,ed->kge", self.open(gap[:, None] + self.opener[1:]), close)
        if edges is not None:
            keyword = keyword + torch.einsum("get,kt->kge", padded[0, : n + 1, 1:], self.edge_weights[4:])
        keyword = keyword.masked_fill(gaps[None, None] <= gaps[None, :, None], float("-inf")).log_softmax(-1).nan_to_num(nan=NONE, neginf=NONE)
        z = self.loop(vectors[order])[:, 0]
        loop = F.logsigmoid(torch.stack([-z, z], -1))
        return Scores(*(x.float().cpu().numpy() for x in (header, loop, self.insert(gap), keyword)))

    def skeletons(self, request: CflowRequest, vectors: torch.Tensor, edges: torch.Tensor | None, successors: torch.Tensor, num_beams: int) -> list[list[str]]:
        """
        Up to `num_beams` skeletons the block grammar accepts, best first, by the log probability of their order and of
        their items in it. The orders: the `order_candidates` best of the successor head (config, 1 by default) and
        the statement order. BlockGrammar.complete when no order has one.
        """
        grammar = request.grammar()
        n = request.n
        columns = list(range(n)) + [successors.shape[-1] - 1]  # each statement, then the end
        log_probs = successors[: n + 1, columns].float().log_softmax(-1).cpu().numpy()
        orders = order_beams(log_probs, request.shapes, getattr(self.config, "order_candidates", 1))
        in_order = decode_order(None, request.shapes)
        if all(order != in_order for _, order in orders):
            orders.append((order_score(log_probs, in_order), in_order))
        keywords = [KEYWORDS.index(k) for k in self.config.skeleton_keywords]
        found = []
        for score, order in orders:
            scores = self.scores(vectors, edges, order)
            # insert holds log odds: a skeleton's probability has every keyword it leaves out, too
            score += -float(np.logaddexp(0, scores.insert[:, keywords]).sum())
            accepted = 0
            for item_score, tokens in Chart(grammar, order, scores, self.config.skeleton_keywords).best(4 * num_beams):
                if grammar.accepts(tokens):
                    found.append((score + item_score, tokens))
                    accepted += 1
                    if accepted == num_beams:
                        break
        found.sort(key=lambda f: -f[0])
        return [tokens for _, tokens in found[:num_beams]] or [grammar.complete(grammar.start())]
