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

from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5Config, T5EncoderModel

from .chart import KEYWORDS, NONE, Chart, Scores, decode_order
from .model_view import CflowRequest
from .skeleton import BLOCK_KEYWORDS

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

    def statements(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Statement vectors (batch, statements, d) and statement counts (batch)."""
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
        return self.statement(torch.cat([heads, sums / sizes.clamp(min=1)[..., None]], -1)), counts

    def successors(self, vectors: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        (batch, 1 + statements, statements + 1) scores of what comes next in source order: rows are the start and each
        statement, columns each statement and the end (the last column).
        """
        b, n, _ = vectors.shape
        start, end = self.order_ends[:1].expand(b, 1, -1), self.order_ends[1:].expand(b, 1, -1)
        scores = self.before(torch.cat([start, vectors], 1)) @ self.after(torch.cat([vectors, end], 1)).transpose(1, 2)
        column = torch.arange(n + 1, device=vectors.device)
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
        vectors, counts = self.statements(input_ids, attention_mask)
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
        loss = F.cross_entropy(self.successors(vectors, counts).flatten(0, 1), successor.flatten(), ignore_index=-100)

        gap = self.gaps(vectors[rows, order.clamp(min=0)], counts)
        close = self.close(gap)
        valid_end = torch.arange(n + 1, device=counts.device)[None] <= counts[:, None]
        # header blocks: gap p + 1 holds the header at p
        scores = self.open(gap[:, 1:] + self.opener[0]) @ close.transpose(1, 2)
        after = torch.arange(n + 1, device=counts.device)[None] > torch.arange(n, device=counts.device)[:, None]
        scores = scores.masked_fill(~(after[None] & valid_end[:, None]), MASKED)
        if (header_end != -100).any():  # a batch without headers has no header loss (not NaN)
            loss = loss + F.cross_entropy(scores.flatten(0, 1), header_end.flatten(), ignore_index=-100)
        # keyword blocks, only where there is one
        b, g, k = (keyword_end != -100).nonzero(as_tuple=True)
        if len(b):
            scores = (self.open(gap[b, g] + self.opener[1 + k])[:, None] * close[b]).sum(-1)
            gaps = torch.arange(n + 1, device=counts.device)[None]
            scores = scores.masked_fill(~((gaps > g[:, None]) & valid_end[b]), MASKED)
            loss = loss + F.cross_entropy(scores, keyword_end[b, g, k])
        logits = self.insert(gap)
        loss = loss + F.binary_cross_entropy_with_logits(logits[valid_end], insert[valid_end].to(logits.dtype))
        logits = self.loop(vectors[rows, order.clamp(min=0)])[..., 0]
        if (loop != -100).any():
            loss = loss + F.binary_cross_entropy_with_logits(logits[loop != -100], loop[loop != -100].to(logits.dtype))
        return {"loss": loss}

    def scores(self, vectors: torch.Tensor, order: list[int]) -> Scores:
        """Item scores of one code object's statement vectors (statements, d) in `order`."""
        n = len(order)
        gap = self.gaps(vectors[order][None], torch.tensor([n], device=vectors.device))[0]
        close = self.close(gap)
        gaps = torch.arange(n + 1, device=vectors.device)
        header = self.open(gap[1:] + self.opener[0]) @ close.T
        header = header.masked_fill(gaps[None] <= gaps[:n, None], float("-inf")).log_softmax(-1)
        keyword = torch.einsum("gkd,ed->kge", self.open(gap[:, None] + self.opener[1:]), close)
        keyword = keyword.masked_fill(gaps[None, None] <= gaps[None, :, None], float("-inf")).log_softmax(-1).nan_to_num(nan=NONE, neginf=NONE)
        z = self.loop(vectors[order])[:, 0]
        loop = F.logsigmoid(torch.stack([-z, z], -1))
        return Scores(*(x.float().cpu().numpy() for x in (header, loop, self.insert(gap), keyword)))

    def skeletons(self, request: CflowRequest, vectors: torch.Tensor, successors: torch.Tensor, num_beams: int) -> list[list[str]]:
        """
        Up to `num_beams` skeletons the block grammar accepts, best first: from the predicted order, then from the
        statement order (the last beam when it is free); BlockGrammar.complete when no order has one.
        """
        grammar = request.grammar()
        predicted = decode_order(successors.float().cpu().numpy(), request.shapes)
        in_order = decode_order(None, request.shapes)
        found = []

        def accepted(order: list[int]) -> Iterator[list[str]]:
            chart = Chart(grammar, order, self.scores(vectors, order), self.config.skeleton_keywords)
            for _, tokens in chart.best(4 * num_beams):
                if grammar.accepts(tokens):
                    yield tokens

        for tokens in accepted(predicted):
            found.append(tokens)
            if len(found) == num_beams:
                break
        if predicted != in_order and (len(found) < num_beams or num_beams > 1):
            for tokens in accepted(in_order):
                if tokens not in found:
                    found = found[: num_beams - 1] + [tokens] if len(found) == num_beams else found + [tokens]
                    if len(found) == num_beams:
                        break
        return found or [grammar.complete(grammar.start())]
