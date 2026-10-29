"""
Exact k-best search over the skeletons the block grammar accepts for one order of the statements (a chart parser).

With the order fixed, a skeleton is a tree of items over the gaps between statements (gap g comes before the statement
at position g, gap n is the end):
    statement           <sP>                        position p
    header and block    [loop] <sP> { body }        its block spans gaps p + 1 .. e
    keyword block       else { body }               spans gaps g .. e, with a statement in it (g < e)
    jump                break / continue            no statement
A body is a sequence of items, and what may follow an item is its chain (block_grammar.CHAIN_AFTER). Like CKY, the
chart holds the best sequence for each frame context, span and last chain, built from shorter ones; k-best follows
Huang and Chiang 2005 (algorithm 3). A keyword block does not start with a keyword block (one keyword per gap) and two
jumps never follow each other, so the search is finite.

A skeleton scores the sum of its items' `Scores`. The rules BlockGrammar checks with counts (blocks, depth, cases left)
are left to BlockGrammar.check on the results.
"""

from __future__ import annotations

import heapq
import sys
from dataclasses import dataclass
from typing import Iterator, NamedTuple

import numpy as np

from .block_grammar import CHAIN_AFTER, KEYWORD_SHAPES, OBLIGATIONS, ROOT, BlockGrammar, Frame, Shape, State
from .skeleton import BLOCK_KEYWORDS, CLOSE, KEYWORD_TEXT, LOOP, OPEN, SIMPLE_KEYWORDS, parse_ref, ref

KEYWORDS = list(KEYWORD_TEXT)  # in the order of Scores.insert
CHAINS = [None, "if", "loop", "try", "except", "except_bare", "except*", "try_else", "case", "case_bare", "deco", "match_empty"]
CLOSABLE = np.array([c not in OBLIGATIONS for c in CHAINS])
NONE = float("-inf")


@dataclass
class Scores:
    """Log scores of the items for one order of n statements."""

    header_end: np.ndarray  # (n, n + 1): the block of the header at position p ends at gap e
    loop: np.ndarray  # (n, 2): the header at position p is an if (0) or a while (1), for cond headers
    insert: np.ndarray  # (n + 1, len(KEYWORDS)): log odds of a keyword at gap g
    keyword_end: np.ndarray  # (len(BLOCK_KEYWORDS), n + 1, n + 1): the keyword block at gap g ends at gap e


class Item(NamedTuple):
    kind: str  # stmt, header, loop (a header after LOOP), keyword, jump
    keyword: str | None
    after: tuple[bool, ...]  # chains it may follow
    chain: int  # chain after it
    child: int  # context of its block, -1 without one
    score: np.ndarray  # stmt: (n,) by position; block: (start, end gap) without the block's body; jump: (n + 1,) by gap
    positions: np.ndarray | None  # of the statements a stmt, header or loop item can be


class Gold(NamedTuple):
    """A skeleton as chart items: positions are in `order`, gaps count the statements before them."""

    order: list[int]
    header_end: dict[int, int]  # position -> end gap
    loop: dict[int, bool]  # position -> written after LOOP
    keywords: list[tuple[int, str, int | None]]  # (gap, keyword, end gap, None for a jump)


def gold_items(tokens: list[str]) -> Gold:
    order, header_end, loop, keywords = [], {}, {}, []
    stack, last, loop_next = [], None, False
    for token in tokens:
        if (i := parse_ref(token)) is not None:
            loop[len(order)] = loop_next
            last, loop_next = ("header", len(order)), False
            order.append(i)
        elif token == LOOP:
            loop_next = True
        elif token == OPEN:
            stack.append(last)
        elif token == CLOSE:
            kind, at = stack.pop()
            if kind == "header":
                header_end[at] = len(order)
            else:
                keywords.append((at, kind, len(order)))
        elif token in SIMPLE_KEYWORDS:
            keywords.append((len(order), token, None))
        else:
            last = (token, len(order))
    return Gold(order, header_end, loop, sorted(keywords, key=lambda k: k[:2]))


def oracle_scores(gold: Gold) -> Scores:
    """Scores that make `gold` the best skeleton when the chart can write it, for tests and coverage."""
    n, bad = len(gold.order), -20.0
    s = Scores(np.full((n, n + 1), bad), np.full((n, 2), bad), np.full((n + 1, len(KEYWORDS)), bad), np.full((len(BLOCK_KEYWORDS), n + 1, n + 1), bad))
    for p, e in gold.header_end.items():
        s.header_end[p, e] = 0
    for p, is_loop in gold.loop.items():
        s.loop[p, int(is_loop)] = 0
    for g, kw, e in gold.keywords:
        s.insert[g, KEYWORDS.index(kw)] = -bad
        if e is not None:
            s.keyword_end[BLOCK_KEYWORDS.index(kw), g, e] = 0
    return s


def decode_order(successors: np.ndarray | None, shapes: list[Shape]) -> list[int]:
    """
    Source order from successor scores (row 0: the first statement, row i + 1: what follows statement i), greedily;
    the statement order without scores. Prefix statements come first either way.
    """
    n = len(shapes)
    order = [i for i in range(n) if shapes[i].prefix]
    if successors is None:
        return order + [i for i in range(n) if not shapes[i].prefix]
    left = np.ones(n, dtype=bool)
    left[order] = False
    last = order[-1] + 1 if order else 0
    while left.any():
        i = int(np.argmax(np.where(left, successors[last, :n], NONE)))
        order.append(i)
        left[i] = False
        last = i + 1
    return order


def view(m: np.ndarray, start: tuple[int, int], cell: tuple[int, int], split: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    """v[i, j] = m[start + i * cell + j * split] (row and column steps), as a read-only view of the matrix `m`."""
    s0, s1 = m.strides
    return np.lib.stride_tricks.as_strided(m[start[0] :, start[1] :], shape, (cell[0] * s0 + cell[1] * s1, split[0] * s0 + split[1] * s1), writeable=False)


def normal(frame: Frame) -> Frame:
    """A frame as a chart context: block counts are left to BlockGrammar.check, and only a match's kind matters."""
    return frame._replace(kind="match" if frame.kind == "match" else "block", chain=None, blocks=0, finallies=0)


class Chart:
    """
    The best skeletons for the statements of `grammar` in `order` (statement indices), by `scores`. Prefix statements
    (docstrings, `from __future__` imports) come first in the order. `keywords`: the keywords the model may write.
    """

    def __init__(self, grammar: BlockGrammar, order: list[int], scores: Scores, keywords: list[str]):
        self.grammar = grammar
        self.order = order
        self.scores = scores
        self.n = n = len(order)
        self.prefixes = next((p for p, i in enumerate(order) if not grammar.shapes[i].prefix), n)
        self.contexts: list[Frame] = [normal(ROOT)]
        self.items: list[list[Item]] = []
        while len(self.items) < len(self.contexts):  # contexts are added as their parents' items are made
            self.items.append(self.context_items(self.contexts[len(self.items)], keywords))
        # [nokw, context, a, b, chain]: the best item sequence over gaps a .. b that ends in the chain, before (pre) and
        # after (post) a trailing jump, and [nokw, context, a, b] the best that may end the block; nokw: no keyword
        # block at gap a. before[nokw, context, chains] holds the best post over the chains an item may follow.
        size = (2, len(self.contexts), n + 1, n + 1)
        self.pre = np.full((*size, len(CHAINS)), NONE, dtype=np.float32)
        self.post = np.full((*size, len(CHAINS)), NONE, dtype=np.float32)
        self.body = np.full(size, NONE, dtype=np.float32)
        self.before: dict[tuple, np.ndarray] = {}
        self.fill()
        self.derivations: dict[tuple, list] = {}  # k-best: derivations of a node found so far, best first
        self.candidates: dict[tuple, list] = {}
        self.seen: dict[tuple, set] = {}
        self.edge_lists: dict[tuple, list] = {}

    def context(self, frame: Frame) -> int:
        frame = normal(frame)
        if frame not in self.contexts:
            self.contexts.append(frame)
        return self.contexts.index(frame)

    def context_items(self, ctx: Frame, keywords: list[str]) -> list[Item]:
        g, n, s = self.grammar, self.n, self.scores
        groups: dict[tuple, list] = {}  # (kind, keyword, chain after, child, shape) -> chains it may follow

        by_shape: dict[Shape, list[int]] = {}
        for p in range(self.prefixes, n):
            by_shape.setdefault(g.shapes[self.order[p]], []).append(p)
        for c, chain in enumerate(CHAINS):
            top = ctx._replace(chain=chain)
            state = State(0, (top,))
            add = []
            for shape in by_shape:
                if shape.opener in ("cond", "while") and not shape.inline and ctx.kind != "match" and chain not in OBLIGATIONS:
                    add.append(("loop", None, "loop", self.context(g.open_frame("while", top)), shape))
                if shape.prefix or not g.can_place(state, shape):
                    continue
                kind = g.block_kind(shape, top) if shape.opener else None
                if shape.opener and not shape.inline:
                    add.append(("header", None, CHAIN_AFTER.get(kind), self.context(g.open_frame(kind, top)), shape))
                else:
                    add.append(("stmt", None, CHAIN_AFTER.get(kind) if shape.opener else "deco" if shape.decorator else None, -1, shape))
            for kw in keywords:
                shape = KEYWORD_SHAPES[kw]
                if g.can_place(state, shape, keyword=True):
                    kind = None if kw in SIMPLE_KEYWORDS else g.block_kind(shape, top)
                    add.append(("jump", kw, None, -1, None) if kind is None else ("keyword", kw, CHAIN_AFTER.get(kind), self.context(g.open_frame(kind, top)), None))
            for kind, kw, then, child, shape in add:
                groups.setdefault((kind, kw, CHAINS.index(then), child, shape), [False] * len(CHAINS))[c] = True
        # statements of different shapes that are the same item are one item
        merged: dict[tuple, list] = {}
        for (kind, kw, chain, child, shape), after in groups.items():
            positions = by_shape.get(shape, [])
            merged.setdefault((kind, kw, tuple(after), chain, child), []).extend(positions)
        items = []
        for (kind, kw, after, chain, child), positions in merged.items():
            positions = np.array(sorted(positions), dtype=int)
            if kind == "keyword":
                score = s.insert[:, KEYWORDS.index(kw), None] + s.keyword_end[BLOCK_KEYWORDS.index(kw)]
            elif kind == "jump":
                score = s.insert[:, KEYWORDS.index(kw)]
            else:
                score = np.full(n, NONE)
                score[positions] = self.loop_score(positions, kind)
                if kind != "stmt":
                    score = score[:, None] + s.header_end
            items.append(Item(kind, kw, after, chain, child, score.astype(np.float32), positions))
        return items

    def loop_score(self, positions: np.ndarray, kind: str) -> np.ndarray:
        """Score of the statements at `positions` as `kind`: only a cond header chooses between if and while."""
        if kind not in ("header", "loop"):
            return np.zeros(len(positions))
        cond = np.array([self.grammar.shapes[self.order[p]].opener == "cond" for p in positions], dtype=bool)
        return np.where(cond, self.scores.loop[positions, int(kind == "loop")], 0.0)

    def start_chain(self, x: int) -> int:
        return CHAINS.index("match_empty" if self.contexts[x].kind == "match" else None)

    def finish(self, nokw: int, x: int, a: np.ndarray, b: np.ndarray):
        """post, body and before from pre, for the cells (a, b) of context x."""
        pre = self.pre[nokw, x, a, b]
        post = pre.copy()
        for item in self.items[x]:
            if item.kind == "jump":
                best = np.where(item.after, pre, NONE).max(-1) + item.score[b]
                post[:, 0] = np.maximum(post[:, 0], best)
        self.post[nokw, x, a, b] = post
        self.body[nokw, x, a, b] = np.where(CLOSABLE, post, NONE).max(-1)
        for after in {item.after for item in self.items[x] if item.kind != "jump"}:
            table = self.before.setdefault((nokw, x, after), np.full((self.n + 1, self.n + 1), NONE, dtype=np.float32))
            table[a, b] = np.where(after, post, NONE).max(-1)

    def block_item(self, item: Item, nokw: int, a, k, b) -> tuple[np.ndarray, np.ndarray]:
        """Scores of a header or keyword `item` from position (or gap) k to gap b in a sequence from gap a: its own, and its block's."""
        if item.kind == "keyword":
            return np.where(k >= a + nokw, item.score[k, b], NONE), self.body[1, item.child][k, b]
        return item.score[k, b], self.body[0, item.child][k + 1, b]

    @staticmethod
    def block_node(item: Item, k: int, b: int) -> tuple:
        keyword = item.kind == "keyword"
        return ("body", int(keyword), item.child, k + (not keyword), b)

    def tables(self) -> list[tuple[int, int]]:
        """The (nokw, context) tables the root needs, nokw ones first: a keyword block over a whole span needs them."""
        need, todo = {(0, 0)}, [(0, 0)]
        while todo:
            for item in self.items[todo.pop()[1]]:
                table = (int(item.kind == "keyword"), item.child)
                if item.child >= 0 and table not in need:
                    need.add(table)
                    todo.append(table)
        return sorted(need, key=lambda t: (-t[0], t[1]))

    def fill(self):
        n = self.n
        tables = self.tables()
        cells = np.arange(n + 1)
        for nokw, x in tables:
            self.pre[nokw, x, cells, cells, self.start_chain(x)] = 0
            self.finish(nokw, x, cells, cells)
        for w in range(1, n + 1):
            a = np.arange(n - w + 1)
            b = a + w
            for nokw, x in tables:
                new = np.full((len(a), len(CHAINS)), NONE, dtype=np.float32)
                for item in self.items[x]:
                    if item.kind == "jump":
                        continue
                    before = self.before[nokw, x, item.after]
                    if item.kind == "stmt":
                        score = before[a, b - 1] + item.score[b - 1]
                    elif item.kind == "keyword" or len(item.positions) >= w:
                        # every split j: before[a, a + j] + score[a + j, b] + body[a + j (+ 1), b], as strided views
                        skip = int(nokw and item.kind == "keyword")  # no keyword block at gap a
                        shape = (len(a), w - skip)
                        body = self.body[int(item.kind == "keyword"), item.child]
                        score = view(before, (0, skip), (1, 1), (0, 1), shape) + view(item.score, (skip, w), (1, 1), (1, 0), shape)
                        score = (score + view(body, (skip + (item.kind != "keyword"), w), (1, 1), (1, 0), shape)).max(1) if w > skip else np.full(len(a), NONE)
                    else:  # only the splits at the item's statements
                        k = np.broadcast_to(item.positions, (len(a), len(item.positions)))
                        k = np.where((k >= a[:, None]) & (k < b[:, None]), k, -1)
                        own, body = self.block_item(item, nokw, a[:, None], k, b[:, None])
                        score = (np.where(k >= 0, before[a[:, None], k], NONE) + own + body).max(1)
                    new[:, item.chain] = np.maximum(new[:, item.chain], score)
                self.pre[nokw, x, a, b] = new
                self.finish(nokw, x, a, b)

    # k-best over the nodes ("pre" | "post", nokw, context, a, b, chain) and ("body", nokw, context, a, b)

    def value(self, node: tuple) -> float:
        table, *index = node
        return float({"pre": self.pre, "post": self.post, "body": self.body}[table][tuple(index)])

    def edges(self, node: tuple, limit: int) -> list[tuple[float, float, tuple, tuple]]:
        """The `limit` best ways to make `node`: (best score, own score, tails, label), best first."""
        table, nokw, x, a, b, *chain = node
        out = []
        if table == "body":
            out = [(0.0, (("post", nokw, x, a, b, int(c)),), ("body",)) for c in np.flatnonzero(CLOSABLE)]
        elif table == "post":
            out = [(0.0, (("pre", nokw, x, a, b, chain[0]),), ("post",))]
            for item in self.items[x] if chain[0] == 0 else []:
                if item.kind == "jump":
                    out += [(float(item.score[b]), (("pre", nokw, x, a, b, int(c)),), ("jump", item.keyword)) for c in np.flatnonzero(item.after)]
        elif a == b:
            out = [(0.0, (), ("leaf",))] if chain[0] == self.start_chain(x) else []
        else:
            scored = []
            for item in self.items[x]:
                if item.chain != chain[0] or item.kind == "jump":
                    continue
                if item.kind == "stmt":
                    k = np.array([b - 1])
                    own, body = item.score[k], np.zeros(1)
                else:
                    k = np.arange(a, b)
                    own, body = self.block_item(item, nokw, a, k, b)
                total = (own + body)[:, None] + self.post[nokw, x, a, k] + np.where(item.after, 0, NONE)
                for i, c in zip(*np.unravel_index(np.argsort(-total, axis=None)[:limit], total.shape)):
                    if total[i, c] > NONE:
                        tails = (("post", nokw, x, a, int(k[i]), int(c)),) + ((self.block_node(item, int(k[i]), b),) if item.kind != "stmt" else ())
                        scored.append((float(total[i, c]), float(own[i]), tails, (item.kind, item.keyword, int(k[i]))))
            return heapq.nlargest(limit, scored, key=lambda e: e[0])
        scored = [(own + sum(self.value(t) for t in tails), own, tails, label) for own, tails, label in out]
        return heapq.nlargest(limit, [e for e in scored if e[0] > NONE], key=lambda e: e[0])

    def kth(self, node: tuple, k: int, limit: int) -> tuple | None:
        """The k-th best derivation of `node` as (score, edge, ranks of its tails), or None."""
        if node not in self.derivations:
            self.edge_lists[node] = edges = self.edges(node, limit)
            self.derivations[node] = []
            self.candidates[node] = [(-score, i, (0,) * len(tails)) for i, (score, _, tails, _) in enumerate(edges)]
            self.seen[node] = {(i, ranks) for _, i, ranks in self.candidates[node]}
        found, heap, seen, edges = self.derivations[node], self.candidates[node], self.seen[node], self.edge_lists[node]
        while len(found) <= k:
            if found:
                _, i, ranks = found[-1]
                _, own, tails, _ = edges[i]
                for t in range(len(tails)):
                    successor = ranks[:t] + (ranks[t] + 1,) + ranks[t + 1 :]
                    if (i, successor) in seen:
                        continue
                    subs = [self.kth(tail, r, limit) for tail, r in zip(tails, successor)]
                    if all(subs):
                        seen.add((i, successor))
                        heapq.heappush(heap, (-(own + sum(d[0] for d in subs)), i, successor))
            if not heap:
                return None
            score, i, ranks = heapq.heappop(heap)
            found.append((-score, i, ranks))
        return found[k]

    def tokens(self, node: tuple, k: int, limit: int) -> list[str]:
        _, i, ranks = self.kth(node, k, limit)
        _, _, tails, label = self.edge_lists[node][i]
        parts = [self.tokens(tail, r, limit) for tail, r in zip(tails, ranks)]
        kind = label[0]
        if kind in ("body", "post", "leaf"):
            return parts[0] if parts else []
        if kind == "jump":
            return parts[0] + [label[1]]
        statement = ref(self.order[label[2]])
        head = {"stmt": [statement], "header": [statement], "loop": [LOOP, statement], "keyword": [label[1]]}[kind]
        return parts[0] + head + ([OPEN, *parts[1], CLOSE] if kind != "stmt" else [])

    def best(self, limit: int) -> Iterator[tuple[float, list[str]]]:
        """Up to `limit` best skeletons with their scores, best first."""
        root = ("body", 0, 0, self.prefixes, self.n)
        prefix = [ref(i) for i in self.order[: self.prefixes]]
        for r in range(limit):
            recursion = sys.getrecursionlimit()
            sys.setrecursionlimit(max(recursion, 40 * self.n + 1000))  # derivations nest about once per item
            try:
                d = self.kth(root, r, limit)
                tokens = d and prefix + self.tokens(root, r, limit)
            finally:
                sys.setrecursionlimit(recursion)
            if d is None:
                return
            yield d[0], tokens
