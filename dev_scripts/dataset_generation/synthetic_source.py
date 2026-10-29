# /// script
# requires-python = ">= 3.12"
# dependencies = [
#   "pylingual",
# ]
# [tool.uv.sources]
# pylingual = { path = "../../", editable = true }
# ///
"""
Random Python programs with nested if / loops / try / with / match, for the control flow model dataset.

Real code rarely nests some statements (a for/else inside an except*), so `generate` forces every chain of one or two
nested compound kinds, and a sample of three, into some program; the rest of each program is random.
"""

from __future__ import annotations

import itertools
import pathlib
import random
from dataclasses import dataclass, replace

import click

from pylingual.utils.version import PythonVersion

NAMES = ["a", "b", "c", "x", "y", "z", "i", "n", "data", "items", "value", "result", "obj", "key", "flag"]
FUNCS = ["f", "g", "h", "check", "process", "log", "compute", "handle"]
EXCS = ["ValueError", "KeyError", "TypeError", "IndexError", "OSError", "RuntimeError", "Exception"]

# max random nesting depth
MAX_DEPTH = 3

# compound statement kinds; version-gated ones are filtered in `kinds_for`
KINDS = [
    "if", "if_else", "if_elif", "if_elif_else",
    "for", "for_else", "while", "while_else", "while_true",
    "try_except", "try_except_as", "try_multi_except", "try_bare", "try_except_else",
    "try_finally", "try_except_finally", "try_except_else_finally",
    "with", "with_as", "with_multi", "match", "try_star",
]  # fmt: skip


def kinds_for(version: tuple[int, int]) -> list[str]:
    ks = list(KINDS)
    if version < (3, 10):
        ks.remove("match")
    if version < (3, 11):
        ks.remove("try_star")
    return ks


@dataclass(frozen=True)
class Ctx:
    """Where a statement is generated, which decides what may appear there (break only in loops, ...)."""

    depth: int = 0
    in_loop: bool = False
    in_func: bool = False
    is_async: bool = False
    is_gen: bool = False
    in_finally: bool = False
    in_star: bool = False
    # small functions use shorter, flatter blocks
    compound_p: float = 0.35
    block_size: tuple[int, int] = (1, 3)


class Gen:
    """Builds programs as lists of source lines; all choices come from `rng`, so a seed gives the same programs."""

    def __init__(self, rng: random.Random, version: tuple[int, int]):
        self.rng = rng
        self.version = version
        self.kinds = kinds_for(version)

    # expressions

    def name(self) -> str:
        return self.rng.choice(NAMES)

    def call(self) -> str:
        args = ", ".join(self.name() for _ in range(self.rng.randint(0, 2)))
        return f"{self.rng.choice(FUNCS)}({args})"

    def atom(self) -> str:
        r = self.rng.random()
        if r < 0.3:
            return self.name()
        if r < 0.5:
            return self.call()
        if r < 0.65:
            return f"{self.name()}.{self.rng.choice(['ready', 'done', 'value', 'size'])}"
        return f"{self.name()} {self.rng.choice(['>', '<', '==', '!=', '>=', 'is', 'is not', 'in', 'not in'])} {self.rng.choice([self.name(), '0', '1', 'None', '10'])}"

    def cond(self) -> str:
        r = self.rng.random()
        if r < 0.55:
            return self.atom()
        if r < 0.65:
            return f"not {self.atom()}"
        if r < 0.8:
            return f"{self.atom()} and {self.atom()}"
        if r < 0.92:
            return f"{self.atom()} or {self.atom()}"
        if r < 0.96 and self.version >= (3, 8):
            return f"({self.name()} := {self.call()})"
        return f"{self.atom()} and ({self.atom()} or {self.atom()})"

    def simple(self, ctx: Ctx) -> str:
        """A simple statement: plain, or with control flow inside the line (ternary, and / or, comprehension, assert)."""
        n, r = self.name(), self.rng.random()
        if ctx.is_async and r < 0.1:
            return f"{n} = await {self.call()}"
        if ctx.is_gen and r < 0.15:
            return f"yield {self.name()}"
        choices = [
            lambda: f"{n} = {self.call()}",
            lambda: f"{n} += {self.rng.randint(1, 9)}",
            lambda: self.call(),
            lambda: f"{n} = {self.atom()} if {self.atom()} else {self.atom()}",
            lambda: f"{n} = {self.atom()} {self.rng.choice(['and', 'or'])} {self.atom()}",
            lambda: f"{n} = [{self.name()} for {self.name()} in {self.name()}]",
            lambda: f"assert {self.atom()}",
            lambda: f"del {n}",
            lambda: "pass",
        ]
        return self.rng.choice(choices)()

    def terminator(self, ctx: Ctx) -> str | None:
        """A block-ending statement valid in `ctx`."""
        if ctx.in_finally:
            return None
        opts = []
        if ctx.in_loop and not ctx.in_star:
            opts += ["break", "continue"]
        if ctx.in_func and not ctx.in_star:
            opts += ["return"] if ctx.is_gen and ctx.is_async else ["return", f"return {self.name()}", f"return {self.call()}"]
        opts += [f"raise {self.rng.choice(EXCS)}({self.name()})"]
        return self.rng.choice(opts)

    # statements

    def block(self, ctx: Ctx, forced: list[str] | None = None, size: tuple[int, int] | None = None) -> list[str]:
        """Lines (indented one level) of a block. `forced`: compound kinds to nest inside it, outermost first."""
        items: list[str | tuple] = []
        for _ in range(self.rng.randint(*(size or ctx.block_size))):
            if ctx.depth < MAX_DEPTH and self.rng.random() < ctx.compound_p:
                items.append(("compound", self.rng.choice(self.kinds), None))
            else:
                items.append(("simple",))
        if forced:
            items.insert(self.rng.randint(0, len(items)), ("compound", forced[0], forced[1:]))
        lines: list[str] = []
        for it in items:
            if it[0] == "simple":
                lines.append(self.simple(ctx))
            else:
                lines += self.compound(it[1], ctx, it[2])
        if self.rng.random() < 0.2 and (t := self.terminator(ctx)):
            lines.append(t)
        return ["    " + x for x in lines]

    def compound(self, kind: str, ctx: Ctx, forced: list[str] | None = None) -> list[str]:
        inner = replace(ctx, depth=ctx.depth + 1)
        loop = replace(inner, in_loop=True)
        rng = self.rng
        # the clauses are chosen first, so the rest of the forced chain goes into exactly one of the bodies
        n_elif = rng.randint(1, 3) if "elif" in kind else 0
        cases = []
        if kind == "match":
            cases = rng.sample(["0", "1 | 2", "[x, y]", "{'k': v}", "Point(x=0)", "str()", '"a"'], rng.randint(1, 3)) + (["_"] if rng.random() < 0.5 else [])
        if kind == "try_star":
            handlers = [f"except* {e} as {self.name()}" for e in rng.sample(EXCS, rng.randint(1, 2))]
        elif kind == "try_bare":
            handlers = ["except"]
        elif kind == "try_multi_except":
            handlers = [f"except {e}{' as ' + self.name() if rng.random() < 0.5 else ''}" for e in rng.sample(EXCS, rng.randint(2, 3))]
            handlers += ["except"] if rng.random() < 0.3 else []
        elif kind.startswith("try") and kind != "try_finally":
            e = rng.choice(EXCS) if rng.random() < 0.8 else f"({', '.join(rng.sample(EXCS, 2))})"
            handlers = [f"except {e}{' as ' + self.name() if kind == 'try_except_as' or rng.random() < 0.4 else ''}"]
        else:
            handlers = []
        if kind.startswith("if"):
            n_bodies = 1 + n_elif + kind.endswith("else")
        elif kind == "match":
            n_bodies = len(cases)
        elif kind.startswith("try"):
            n_bodies = 1 + len(handlers) + ("else" in kind) + ("finally" in kind)
        else:
            n_bodies = 1 + kind.endswith("_else")
        slot = rng.randrange(n_bodies)
        bodies = itertools.count()

        def body(c: Ctx) -> list[str]:
            return self.block(c, forced if next(bodies) == slot else None)

        if kind.startswith("if"):
            out = [f"if {self.cond()}:", *body(inner)]
            for _ in range(n_elif):
                out += [f"elif {self.cond()}:", *body(inner)]
            if kind.endswith("else"):
                out += ["else:", *body(inner)]
            return out
        if kind.startswith("for"):
            head = "async for" if ctx.is_async and rng.random() < 0.3 else "for"
            target = f"{self.name()}, {self.name()}" if rng.random() < 0.2 else self.name()
            out = [f"{head} {target} in {rng.choice([self.name(), self.call(), 'range(10)', '[1, 2, 3]'])}:", *body(loop)]
            if kind == "for_else":
                out += ["else:", *body(inner)]
            return out
        if kind == "while_true":
            b = body(loop)
            # always exit the loop so the code after it is reachable
            brk = [f"    if {self.cond()}:", "        break"]
            ends = b[-1].strip().split(" ")[0] in ("return", "raise", "continue")
            b = brk + b if ends or rng.random() < 0.5 else b + brk
            return ["while True:", *b]
        if kind.startswith("while"):
            out = [f"while {self.cond()}:", *body(loop)]
            if kind == "while_else":
                out += ["else:", *body(inner)]
            return out
        if kind.startswith("with"):
            head = "async with" if ctx.is_async and rng.random() < 0.3 else "with"
            items = {"with": self.call(), "with_as": f"{self.call()} as {self.name()}", "with_multi": f"{self.call()} as {self.name()}, {self.call()}"}[kind]
            return [f"{head} {items}:", *body(inner)]
        if kind == "match":
            out = [f"match {self.name()}:"]
            for pat in cases:
                guard = f" if {self.atom()}" if pat != "_" and rng.random() < 0.2 else ""
                out += [f"    case {pat}{guard}:", *["    " + x for x in body(inner)]]
            return out
        # try statements
        out = ["try:", *body(inner)]
        for h in handlers:
            out += [f"{h}:", *body(replace(inner, in_star=True) if kind == "try_star" else inner)]
        if "else" in kind:
            out += ["else:", *body(inner)]
        if "finally" in kind:
            out += ["finally:", *body(replace(inner, in_finally=True))]
        return out

    def function(self, forced: list[str] | None = None) -> list[str]:
        r = self.rng.random()
        is_async = r < 0.15
        is_gen = 0.15 <= r < 0.25 or (is_async and self.rng.random() < 0.2)
        ctx = Ctx(in_func=True, is_async=is_async, is_gen=is_gen)
        head = f"{'async ' if is_async else ''}def {self.rng.choice(FUNCS)}_{self.rng.randint(0, 999)}({', '.join(self.rng.sample(NAMES, self.rng.randint(0, 3)))}):"
        if self.rng.random() < 0.4:
            # small function: one compound statement with short bodies, maybe a statement before/after it
            body = self.block(replace(ctx, compound_p=0.15, block_size=(1, 2)), forced or [self.rng.choice(self.kinds)], size=(0, 1))
        else:
            body = self.block(ctx, forced, size=(1, 4))
        if self.rng.random() < 0.3 and not (is_gen and is_async):
            body.append(f"    return {self.name()}")
        return [head, *body]

    def program(self, forced: list[str] | None = None) -> str:
        """A module of functions (some inside a class) and maybe module level code; `forced` goes into one of them."""
        n_funcs = self.rng.randint(5, 20) if self.rng.random() < 0.2 else self.rng.randint(1, 4)
        module_code = self.rng.random() < 0.5
        at = self.rng.randrange(n_funcs + module_code) if forced else -1
        lines: list[str] = []
        for k in range(n_funcs):
            fn = self.function(forced if k == at else None)
            if self.rng.random() < 0.15:
                lines += [f"class C{self.rng.randint(0, 99)}:", *["    " + x for x in fn]]
            else:
                lines += fn
            lines.append("")
        # module level code: no return
        if module_code:
            lines += [x[4:] for x in self.block(Ctx(), forced if at == n_funcs else None, size=(1, 3))]
        return "\n".join(lines) + "\n"


def generate(n: int, version: tuple[int, int], seed: int = 0) -> list[str]:
    """`n` programs: first all depth-1/2 nestings (and a sample of depth-3), then random programs."""
    rng = random.Random(f"{seed}-{version}")
    gen = Gen(rng, version)
    kinds = kinds_for(version)
    forced = [list(c) for d in (1, 2) for c in itertools.product(kinds, repeat=d)]
    triples = [list(c) for c in itertools.product(kinds, repeat=3)]
    forced += rng.sample(triples, min(len(triples), n // 4))
    programs = [gen.program(forced.pop() if forced and rng.random() < 0.7 else None) for _ in range(n)]
    rng.shuffle(programs)
    return programs


@click.command(help="Write synthetic programs as <out>/synth_<k>/program.py, a source_path for a DataRequest.")
@click.option("-o", "--out-dir", type=pathlib.Path, required=True)
@click.option("-n", "--num-programs", type=int, required=True)
@click.option("-v", "--version", type=PythonVersion, required=True, help="Python version the programs must compile with.")
@click.option("-s", "--seed", type=int, default=0)
def main(out_dir: pathlib.Path, num_programs: int, version: PythonVersion, seed: int):
    for k, src in enumerate(generate(num_programs, version.as_tuple(), seed)):
        path = out_dir / f"synth_{k:06d}" / "program.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src)


if __name__ == "__main__":
    main()
