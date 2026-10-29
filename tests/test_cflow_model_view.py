import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from dataset_generation.bytecode2cflow import assign_lines
from pylingual.control_flow_reconstruction.model_view import encode_code_object, split_statements
from pylingual.control_flow_reconstruction.skeleton import BLOCK_KEYWORDS, CLOSE, KEYWORD_TEXT, LOOP, OPEN, PLACE, SkeletonGrammar, from_model_order, parse_ref, ref, to_model_order
from pylingual.control_flow_reconstruction.source import SourceContext
from pylingual.control_flow_reconstruction.structure import bc_to_cft
from pylingual.control_flow_reconstruction.templates.Skeleton import SkeletonTemplate, parse_skeleton
from pylingual.editable_bytecode import PYCFile
from pylingual.utils.generate_bytecode import _uv_python, compile_version
from pylingual.utils.version import PythonVersion

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).parent / "data" / "cflow_golden.json"
FIXTURES = ["Conditional", "Exception", "Generator", "Loop", "Match", "With"]


def compile_marked(src: Path, tmp_path: Path, version: PythonVersion) -> tuple[PYCFile, list[str]]:
    """Compiled `src` with line starts marked as the segmentation model would give them."""
    pyc_path = tmp_path / f"{src.stem}.pyc"
    compile_version(src, pyc_path, version)
    pyc = PYCFile(str(pyc_path))
    lines = src.read_text().split("\n")
    assign_lines(pyc, lines)
    return pyc, lines


def encode_fixture(name: str, tmp_path: Path) -> list[dict]:
    pyc, lines = compile_marked(ROOT / "test" / f"{name}.py", tmp_path, PythonVersion("3.12"))
    out = []
    for bc in pyc.iter_bytecodes():
        if not bc.is_comprehension:
            stmts, tokens = encode_code_object(bc, lines)
            out.append({"name": bc.name, "n_stmts": len(stmts), "tokens": " ".join(tokens)})
    return out


def test_golden_encoding(tmp_path):
    if _uv_python(PythonVersion("3.12")) is None and sys.version_info[:2] != (3, 12):
        pytest.skip("Python 3.12 not available")
    golden = json.loads(GOLDEN.read_text())
    assert {name: encode_fixture(name, tmp_path) for name in FIXTURES} == golden


def test_rerender_after_update_lines(tmp_path):
    src = tmp_path / "f.py"
    src.write_text("def f(a):\n    if a:\n        print(1)\n    else:\n        print(2)\n    print(3)\n")
    pyc, lines = compile_marked(src, tmp_path, PythonVersion(sys.version_info[:2]))
    cfts = {bc.codeobj: bc_to_cft(bc, lines) for bc in pyc.iter_bytecodes()}
    f = next(bc for bc in pyc.iter_bytecodes() if bc.codeobj.co_name == "f")
    stmts = split_statements(f)
    assert len(stmts) == 4
    cfts[f.codeobj] = SkeletonTemplate(f, stmts, "<s0> { <s1> } else { <s2> } <s3>".split())
    ctx = SourceContext(pyc, lines, cfts)
    assert "    else:\n        print(2)" in str(ctx)
    snapshot = ctx.snapshot()
    ctx.update_lines([x.replace("print(2)", "print(5)") for x in lines])
    assert "    else:\n        print(5)" in str(ctx)
    skeleton = ctx.cfts[f.codeobj]
    ctx.update_cft(f, bc_to_cft(f, lines))
    ctx.restore(snapshot)
    assert ctx.cfts[f.codeobj] is skeleton and "    else:\n        print(2)" in str(ctx)


def test_model_order_round_trip():
    for skeleton in ["try { <s0> } <s1> { loop <s2> { <s3> } } else { break } finally { <s4> }", "<s0> { <s1> } <s2>", "} else { loop <s7> {"]:
        assert from_model_order(to_model_order(skeleton.split())) == skeleton.split()
    assert to_model_order("} else { loop <s7> {".split()) == "<s7> } else { loop = {".split()


def test_parse_skeleton_tolerates_bad_tokens():
    tree = parse_skeleton("<s0> { <s1> <s1> <s9> } } else { <s2> loop".split(), 3)
    assert [n.sid for n in tree] == [0, None]
    assert [n.sid for n in tree[0].children] == [1]
    assert tree[1].keyword == "else"
    assert [n.sid for n in tree[1].children] == [2]


class Vocab:
    """The part of a tokenizer SkeletonGrammar uses."""

    eos_token_id = 1

    def __init__(self, max_stmts: int):
        self.tokens = ["<pad>", "</s>", "<unk>", OPEN, CLOSE, LOOP, PLACE, *KEYWORD_TEXT, *map(ref, range(max_stmts))]

    def get_vocab(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.tokens)}


def test_grammar_random_walk():
    """Random walks through the grammar only give skeletons that place every statement once in balanced blocks."""
    vocab = Vocab(6)
    rng = random.Random(0)

    def check_blocks(nodes):
        for node in nodes:
            assert node.keyword not in BLOCK_KEYWORDS or node.children, tokens  # keyword blocks are never empty
            check_blocks(node.children or [])

    for _ in range(300):
        n = rng.randint(1, 6)
        # without break / continue, a keyword block opened after the last statement would have no way to close
        keywords = rng.choice([list(KEYWORD_TEXT), ["else", "try", "except", "break"], ["else", "try", "except"]])
        grammar = SkeletonGrammar(vocab, 6, keywords)
        prefix = (0,)
        for _ in range(2000):
            row = np.zeros(len(vocab.tokens), dtype=bool)
            s = grammar.state(prefix)
            grammar.fill(row, s, n)
            if rng.random() < 0.3:
                grammar.finish_placing(row, s, n, vocab.eos_token_id)
            allowed = np.flatnonzero(row).tolist()
            # keywords are rare, so the walk ends instead of writing keyword blocks forever
            prefix += tuple(rng.choices(allowed, [0.05 if vocab.tokens[t] in BLOCK_KEYWORDS else 0.2 if vocab.tokens[t] in KEYWORD_TEXT else 1 for t in allowed]))
            if prefix[-1] == vocab.eos_token_id:
                break
        assert prefix[-1] == vocab.eos_token_id
        tokens = from_model_order([vocab.tokens[t] for t in prefix[1:-1]])
        assert sorted(i for t in tokens if (i := parse_ref(t)) is not None) == list(range(n)), tokens
        assert not set(tokens) & (set(KEYWORD_TEXT) - set(keywords)), tokens
        depth = 0
        for t, after in zip(tokens, tokens[1:] + [None]):
            depth += (t == OPEN) - (t == CLOSE)
            assert depth >= 0, tokens
            assert t not in BLOCK_KEYWORDS or after == OPEN, tokens
            assert t != LOOP or parse_ref(after) is not None, tokens
        assert depth == 0, tokens
        check_blocks(parse_skeleton(tokens, n))
