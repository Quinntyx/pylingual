import sys

import pytest

from pylingual.control_flow_reconstruction.source import SourceContext
from pylingual.control_flow_reconstruction.structure import bc_to_cft
from pylingual.control_flow_reconstruction.templates.Skeleton import SkeletonTemplate
from pylingual.decompiler import Decompiler, has_comp_error
from pylingual.models import ControlFlowPredictor
from pylingual.utils.version import PythonVersion

from test_cflow_model_view import compile_marked

VERSION = PythonVersion(sys.version_info[:2])
SOURCE = "def f(a):\n    if a:\n        print(1)\n    else:\n        print(2)\n    print(3)\n"
FLAT = "<s0> <s1> <s2> <s3>"
NESTED = "<s0> { <s1> <s2> } <s3>"
RIGHT = "<s0> { <s1> } else { <s2> } <s3>"
# g's line is mistranslated, and f's skeleton puts a bare except before `except ValueError:`
SOURCE_TRY = "def g(a):\n    print(a)\n\n\ndef f(a):\n    try:\n        print(1)\n    except ValueError:\n        print(2)\n"
EXCEPT_FIRST = "<s0> { <s1> } except { <s3> } <s2> { }"


class StubPredictor(ControlFlowPredictor):
    """ControlFlowPredictor with canned candidates, best first, instead of a model."""

    def __init__(self, candidates: list[str], error: Exception | None = None):
        self.candidates = [x.split() for x in candidates]
        self.error = error
        self.max_stmts, self.max_input_tokens = 512, 2048
        self.beams = []

    def _generate(self, batch, num_beams):
        if self.error:
            raise self.error
        self.beams.append(num_beams)
        return [self.candidates[:num_beams] for _ in batch]


def decompile_with(predictor: ControlFlowPredictor | None, tmp_path, source=SOURCE, mistranslated: dict[int, str] | None = None) -> Decompiler:
    """Runs the control flow phases and correction on `source`, with the source as the translation but for `mistranslated` lines."""
    src = tmp_path / "f.py"
    src.write_text(source)
    pyc, lines = compile_marked(src, tmp_path, VERSION)
    d = Decompiler(pyc, None, None, VERSION, cflow_model=predictor)
    d.tmp = tmp_path
    d.ordered_bytecodes = list(pyc.iter_bytecodes())
    d.source_lines = [(mistranslated or {}).get(n, x) for n, x in enumerate(lines, start=1)]
    d.segmentation_results = []  # leaves correct_segmentation nothing to try
    d.run_cflow_prediction()
    d.run_cflow_reconstruction()
    d.reconstruct_source()
    d.equivalence_results = d.check_reconstruction(d.indented_source)
    d.correct_failures()
    if has_comp_error(d.equivalence_results):
        d.equivalence_results += d.purge_comp_errors()
    return d


def cft_of_f(d: Decompiler):
    return next(cft for co, cft in d.source_context.cfts.items() if co.co_name == "f")


def test_beam_fixes_greedy_skeleton(tmp_path):
    predictor = StubPredictor([NESTED, FLAT, RIGHT])
    d = decompile_with(predictor, tmp_path)
    assert predictor.beams == [1, 4]  # beam search only for the failing code object
    assert isinstance(cft_of_f(d), SkeletonTemplate) and cft_of_f(d).tokens == RIGHT.split()
    assert d.equivalence_results and all(r.success for r in d.equivalence_results)


def f_index(d: Decompiler) -> int:
    return [bc.codeobj.co_name for bc in d.ordered_bytecodes].index("f")


def in_order(d: Decompiler) -> list[str]:
    grammar = d.cflow_requests[f_index(d)].grammar()
    return grammar.complete(grammar.start())


def test_best_skeleton_stays_after_failing_beams(tmp_path):
    """No template fallback: when no beam passes, the greedy skeleton, which fits the grammar, stays."""
    predictor = StubPredictor([NESTED, FLAT])
    d = decompile_with(predictor, tmp_path)
    assert predictor.beams == [1, 4]
    assert isinstance(cft_of_f(d), SkeletonTemplate) and cft_of_f(d).tokens == NESTED.split()
    assert {r.bc_a.name: r.success for r in d.equivalence_results} == {"<module>": True, "<module>.f": False}


def test_skeletons_outside_the_grammar_give_statement_order(tmp_path):
    # FLAT leaves the if header without a block, EXCEPT_FIRST puts `except ValueError:` after a bare except
    for candidates, source in (([FLAT], SOURCE), ([EXCEPT_FIRST], SOURCE_TRY)):
        d = decompile_with(StubPredictor(candidates), tmp_path, source)
        assert d.cflow_results[f_index(d)] == [in_order(d)]
        assert d.cflow_requests[f_index(d)].grammar().accepts(in_order(d))


def test_statement_order_after_purging_another_code_object(tmp_path):
    # f's greedy skeleton is outside the grammar, so f keeps its statements in order (which compile) while g is purged
    d = decompile_with(StubPredictor([EXCEPT_FIRST]), tmp_path, SOURCE_TRY, {2: "1 = a"})
    assert isinstance(cft_of_f(d), SkeletonTemplate) and cft_of_f(d).tokens == in_order(d)
    results = {r.bc_a.name: r.success for r in d.equivalence_results if not isinstance(r, Exception)}
    assert results["<module>"] and not results["<module>.g"] and "<module>.f" in results


def test_model_errors_give_statement_order(tmp_path):
    d = decompile_with(StubPredictor([RIGHT], error=RuntimeError("boom")), tmp_path)
    assert isinstance(cft_of_f(d), SkeletonTemplate) and cft_of_f(d).tokens == in_order(d)
    with pytest.raises(TimeoutError):
        decompile_with(StubPredictor([RIGHT], error=TimeoutError()), tmp_path)


def test_no_model_uses_templates_only(tmp_path):
    d = decompile_with(None, tmp_path)
    assert d.cflow_requests == [None, None] and d.cflow_results == [None, None]
    templates = SourceContext(d.pyc, d.source_lines, {bc.codeobj: bc_to_cft(bc, d.source_lines) for bc in d.ordered_bytecodes})
    templates.header_lines = d.source_context.header_lines
    assert str(d.source_context) == str(templates)
