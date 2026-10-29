import pytest
import torch

from dataset_generation.bytecode2cflow import make_examples
from pylingual.control_flow_reconstruction.model_view import INPUT_TOKENS, CflowRequest
from pylingual.control_flow_reconstruction.templates.Skeleton import SkeletonTemplate
from pylingual.control_flow_reconstruction.transition import SPECIAL, TransitionConfig, TransitionModel
from pylingual.editable_bytecode import PYCFile
from pylingual.models import PREDICTORS, TransitionPredictor
from pylingual.utils.version import PythonVersion

from test_cflow_dataset import VERSIONS, compiled_fixture
from test_cflow_decompiler import cft_of_f, decompile_with


def tiny_model(seed=0) -> TransitionModel:
    torch.manual_seed(seed)
    return TransitionModel(TransitionConfig(vocab=[*SPECIAL, *INPUT_TOKENS, "K_stmt", "RETURN_CONST"], d_model=32, layers=1, heads=2, d_ff=64)).eval()


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("fixture", ["Conditional", "Exception", "Loop", "Match", "With"])
def test_random_model_skeletons_are_accepted(fixture, version, tmp_path):
    """An untrained model's skeletons for real code objects are accepted by their grammar, distinct, and k at most."""
    src, pyc_path = compiled_fixture(fixture, PythonVersion(version), tmp_path)
    requests = [CflowRequest.from_row(e) for e in make_examples(PYCFile(str(pyc_path)), src)]
    predictor = TransitionPredictor(tiny_model())
    for k in (1, 3):
        results = predictor(requests, num_beams=k)
        assert all(results), "every request gets a skeleton"
        for request, skeletons in zip(requests, results):
            assert all(request.grammar().accepts(s) for s in skeletons)
            assert 1 <= len(skeletons) <= k and len({tuple(s) for s in skeletons}) == len(skeletons)


def test_save_and_load(tmp_path):
    model = tiny_model(seed=1)
    model.config.max_stmts = 100
    model.save_pretrained(tmp_path)
    loaded = PREDICTORS["transition"].from_pretrained(str(tmp_path))
    assert loaded.max_stmts == 100 and loaded.model.config.vocab == model.config.vocab
    request = CflowRequest.from_row({"input": "| <s0> K_cond POP_JUMP_IF_FALSE f+ <s2> | <s1> K_stmt | <s2> K_stmt RETURN_CONST nf", "shapes": "cond stmt stmt", "version": "3.12"})
    assert loaded([request], num_beams=4) == TransitionPredictor(model)([request], num_beams=4)


def test_decompiler_with_transition_model(tmp_path):
    """A CONTROL_FLOW_MODEL with ARCH transition: the decompiler takes its skeletons, then the templates if they fail."""
    model_dir = tmp_path / "model"
    tiny_model().save_pretrained(model_dir)
    config = {"REPO": str(model_dir), "REVISION": "main", "ARCH": "transition"}
    predictor = PREDICTORS[config["ARCH"]].from_pretrained(config["REPO"], revision=config["REVISION"])
    d = decompile_with(predictor, tmp_path)
    f = [bc.codeobj.co_name for bc in d.ordered_bytecodes].index("f")
    assert d.cflow_results[f] and all(d.cflow_requests[f].grammar().accepts(s) for s in d.cflow_results[f])
    assert d.equivalence_results and all(r.success for r in d.equivalence_results)
    assert not isinstance(cft_of_f(d), SkeletonTemplate) or d.cflow_requests[f].grammar().accepts(cft_of_f(d).tokens)
