import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dataset_generation.bytecode2cflow import make_examples
from pylingual.control_flow_reconstruction.model_view import INPUT_TOKENS, CflowRequest
from pylingual.control_flow_reconstruction.templates.Skeleton import SkeletonTemplate
from pylingual.control_flow_reconstruction.transition import SPECIAL, TransitionConfig, TransitionModel
from pylingual.decompiler import has_comp_error
from pylingual.editable_bytecode import PYCFile
from pylingual.models import PREDICTORS, TransitionPredictor, control_flow_config
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
    assert loaded.model.config.max_stmts == 100 and loaded.model.config.vocab == model.config.vocab
    request = CflowRequest.from_row({"input": "| <s0> K_cond POP_JUMP_IF_FALSE f+ <s2> | <s1> K_stmt | <s2> K_stmt RETURN_CONST nf", "shapes": "cond stmt stmt", "version": "3.12"})
    assert loaded([request], num_beams=4) == TransitionPredictor(model)([request], num_beams=4)


def test_decompiler_with_transition_model(tmp_path):
    """A CONTROL_FLOW_MODEL with ARCH transition: the decompiler takes its skeletons, right or not, never the templates."""
    model_dir = tmp_path / "model"
    tiny_model().save_pretrained(model_dir)
    config = {"REPO": str(model_dir), "REVISION": "main", "ARCH": "transition"}
    predictor = PREDICTORS[config["ARCH"]].from_pretrained(config["REPO"], revision=config["REVISION"])
    d = decompile_with(predictor, tmp_path)
    f = [bc.codeobj.co_name for bc in d.ordered_bytecodes].index("f")
    assert d.cflow_results[f] and all(d.cflow_requests[f].grammar().accepts(s) for s in d.cflow_results[f])
    # the untrained model's skeleton can be wrong, but it fits the grammar, so the source compiles
    assert isinstance(cft_of_f(d), SkeletonTemplate) and d.cflow_requests[f].grammar().accepts(cft_of_f(d).tokens)
    assert d.equivalence_results and not has_comp_error(d.equivalence_results)


def test_shared_control_flow_config():
    """A top-level CONTROL_FLOW_MODEL serves every version; a version's own entry overrides it."""
    shared, own = {"REPO": "unified", "ARCH": "transition"}, {"REPO": "py312", "ARCH": "transition"}
    config = {"v3.6": {}, "v3.12": {"CONTROL_FLOW_MODEL": own}}
    assert control_flow_config(config, PythonVersion((3, 6))) is None
    config["CONTROL_FLOW_MODEL"] = shared
    assert control_flow_config(config, PythonVersion((3, 6))) == shared
    assert control_flow_config(config, PythonVersion((3, 12))) == own


def test_in_order_past_the_limits():
    """Requests the model cannot take get their statements in order, so every request has a skeleton."""
    model = tiny_model()
    model.config.max_stmts = 0
    request = CflowRequest.from_row({"input": "| <s0> K_cond POP_JUMP_IF_FALSE f+ <s2> | <s1> K_stmt | <s2> K_stmt RETURN_CONST nf", "shapes": "cond stmt stmt", "version": "3.12"})
    grammar = request.grammar()
    assert TransitionPredictor(model)([request, None], num_beams=2) == [[grammar.complete(grammar.start())], None]


def test_budget_batches():
    """Every row is in one batch of each epoch, batches keep to the budget, and epochs differ."""
    sys.path.insert(0, str(Path(__file__).parents[1] / "dev_scripts" / "control_flow"))
    from train_transition import BudgetBatches

    rng = np.random.default_rng(0)
    n_stmts = rng.integers(1, 40, 2000)
    n_stmts[:5] = 2000  # a few huge code objects
    rows = {"n_stmts": n_stmts, "length": 4 * n_stmts, "steps": 3 * n_stmts}
    sampler = BudgetBatches(rows, rows=16, shuffle=True, seed=0)
    epochs = [list(sampler) for _ in range(2)]
    for batches in epochs:
        assert sorted(i for b in batches for i in b) == list(range(2000))
        assert abs(len(batches) - len(sampler)) <= len(sampler) // 10
        for b in batches:
            assert len(b) <= 16
            if len(b) > 1:
                assert len(b) * max(rows["length"][b]) <= 16 * BudgetBatches.TOKENS_PER_ROW
                assert len(b) * max(n_stmts[b]) * max(rows["steps"][b]) <= 16 * BudgetBatches.CELLS_PER_ROW
    assert epochs[0] != epochs[1]
    assert [b for b in BudgetBatches(rows, rows=16, shuffle=True, seed=0)] == epochs[0]
