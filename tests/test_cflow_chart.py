import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from transformers import T5Config

from dataset_generation.bytecode2cflow import make_examples
from pylingual.control_flow_reconstruction import block_grammar
from pylingual.control_flow_reconstruction.chart import KEYWORDS, Chart, Scores, decode_order, gold_items, oracle_scores, order_beams, order_score
from pylingual.control_flow_reconstruction.chart_model import EDGE_KINDS, ChartParser
from pylingual.control_flow_reconstruction.model_view import MID, STMT_SEP, CflowRequest
from pylingual.control_flow_reconstruction.skeleton import BLOCK_KEYWORDS
from pylingual.editable_bytecode import PYCFile
from pylingual.models import PREDICTORS, ChartPredictor
from pylingual.utils.version import PythonVersion

from test_cflow_dataset import FIXTURES, VERSIONS, compiled_fixture
from test_cflow_decompiler import decompile_with

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev_scripts" / "control_flow"))
from train_cflow import build_tokenizer  # noqa: E402
from train_chart import Collator, chart_labels  # noqa: E402


def fixture_rows(version: str, tmp_path: Path, fixtures=FIXTURES) -> list[dict]:
    """Dataset rows of the fixtures that compile with `version`; skips when none does."""
    rows = []
    for fixture in fixtures:
        try:
            src, pyc_path = compiled_fixture(fixture, PythonVersion(version), tmp_path)
        except pytest.skip.Exception:
            continue
        rows += [{k: v for k, v in e.items() if not k.startswith("_")} for e in make_examples(PYCFile(str(pyc_path)), src)]
    if not rows:
        pytest.skip(f"no fixture compiles with {version}")
    return rows


def tiny_model(rows: list[dict]) -> tuple[ChartParser, object]:
    """A randomly initialized ChartParser, with a vocabulary from `rows`."""
    tokenizer = build_tokenizer([r["input"] + " " + r["target"] for r in rows], 512)
    config = T5Config(
        vocab_size=len(tokenizer), d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=4, feed_forward_proj="gated-gelu",
        is_encoder_decoder=False, use_cache=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )  # fmt: skip
    config.max_stmts, config.max_input_tokens = 512, 1 << 16
    config.skeleton_keywords = KEYWORDS
    config.statement_token_id = tokenizer.convert_tokens_to_ids(STMT_SEP)
    config.order_buckets, config.order_candidates, config.edge_features, config.loop_edges, config.loop_mlp = 32, 2, True, True, 8
    config.edge_token_ids = [[tokenizer.convert_tokens_to_ids(kind + d) for d in "+-="] for kind in EDGE_KINDS]
    config.mid_token_id = tokenizer.convert_tokens_to_ids(MID)
    config.ref_token_id = tokenizer.convert_tokens_to_ids("<s0>")
    torch.manual_seed(0)
    model = ChartParser(config).eval()
    with torch.no_grad():  # trained models score edges; zeros would leave that path untested
        model.edge_weights.normal_()
        model.loop_edge_weights.normal_()
    return model, tokenizer


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
@pytest.mark.parametrize("version", VERSIONS)
def test_chart_writes_gold_skeletons(version, tmp_path, monkeypatch):
    """
    With scores that favor the gold skeleton, in its order, the chart finds it. With random scores, its k best are
    distinct and accepted but for the block counts, which the chart leaves to BlockGrammar.check.
    """
    rng = np.random.default_rng(0)
    rows = fixture_rows(version, tmp_path)
    for row in rows:
        gold = gold_items(row["target"].split())
        best = Chart(CflowRequest.from_row(row).grammar(), gold.order, oracle_scores(gold), KEYWORDS).best(1)
        assert [t for _, t in best] == [row["target"].split()]
    for name in ("MAX_BLOCKS", "MAX_DEPTH", "MAX_FINALLY"):
        monkeypatch.setattr(block_grammar, name, 1000)
    for row in rows:
        request = CflowRequest.from_row(row)
        if (n := request.n) <= 30:
            scores = Scores(rng.normal(size=(n, n + 1)), rng.normal(size=(n, 2)), rng.normal(size=(n + 1, len(KEYWORDS))) - 1, rng.normal(size=(len(BLOCK_KEYWORDS), n + 1, n + 1)))
            best = [t for _, t in Chart(request.grammar(), gold_items(row["target"].split()).order, scores, KEYWORDS).best(5)]
            assert len(best) == 5 and all(request.grammar().accepts(t) for t in best)
            assert len({tuple(t) for t in best}) == len(best)


def test_order_beams():
    """One beam is the greedy order; beams are distinct permutations with prefix statements first, scored by order_score."""
    rng = np.random.default_rng(0)
    shapes = [block_grammar.Shape.parse(s) for s in ["prefix"] + ["stmt"] * 8]
    n = len(shapes)
    log_probs = torch.tensor(rng.normal(size=(n + 1, n + 1))).log_softmax(-1).numpy()
    assert order_beams(log_probs, shapes, 1)[0][1] == decode_order(log_probs, shapes)
    beams = order_beams(log_probs, shapes, 5)
    assert len(beams) == 5 and len({tuple(o) for _, o in beams}) == 5
    assert [s for s, _ in beams] == sorted((s for s, _ in beams), reverse=True)
    for score, order in beams:
        assert order[0] == 0 and sorted(order) == list(range(n)) and score == pytest.approx(order_score(log_probs, order))


def test_large_requests_get_the_statement_order(tmp_path):
    """Requests over the model's statement limit get BlockGrammar.complete rather than no skeleton."""
    rows = fixture_rows("3.12", tmp_path, ["Exception", "Loop"])
    model, tokenizer = tiny_model(rows)
    model.config.max_stmts = 1
    requests = [CflowRequest.from_row(r) for r in rows]
    for request, skeletons in zip(requests, ChartPredictor(model, tokenizer)(requests)):
        assert skeletons and skeletons[0] == request.grammar().complete(request.grammar().start())


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
@pytest.mark.parametrize("version", ["3.8", "3.11", "3.12", "3.14"])
def test_random_model_skeletons_are_accepted(version, tmp_path):
    rows = fixture_rows(version, tmp_path, ["Exception", "Loop", "Match", "With"])
    model, tokenizer = tiny_model(rows)
    predictor = ChartPredictor(model, tokenizer)
    requests = [CflowRequest.from_row(r) for r in rows]
    for num_beams in (1, 3):
        results = predictor(requests, num_beams=num_beams)
        for request, skeletons in zip(requests, results):
            grammar = request.grammar()
            assert skeletons and 1 <= len(skeletons) <= num_beams
            assert all(grammar.accepts(s) for s in skeletons)
            assert len({tuple(s) for s in skeletons}) == len(skeletons)


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
def test_training_step_and_round_trip(tmp_path):
    rows = fixture_rows("3.12", tmp_path, ["Exception", "Loop"])
    model, tokenizer = tiny_model(rows)
    batch = Collator(tokenizer)([tokenizer(r["input"]) | chart_labels(r) for r in rows[:8]])
    loss = model.train()(**batch)["loss"]
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    flat = [tokenizer(r["input"]) | chart_labels(r) for r in rows if "{" not in r["target"]]
    assert flat and torch.isfinite(model(**Collator(tokenizer)(flat))["loss"])
    model.eval().save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    requests = [CflowRequest.from_row(r) for r in rows]
    loaded = ChartPredictor.from_pretrained(str(tmp_path / "model"))
    assert loaded(requests, num_beams=2) == ChartPredictor(model, tokenizer)(requests, num_beams=2)


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
def test_decompiler_with_chart_config(tmp_path):
    """A CONTROL_FLOW_MODEL with ARCH chart, as load_models reads it, runs the control flow phases; failing skeletons fall back to the templates."""
    rows = fixture_rows(".".join(map(str, sys.version_info[:2])), tmp_path, ["Conditional"])
    model, tokenizer = tiny_model(rows)
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    config = yaml.safe_load(f"CONTROL_FLOW_MODEL: {{REPO: {tmp_path / 'model'}, ARCH: chart}}")["CONTROL_FLOW_MODEL"]
    predictor = PREDICTORS[config.get("ARCH", "t5")].from_pretrained(config["REPO"], revision=config.get("REVISION"))
    d = decompile_with(predictor, tmp_path)
    f = [bc.codeobj.co_name for bc in d.ordered_bytecodes].index("f")
    assert d.cflow_results[f] and all(d.cflow_requests[f].grammar().accepts(s) for s in d.cflow_results[f])
    assert d.equivalence_results and all(r.success for r in d.equivalence_results)


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
def test_edges_follow_the_input(tmp_path):
    """The edge features hold each (statement, target, kind, inside) of the input's references."""
    rows = fixture_rows("3.12", tmp_path, ["Exception", "Loop"])
    model, tokenizer = tiny_model(rows)
    for row in rows:
        tokens, expected, statement = row["input"].split(), set(), -1
        for i, token in enumerate(tokens):
            statement += token == STMT_SEP
            if token[:1] in EDGE_KINDS and token[1:] in ("+", "-", "=") and i + 1 < len(tokens):
                inside = i + 2 < len(tokens) and tokens[i + 2] == MID
                expected.add((statement, int(tokens[i + 1][2:-1]), EDGE_KINDS.index(token[0]) + len(EDGE_KINDS) * inside))
        encoded = tokenizer(row["input"], return_tensors="pt")
        _, _, edges = model.statements(encoded.input_ids, encoded.attention_mask)
        assert {tuple(x) for x in edges[0].nonzero().tolist()} == expected
