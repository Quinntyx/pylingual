"""Run: python -m unittest test.test_shared_models -v (no Hub downloads)."""
import json
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForTokenClassification, T5Config, T5ForConditionalGeneration
from tokenizers import Tokenizer, models

from pylingual.lora import AdapterBank, VersionedPipeline
from pylingual.models import CacheTranslator, VersionedTranslator, load_models
from pylingual.utils.version import PythonVersion
from dev_scripts.shared_training import grouped_examples
from pylingual.utils.generate_bytecode import CompileError, _compile_uv
from pylingual.shared_models import _registry, load_family, load_version


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(4, 4)
        self.value = nn.Linear(4, 4)

    def forward(self, x):
        return self.value(self.query(x))


class SharedModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Tiny()
        self.original = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        self.input = torch.randn(3, 4)
        self.before = self.model(self.input).detach()
        self.bank = AdapterBank(self.model, "immutable-backbone-1", ("query", "value"))
        self.bank.add("3.9", rank=2, alpha=4)
        self.bank.add("3.10", rank=2, alpha=4)

    def test_zero_initialized_adapter_is_identity(self):
        with self.bank.use("3.9"):
            torch.testing.assert_close(self.model(self.input), self.before)

    def test_only_selected_adapter_can_train(self):
        params = self.bank.trainable("3.9")
        loss = self.model(self.input).square().sum()
        loss.backward()
        self.assertEqual(len(params), 4)
        for name, parameter in self.model.named_parameters():
            if ".py3_9." in name:
                self.assertIsNotNone(parameter.grad)
            else:
                self.assertIsNone(parameter.grad)
        optimizer = torch.optim.SGD(params, lr=0.01)
        optimizer.step()
        for name, layer in self.bank.layers.items():
            torch.testing.assert_close(layer.base.weight, self.original[f"{name}.weight"])
            torch.testing.assert_close(layer.base.bias, self.original[f"{name}.bias"])
        with self.bank.use("3.10"):
            torch.testing.assert_close(self.model(self.input), self.before)
        with self.bank.use("3.9"):
            self.assertFalse(torch.equal(self.model(self.input), self.before))

    def test_selection_restored_on_nested_exception(self):
        self.bank.select("3.9")
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with self.bank.use("3.10"):
                raise RuntimeError("failed")
        self.assertEqual(self.model.query.active, "py3_9")
        with self.assertRaisesRegex(ValueError, "No adapter"):
            with self.bank.use("3.15"):
                pass
        self.assertEqual(self.model.query.active, "py3_9")

    def test_checkpoint_roundtrip_and_backbone_guard(self):
        with torch.no_grad():
            self.model.query.adapters["py3_9"].b.fill_(0.1)
        with self.bank.use("3.9"):
            expected = self.model(self.input).detach()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.bank.save("3.9", root)
            restored = Tiny()
            restored.load_state_dict(self.original)
            bank = AdapterBank(restored, "immutable-backbone-1", ("query", "value"))
            bank.load("3.9", root)
            with bank.use("3.9"):
                torch.testing.assert_close(restored(self.input), expected)
            wrong = AdapterBank(Tiny(), "different-backbone", ("query", "value"))
            with self.assertRaisesRegex(ValueError, "backbone ID mismatch"):
                wrong.load("3.9", root)
            self.assertFalse(wrong.specs)
            bad = json.loads((root / "adapter.json").read_text())
            bad["layers"] = []
            (root / "adapter.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "target layers"):
                AdapterBank(Tiny(), "immutable-backbone-1", ("query", "value")).load("3.9", root)

    def test_pipeline_materializes_iterators_under_lock(self):
        bank = self.bank
        entered = threading.Event()
        observations = []

        class Pipeline:
            def __call__(self):
                def iterator():
                    entered.set()
                    observations.append(bank.model.query.active)
                    time.sleep(0.03)
                    observations.append(bank.model.query.active)
                    yield "result"
                return iterator()

        pipeline = VersionedPipeline(Pipeline(), bank, "3.9")
        def other_request():
            entered.wait(1)
            with bank.use("3.10"):
                observations.append(bank.model.query.active)
        thread = threading.Thread(target=other_request)
        thread.start()
        self.assertEqual(pipeline(), ["result"])
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(observations, ["py3_9", "py3_9", "py3_10"])

    def test_translation_retries_preserve_all_requests_and_prefixes(self):
        class Encoding(dict):
            def to(self, device):
                return self

        class Tokenizer:
            def __init__(self):
                self.batches = []

            def __call__(self, batch, **kwargs):
                self.batches.append(batch)
                return Encoding(input_ids=torch.tensor([[len(batch)]] * len(batch)))

            def decode(self, tokens, **kwargs):
                return "pass"

        tokenizer = Tokenizer()
        calls = []
        fail_beta = False
        def generate(input_ids, **kwargs):
            calls.append(self.model.query.active)
            if len(input_ids) > 1 or (fail_beta and "beta" in tokenizer.batches[-1][0]):
                raise RuntimeError("simulated large-batch OOM")
            return torch.tensor([[5]])
        self.model.generate = generate
        translator = VersionedTranslator(self.bank, tokenizer, torch.device("cpu"), "3.9", "Python {version}: ")
        requests = ["LOAD_CONST , alpha", "LOAD_CONST , beta", "LOAD_CONST , gamma"]
        self.assertEqual(translator(requests), ["pass"] * 3)
        self.assertEqual(len(translator.cache), 3)
        self.assertEqual(len(tokenizer.batches), 4)
        self.assertTrue(all(text.startswith("Python 3.9: ") for batch in tokenizer.batches for text in batch))
        self.assertTrue(all(active == "py3_9" for active in calls))
        self.assertIsNone(self.model.query.active)
        self.assertEqual(translator(requests), ["pass"] * 3)
        self.assertEqual(len(tokenizer.batches), 4)
        # If one single-row retry also fails, the final fallback must replay all
        # requests, including rows processed before that failure.
        fail_beta = True
        second = VersionedTranslator(self.bank, tokenizer, torch.device("cpu"), "3.10", "Python {version}: ")
        results = second(requests)
        self.assertEqual(results[0], "pass")
        self.assertIn("Decompiler error", results[1])
        self.assertEqual(results[2], "pass")
        self.assertEqual(len(second.cache), 3)
        self.assertIsNone(self.model.query.active)

    def test_translation_timeout_is_not_swallowed(self):
        translator = VersionedTranslator(self.bank, None, torch.device("cpu"), "3.9", "Python {version}: ")
        def timeout(*args, **kwargs):
            raise TimeoutError("deadline")
        translator._translate_and_decode = timeout
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            translator(["LOAD_CONST , alpha"])
        self.assertIsNone(self.model.query.active)

    def test_family_registry_reuses_backbone_and_checks_versions_and_paths(self):
        model = T5ForConditionalGeneration(T5Config(vocab_size=4, d_model=16, d_kv=4, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=4, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2, "return": 3}, unk_token="<unk>")), pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model.save_pretrained(root / "backbone")
            tokenizer.save_pretrained(root / "tokenizer")
            bank = AdapterBank(model, "registry-test", ("q", "v"))
            for version in ("3.9", "3.10"):
                bank.add(version, rank=2)
                bank.save(version, root / "adapters" / version)
            metadata = {"format": 1, "complete": True, "task": "statement", "backbone_id": "registry-test", "targets": ["q", "v"], "versions": ["3.9", "3.10"], "adapters": {v: f"adapters/{v}" for v in ("3.9", "3.10")}}
            (root / "family.json").write_text(json.dumps(metadata))
            try:
                first, first_tokenizer, loaded_metadata = load_version(root, "statement", "3.9", torch.device("cpu"))
                second, second_tokenizer, _ = load_version(root / ".", "statement", "3.10", torch.device("cpu"))
                self.assertIs(first, second)
                self.assertIs(first.model, second.model)
                self.assertIs(first_tokenizer, second_tokenizer)
                self.assertEqual(set(first.specs), {"3.9", "3.10"})
                with self.assertRaisesRegex(ValueError, "No statement adapter"):
                    load_version(root, "statement", "3.15", torch.device("cpu"))
                loaded_metadata["versions"].append("3.14")
                loaded_metadata["adapters"]["3.14"] = "../outside"
                with self.assertRaisesRegex(ValueError, "escapes"):
                    load_version(root, "statement", "3.14", torch.device("cpu"))
                unfinished = root / "unfinished"
                unfinished.mkdir()
                (unfinished / "family.json").write_text(json.dumps({"format": 1, "complete": False, "task": "statement"}))
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    load_family(unfinished, "statement", torch.device("cpu"))
            finally:
                _registry.pop((str(root.resolve()), "statement", "cpu"), None)

    def test_no_adapter_ablation_keeps_held_out_version_prefix(self):
        self.bank.select("3.9")
        translator = VersionedTranslator(self.bank, object(), torch.device("cpu"), "3.15", "Python {version}: ", use_adapter=False)
        self.assertEqual(translator.input_prefix, "Python 3.15: ")
        with mock.patch.object(CacheTranslator, "__call__", side_effect=lambda *args, **kwargs: self.model.query.active):
            self.assertIsNone(translator(["LOAD_CONST 1"]))
        self.assertEqual(self.model.query.active, "py3_9")
        self.assertNotIn("3.15", self.bank.specs)

    def test_mixed_legacy_segmentation_and_shared_translation_routing(self):
        for use_adapters in (True, False):
            with self.subTest(use_adapters=use_adapters):
                config = {"SHARED_BACKBONES": {"STATEMENT_MODEL": "statement", "USE_ADAPTERS": use_adapters}, "v3.15": {"SEGMENTATION_MODEL": {"REPO": "legacy-segmenter"}}}
                legacy_segmenter = object()
                metadata = {"source_prefix": "Python {version}: "}
                load_models.cache_clear()
                with mock.patch.object(Path, "open", mock.mock_open(read_data="")), mock.patch("pylingual.models.yaml.safe_load", return_value=config), mock.patch("pylingual.models._load_legacy_segmenter", return_value=legacy_segmenter) as legacy, mock.patch("pylingual.shared_models.load_version", return_value=(self.bank, object(), metadata)) as adapted, mock.patch("pylingual.shared_models.load_family", return_value=(self.bank, object(), metadata, None)) as unadapted:
                    segmenter, translator = load_models(Path("test-config.yaml"), PythonVersion("3.15"))
                    self.assertIs(segmenter, legacy_segmenter)
                    self.assertIs(translator.model, self.model)
                    self.assertEqual(translator.version, "3.15" if use_adapters else None)
                    legacy.assert_called_once()
                    self.assertEqual(adapted.call_count, int(use_adapters))
                    self.assertEqual(unadapted.call_count, int(not use_adapters))
        load_models.cache_clear()

    def test_held_out_version_cannot_enter_default_backbone_groups(self):
        rows = [{"version": "3.15", "file": "held-out"}]
        with mock.patch("dev_scripts.shared_training.encode_row", return_value={"input_ids": [1, 2]}):
            with self.assertRaisesRegex(ValueError, "Training versions differ"):
                grouped_examples("statement", object(), rows)
            grouped, dropped = grouped_examples("statement", object(), rows, versions=("3.15",))
        self.assertEqual(set(grouped), {"3.15"})
        self.assertEqual(sum(dropped.values()), 0)

    def test_successful_uv_compile_ignores_prerelease_warning(self):
        completed = mock.Mock(returncode=0, stdout="", stderr="warning: You're using a pre-release version of Python (3.15.0b2) but a stable version is available.\n")
        with mock.patch("pylingual.utils.generate_bytecode.subprocess.run", return_value=completed) as run:
            _compile_uv("valid.py", "valid.pyc", PythonVersion("3.15"))
        self.assertIn("doraise=True", run.call_args.args[0][-1])

    def test_failed_uv_compile_reports_source_line_not_wrapper_traceback(self):
        stderr = 'Traceback (most recent call last):\n  File "py_compile.py", line 150\npy_compile.PyCompileError: Sorry: IndentationError: unexpected indent (source.py, line 7)\n'
        with mock.patch("pylingual.utils.generate_bytecode.subprocess.run", return_value=mock.Mock(returncode=1, stdout="", stderr=stderr)):
            with self.assertRaises(CompileError) as caught:
                _compile_uv("source.py", "source.pyc", PythonVersion("3.15"))
        self.assertIn("line 7", str(caught.exception))
        self.assertNotIn("line 150", str(caught.exception))
        self.assertNotIn("Traceback", str(caught.exception))

    def test_failed_uv_compile_with_empty_stderr_still_fails(self):
        with mock.patch("pylingual.utils.generate_bytecode.subprocess.run", return_value=mock.Mock(returncode=1, stdout="", stderr="")):
            with self.assertRaisesRegex(CompileError, "status 1"):
                _compile_uv("invalid.py", "invalid.pyc", PythonVersion("3.15"))

    def test_real_t5_forward_backward_and_generation(self):
        model = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=16, d_kv=4, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=4, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))
        bank = AdapterBank(model, "tiny-t5", ("q", "v"))
        bank.add("3.14", rank=2)
        bank.trainable("3.14")
        inputs = torch.tensor([[2, 3, 1]])
        model(input_ids=inputs, labels=inputs).loss.backward()
        with bank.use("3.14"):
            generated = model.generate(inputs, max_new_tokens=3)
        self.assertEqual(generated.shape[0], 1)
        self.assertTrue(all(not layer.base.weight.requires_grad for layer in bank.layers.values()))

    def test_real_roberta_forward_backward(self):
        model = RobertaForTokenClassification(RobertaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=32, num_labels=3))
        bank = AdapterBank(model, "tiny-roberta", ("query", "value"))
        bank.add("3.14", rank=2)
        bank.trainable("3.14")
        output = model(input_ids=torch.tensor([[2, 3, 4]]), labels=torch.tensor([[0, 1, 2]]))
        output.loss.backward()
        self.assertEqual(tuple(output.logits.shape), (1, 3, 3))


if __name__ == "__main__":
    unittest.main()
