import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dev_scripts.benchmark_stage_timing import FileTimer, RuntimeProfiler, aggregate_timings, _ACTIVE


class Clock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def advance(self, value): self.now += value


class TimingTests(unittest.TestCase):
    def test_exclusive_nested_stages_and_denominator(self):
        clock = Clock()
        with FileTimer(clock=clock) as timer:
            clock.advance(1)
            with timer.stage("segmentation"):
                clock.advance(2)
                with timer.stage("tokenization.encode"): clock.advance(3)
                clock.advance(4)
            clock.advance(5)
        result = timer.result()
        self.assertEqual(result["elapsed_seconds"], 15)
        self.assertEqual(result["exclusive_seconds"]["segmentation"], 6)
        self.assertEqual(result["exclusive_seconds"]["tokenization"], 3)
        self.assertEqual(result["exclusive_seconds"]["other"], 6)
        self.assertAlmostEqual(sum(result["percentages"].values()), 100)
        self.assertIsNone(_ACTIVE.get())

    def test_same_category_reentry_never_double_counts(self):
        clock = Clock()
        with FileTimer(clock=clock) as timer:
            with timer.stage("translation"):
                clock.advance(2)
                with timer.stage("translation.model"): clock.advance(3)
        self.assertEqual(timer.result()["exclusive_seconds"]["translation"], 5)

    def test_exception_retains_partial_and_restores_context(self):
        clock = Clock()
        with self.assertRaises(TimeoutError):
            with FileTimer(clock=clock) as timer:
                with timer.stage("translation.model", gpu=True):
                    clock.advance(7)
                    raise TimeoutError("deadline")
        result = timer.result()
        self.assertTrue(result["partial"])
        self.assertTrue(result["gpu_sync_incomplete"])
        self.assertEqual(result["exclusive_seconds"]["translation"], 7)
        self.assertIsNone(_ACTIVE.get())

    def test_gpu_synchronization_only_at_explicit_model_boundaries(self):
        sync = []
        with FileTimer(synchronize=lambda: sync.append(1)) as timer:
            with timer.stage("tokenization.encode"): pass
            with timer.stage("translation.model", gpu=True): pass
        self.assertEqual(len(sync), 2)

    def test_aggregation_weights_seconds_instead_of_percentages(self):
        rows = []
        for duration, token_time in ((1, 1), (9, 0)):
            clock = Clock()
            with FileTimer(clock=clock) as timer:
                with timer.stage("tokenization.encode"): clock.advance(token_time)
                clock.advance(duration-token_time)
            rows.append({"success": True, "timing": timer.result()})
        result = aggregate_timings(rows + [{"success": False}])
        self.assertEqual(result["files_profiled"], 2)
        self.assertEqual(result["files_missing_timings"], 1)
        self.assertEqual(result["all_files"]["percentages"]["tokenization"], 10)

    def test_patch_restoration_and_no_effect_outside_recording(self):
        class Dummy:
            def run(self, value): return value + 1
        original = Dummy.run
        profiler = RuntimeProfiler()
        try:
            profiler._wrap(Dummy, "run", "translation")
            self.assertEqual(Dummy().run(2), 3)
            with profiler.record() as timer: self.assertEqual(Dummy().run(3), 4)
            self.assertEqual(timer.result()["calls"]["translation"], 1)
        finally: profiler.close()
        self.assertIs(Dummy.run, original)

    def fixture_models(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration
        from pylingual.models import CacheTranslator
        torch.set_num_threads(1)
        backend = Tokenizer(models.WordLevel({"<pad>":0,"<s>":1,"</s>":2,"<unk>":3,"<mask>":4,"LOAD_CONST":5,"return":6},unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",unk_token="<unk>",eos_token="</s>",bos_token="<s>")
        model = T5ForConditionalGeneration(T5Config(vocab_size=7,d_model=8,d_ff=16,num_layers=1,num_decoder_layers=1,num_heads=2,d_kv=4,decoder_start_token_id=0,eos_token_id=2,pad_token_id=0)).eval()
        translator = CacheTranslator(model,tokenizer)
        translator.generation_kwargs = {"max_new_tokens":2}
        segmenter = SimpleNamespace(model=torch.nn.Linear(2,3).eval(),tokenizer=tokenizer)
        return segmenter, translator

    def test_real_hf_generate_and_tokenizer_output_unchanged(self):
        import torch
        segmenter, translator = self.fixture_models()
        input = torch.ones(1,2)
        before = translator(["LOAD_CONST"])
        logits = segmenter.model(input).detach().clone()
        translator.cache.clear()
        profiler = RuntimeProfiler()
        try:
            profiler.install(segmenter,translator)
            with profiler.record() as timer:
                after = translator(["LOAD_CONST"])
                actual = segmenter.model(input)
            self.assertEqual(before,after)
            torch.testing.assert_close(logits,actual)
            timings = timer.result()
            for category in ("tokenization","segmentation","translation"):
                self.assertGreater(timings["exclusive_seconds"][category],0)
            self.assertIn("translation.model",timings["calls"])
            self.assertAlmostEqual(sum(timings["exclusive_seconds"].values()),timings["elapsed_seconds"])
        finally: profiler.close()
        self.assertNotIn("generate",translator.model.__dict__)
        self.assertNotIn("forward",segmenter.model.__dict__)

    def test_cli_run_marker_enables_already_queued_benchmarks(self):
        from dev_scripts import benchmark_full_retrain as benchmark
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"benchmark-timing.json").write_text('{"enabled":true}')
            argv=["benchmark","--run",directory,"--output",directory,"--version","3.14","--arm","statement_adapter"]
            with patch("sys.argv",argv),patch.object(benchmark,"unit") as unit:
                benchmark.main()
                self.assertTrue(unit.call_args.args[0].stage_timing)
            with patch("sys.argv",argv+["--no-stage-timing"]),patch.object(benchmark,"unit") as unit:
                benchmark.main()
                self.assertFalse(unit.call_args.args[0].stage_timing)

    def test_provenance_rejects_any_production_source_change(self):
        import hashlib
        from dev_scripts.benchmark_full_retrain import instrumentation_provenance
        digest=lambda value: hashlib.sha256(value.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);repo=root/"repo";repo.mkdir()
            originals={"pylingual/view.py":"frozen","dev_scripts/benchmark_full_retrain.py":"old","dev_scripts/evaluate_shared.py":"old"}
            for name,text in originals.items():
                (repo/name).parent.mkdir(parents=True,exist_ok=True);(repo/name).write_text(text)
            manifest=root/"source-manifest.json"
            manifest.write_text(json.dumps({"files":{name:digest(text) for name,text in originals.items()}}))
            overrides={}
            for name in ("dev_scripts/benchmark_full_retrain.py","dev_scripts/evaluate_shared.py","dev_scripts/benchmark_stage_timing.py"):
                (repo/name).write_text("logging")
                overrides[name]={"original_sha256":digest(originals[name]) if name in originals else None,"instrumented_sha256":digest("logging")}
            (root/"benchmark-instrumentation-manifest.json").write_text(json.dumps({"original_source_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"overrides":overrides}))
            self.assertIsNotNone(instrumentation_provenance(root,repo))
            (repo/"pylingual/view.py").write_text("changed")
            with self.assertRaisesRegex(ValueError,"Benchmark source differs"):
                instrumentation_provenance(root,repo)

    def test_file_metrics_persists_timings_for_success_and_timeout(self):
        from dev_scripts import evaluate_shared as evaluator
        segmenter,translator=self.fixture_models()
        items=[{"version":"3.14","relative":"ok.pyc"},{"version":"3.14","relative":"timeout.pyc"}]
        def decompile(path,*args,**kwargs):
            timer = _ACTIVE.get()
            self.assertIsNotNone(timer)
            with timer.stage("cfg_reconstruction.build"):
                if path.name=="timeout.pyc": raise TimeoutError("fixture deadline")
            return SimpleNamespace(equivalence_results=[SimpleNamespace(bc_a=object(),success=True)])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(output=root,config=root/"runtime.json",versions=["3.14"],dataset=root,timeout=10,resume=False,stage_timing=True)
            with patch.object(evaluator,"load_models",return_value=(segmenter,translator)),patch.object(evaluator,"decompile",side_effect=decompile),contextlib.redirect_stdout(io.StringIO()):
                evaluator.file_metrics(args,{"files":items})
            rows=[json.loads(line) for line in (root/"file_results.jsonl").read_text().splitlines()]
            summary=json.loads((root/"timing-summary.json").read_text())
            self.assertEqual([row["category"] for row in rows],["Equal","Timeout"])
            self.assertEqual(summary["files_profiled"],2)
            self.assertEqual(summary["failed_files"]["partial_files"],1)
            self.assertGreater(summary["all_files"]["exclusive_seconds"]["cfg_reconstruction"],0)

if __name__ == "__main__": unittest.main()
