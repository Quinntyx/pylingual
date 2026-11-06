"""Opt-in exclusive wall-clock instrumentation; never changes decompiler outputs.

Model boundaries are CUDA-synchronized, not individual autoregressive tokens.
Percentages use warm decompilation time, excluding cold model loading and output
logging. This is a diagnostic run, not an uninstrumented throughput measurement.
"""
from __future__ import annotations
from collections import Counter
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
import time
from unittest.mock import patch

_ACTIVE = ContextVar("benchmark_stage_timer", default=None)
CATEGORIES = ("tokenization", "segmentation", "translation", "cfg_reconstruction",
              "verification", "preprocessing", "unmasking", "other")
TIMING_PROTOCOL = {
    "schema": 1, "clock": "time.perf_counter", "accounting": "exclusive nested wall time",
    "denominator": "warm decompilation call, including failures, excluding model startup and JSON/output logging",
    "gpu": "synchronize at model forward/generate boundaries; not GPU kernel-only time",
    "definitions": {
        "tokenization": "model tokenizer encoding, decoding and padding, including instruction-window sizing",
        "segmentation": "segmentation and boundary-repair work, excluding nested tokenizer/translation/CFG/verification work",
        "translation": "statement translation, cache handling and retries, excluding nested tokenizer work",
        "cfg_reconstruction": "control-flow/CFT construction, source rendering and reconstruction updates",
        "verification": "compilation and bytecode-equivalence checks",
        "preprocessing": "bytecode preprocessing and masking",
        "unmasking": "restoring source placeholders",
        "other": "PYC loading, orchestration and otherwise unattributed time"},
    "limitations": ["Synchronization and Python hooks can perturb throughput/overlap.",
                    "Instrumentation overhead is included, not independently subtracted.",
                    "CPU parsing/CFG costs are not tokenization. Tokenizer decode is broken out separately.",
                    "Failed/timed-out calls retain partial timings; asynchronous work on an exceptional GPU exit may be incomplete."]}


class FileTimer:
    def __init__(self, *, clock=time.perf_counter, synchronize=lambda: None):
        self.clock, self.synchronize = clock, synchronize
        self.exclusive, self.calls = Counter(), Counter()
        self.errors = Counter()
        self.stack = ["other"]
        self.running = False
        self.gpu_sync_incomplete = False

    def __enter__(self):
        if self.running:
            raise RuntimeError("Timer already running")
        self.start = self.last = self.clock()
        self.running = True
        self.token = _ACTIVE.set(self)
        return self

    def _charge(self):
        now = self.clock()
        self.exclusive[self.stack[-1]] += now - self.last
        self.last = now
        return now

    @contextmanager
    def stage(self, name, *, gpu=False):
        self._charge()
        self.stack.append(name)
        self.calls[name] += 1
        try:
            if gpu:
                self.synchronize()
            yield
            if gpu:
                self.synchronize()
        except BaseException as exc:
            self.errors[type(exc).__name__] += 1
            if gpu:
                self.gpu_sync_incomplete = True
            raise
        finally:
            self._charge()
            self.stack.pop()

    def __exit__(self, kind, exc, traceback):
        self.partial = kind is not None
        self.elapsed = self._charge() - self.start
        self.running = False
        _ACTIVE.reset(self.token)
        return False

    def result(self):
        if self.running:
            raise RuntimeError("Timer still running")
        seconds = {name: 0.0 for name in CATEGORIES}
        for label, duration in self.exclusive.items():
            seconds[label.split(".", 1)[0]] += duration
        return {"schema": 1, "elapsed_seconds": self.elapsed,
                "exclusive_seconds": seconds,
                "percentages": {k: 100*v/self.elapsed if self.elapsed else 0.0 for k,v in seconds.items()},
                "detail_seconds": dict(self.exclusive), "calls": dict(self.calls),
                "partial": self.partial, "gpu_sync_incomplete": self.gpu_sync_incomplete,
                "stage_exceptions": dict(self.errors)}


class RuntimeProfiler:
    """Process-local, reversible hooks. Nothing is timed outside a FileTimer."""
    def __init__(self, *, synchronize=lambda: None):
        self.synchronize = synchronize
        self.patches = ExitStack()
        self.installed = set()

    def _wrap(self, owner, attribute, label, *, gpu=False):
        if (id(owner), attribute) in self.installed or not callable(getattr(owner, attribute, None)):
            return
        original = getattr(owner, attribute)
        @wraps(original)
        def measured(*args, **kwargs):
            recorder = _ACTIVE.get()
            if recorder is None:
                return original(*args, **kwargs)
            with recorder.stage(label, gpu=gpu):
                return original(*args, **kwargs)
        self.patches.enter_context(patch.object(owner, attribute, measured))
        self.installed.add((id(owner), attribute))

    def install(self, segmenter, translator):
        import importlib
        decompiler = importlib.import_module("pylingual.decompiler")
        from pylingual.models import CacheTranslator
        from pylingual.control_flow_reconstruction.source import SourceContext
        for method, label in {
            "run_segmentation": "segmentation", "correct_segmentation": "segmentation.repair",
            "run_translation": "translation", "run_cflow_reconstruction": "cfg_reconstruction.build",
            "reconstruct_source": "cfg_reconstruction.render", "check_reconstruction": "verification",
            "preprocess_bytecode": "preprocessing", "mask_bytecode": "preprocessing.mask",
            "unmask_lines": "unmasking"}.items():
            self._wrap(decompiler.Decompiler, method, label)
        self._wrap(decompiler, "bc_to_cft", "cfg_reconstruction.build")
        for method in ("__str__", "update_lines", "snapshot", "restore", "purge"):
            self._wrap(SourceContext, method, "cfg_reconstruction.render")
        # Covers initial translation AND direct translator calls during repairs.
        self._wrap(CacheTranslator, "__call__", "translation")
        self._wrap(segmenter.model, "forward", "segmentation.model", gpu=True)
        self._wrap(translator.model, "generate", "translation.model", gpu=True)
        for tokenizer in (segmenter.tokenizer, translator.tokenizer):
            for method in ("__call__", "encode", "encode_plus", "batch_encode_plus", "tokenize"):
                self._wrap(type(tokenizer), method, "tokenization.encode")
            for method in ("decode", "batch_decode"):
                self._wrap(type(tokenizer), method, "tokenization.decode")
            self._wrap(type(tokenizer), "pad", "tokenization.padding")

    def record(self):
        return FileTimer(synchronize=self.synchronize)

    def close(self):
        self.patches.close()


def aggregate_timings(rows):
    """Weight by total seconds, never by the mean of per-file percentages."""
    rows = list(rows)
    timed = [row for row in rows if "timing" in row]
    def summarize(subset):
        seconds = {name: sum(row["timing"]["exclusive_seconds"].get(name, 0.0) for row in subset) for name in CATEGORIES}
        total = sum(row["timing"]["elapsed_seconds"] for row in subset)
        details = Counter()
        calls = Counter()
        for row in subset:
            details.update(row["timing"].get("detail_seconds", {}))
            calls.update(row["timing"].get("calls", {}))
        return {"files": len(subset), "elapsed_seconds": total, "exclusive_seconds": seconds,
                "percentages": {k: 100*v/total if total else 0.0 for k,v in seconds.items()},
                "detail_seconds": dict(details), "calls": dict(calls),
                "partial_files": sum(bool(row["timing"].get("partial")) for row in subset),
                "gpu_sync_incomplete_files": sum(bool(row["timing"].get("gpu_sync_incomplete")) for row in subset)}
    return {"protocol": TIMING_PROTOCOL, "files_total": len(rows), "files_profiled": len(timed),
            "files_missing_timings": len(rows)-len(timed), "all_files": summarize(timed),
            "successful_files": summarize([row for row in timed if row.get("success")]),
            "failed_files": summarize([row for row in timed if not row.get("success")])}
