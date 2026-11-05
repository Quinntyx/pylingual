# Shared v3 150k restart: official benchmark — 2026-11-05

All five training stages and all eight official benchmark units are complete. The independent
final audit accepted **27,950/27,950 evaluations**: no missing or duplicate rows, no model-load
failures, and zero pooled raw-source overlap. Final models and final-step resumable checkpoints
are public and pinned in `experiments/shared-v3-main-150k-2026-11-02.json`.

## Protocol and training coverage

This is the approved restart from new source collection, **not a continuation of the aborted v2
100k weights**. The backbone used 150,000 source files across Python 3.6–3.15 (13,500 PyPI + 1,500
CSN per version); each 3.14/3.15 adapter used 150,000 training + 1,000 validation files. The frozen
v1 tokenizer and every other approved v2 training setting were unchanged. All five stages
completed two epochs; the statement LoRAs are rank 64. Both arms use the same universal segmenter
without segmentation LoRAs. Only the matching statement LoRA is enabled/disabled.

Scoring uses the unchanged `benchmark_shared_v1.py` at source commit
`d084b91508b57b6030acd6d913b6f5e79cd8417e`:

- Strict **full-file bytecode equivalence**, not statement accuracy or behavioral testing.
- FP32 inference, 300 seconds per file; all official listed files retained, including failures.
- Eval v1.1.0 at `a60d3b95748283ba0fbcd5e94da307d5bd5df870`; eval v2.0.1 at
  `258bc0380f7c8550fa28f294b896cf168a5e4055`.
- Verification compilers **3.14.6** and **3.15.0b2**, not final 3.15.
- `TORCH_DISABLE_NATIVE_JIT=1`; triplesec1 evaluated 3.14 and triplesec2 evaluated 3.15, with
  the two arms side by side. Two-file smoke checks were separate from the official results.

## Full-file accuracy and paired LoRA effect

| Eval | Python | No LoRA | With LoRA | Newly correct | Regressed | Net files | Gain, pp |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | 3.14 | 4137/4991 = 82.89% | 4155/4991 = 83.25% | 38 | 20 | +18 | +0.36065 |
| v1 | 3.15 | 4124/4991 = 82.63% | 4139/4991 = 82.93% | 45 | 30 | +15 | +0.30054 |
| v2 | 3.14 | 711/1915 = 37.13% | 717/1915 = 37.44% | 26 | 20 | +6 | +0.31332 |
| v2 | 3.15 | 846/2078 = 40.71% | 849/2078 = 40.86% | 27 | 24 | +3 | +0.14437 |

The full set is also the non-overlap stratum in every unit. The overlap check independently
reconstructed **1,052,604 identifiers from 552,000 accepted source records**: tokenizer 100k,
backbone 150k, and 150k train + 1k validation per adapter. It checks raw source/identifier membership;
it does not rederive canonical source equivalence from bytecode or establish project-disjointness.
The pre-training canonical/raw/NAS exclusion audits are retained in the run workspace.

## Comparison with the October 31 shared v1 restart

| Eval | Python | Prior no LoRA | v3 no LoRA | Prior with LoRA | v3 with LoRA | With-LoRA change, pp |
|---|---|---:|---:|---:|---:|---:|
| v1 | 3.14 | 80.32% | 82.89% | 81.43% | 83.25% | +1.82328 |
| v1 | 3.15 | 65.30% | 82.63% | 76.46% | 82.93% | +6.47165 |
| v2 | 3.14 | 32.90% | 37.13% | 34.26% | 37.44% | +3.18538 |
| v2 | 3.15 | 23.34% | 40.71% | 33.30% | 40.86% | +7.55534 |

This comparison is **not controlled for training size**. The prior run's 50k universal backbone
covered 3.6–3.14 and excluded 3.15; this run's 150k backbone covers 3.6–3.15. Adapter training
budgets and final weights also differ. In particular, the earlier large 3.15 LoRA gain concerned
a version absent from backbone training, whereas 3.15 is now in-distribution. Do not interpret
these results as unseen-version transfer or attribute the cross-run changes solely to 150k data.
Only the paired within-v3 LoRA comparison isolates the adapter toggle.

## Failure accounting

| Eval | Python | Arm | Equal | CompileError | Different | Error | Timeout |
|---|---|---|---:|---:|---:|---:|---:|
| v1 | 3.14 | no_adapter | 4137 | 331 | 521 | 1 | 1 |
| v1 | 3.14 | statement_adapter | 4155 | 356 | 479 | 0 | 1 |
| v1 | 3.15 | no_adapter | 4124 | 348 | 517 | 1 | 1 |
| v1 | 3.15 | statement_adapter | 4139 | 355 | 495 | 1 | 1 |
| v2 | 3.14 | no_adapter | 711 | 603 | 599 | 1 | 1 |
| v2 | 3.14 | statement_adapter | 717 | 635 | 562 | 0 | 1 |
| v2 | 3.15 | no_adapter | 846 | 623 | 607 | 1 | 1 |
| v2 | 3.15 | statement_adapter | 849 | 643 | 584 | 1 | 1 |

The six `Error` outcomes concern source `7bdf309e…`, which hits a predicate-selection assertion.
The same `_models_py3.pyc` source (`a8f152dd…`) times out in all eight units. All remain failures;
no cases were discarded, repaired, or rerun under a changed protocol. Full traces remain in the
per-file journals. Model-load errors are zero.

## Verification and interpretation

The final audit rehashed all 18 staged model/identity files on both hosts, verified the official
release commits, committed lists and every selected PYC hash, checked all 69 runtime source-file
fingerprints, and matched every copied journal to its host digest. Both arms and both hosts use
the same family identities and compiler binaries. Actual model parameter byte counts confirm
FP32. Fixed counts, uniqueness, strict success/category consistency, zero source overlap, paired
input identities, and both runtime toggles were checked independently of the official aggregator.
Six in-memory negative controls were rejected: dropped row, duplicate row, changed PYC identity,
fake success, model-load failure, and a falsely claimed overlap flag. The unchanged official
reporter then reproduced the independently computed counts, categories and paired gains.

The LoRAs add **0.14–0.36 percentage points** in this run, with appreciable per-file regressions as
well as gains. The stronger backbone already handles both target versions well on eval v1, but
eval v2 remains much harder. This is one training seed, without ordinary per-version controls;
no production parity, training-seed uncertainty, controlled speedup or peak-VRAM claim is made.

## Evidence

- Committed machine-readable results and full provenance:
  [`experiments/shared-v3-benchmark-2026-11-05.json`](experiments/shared-v3-benchmark-2026-11-05.json).
- Plan and immutable public artifact revisions:
  [`experiments/shared-v3-main-150k-2026-11-02.json`](experiments/shared-v3-main-150k-2026-11-02.json).
- Local workspace: `/home/henry/docs/pylingual-experiments/shared-training-v3-150k-20261102`.
- Independent receipts: `benchmark-completion-verified-20261105.json`,
  `benchmark-host-final-triplesec1-gpu.json`, `benchmark-host-final-triplesec2-gpu.json`.
- Merged official report: `bench-results/results/report.json`; raw host-specific backups:
  `bench-results/hosts/{triplesec1-gpu,triplesec2-gpu}/` (journals, selections, configs,
  provenance, model stats, decompiled files and logs).
- Remote results retained at `~/pylingual-bench-v3/results` on both hosts. Existing eval and compiler
  symlinks were read-only; no linked model weights were modified.
- Prior-run report: [`SHARED_V1_BENCHMARK_20261031.md`](SHARED_V1_BENCHMARK_20261031.md).
