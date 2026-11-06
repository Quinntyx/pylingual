# Shared v1 restart: official benchmark results — 2026-10-31

Models: the universal statement backbone (50k files, Python 3.6–3.14, no 3.15), rank-64 statement
LoRAs for 3.14 and 3.15, and the base-only universal segmenter. All of them were rebuilt from the
pinned Hub revisions in `experiments/shared-v1-restart-2026-10-28.json`. The two arms differ only in
whether the version's statement LoRA is enabled.

Scoring is the same as October (`SHARED_FULL_TRAINING_BENCHMARK_20261022.md`):
- A file passes only if its bytecode is fully equivalent.
- Inference is FP32 with a 300-second limit per file.
- Every listed file in eval v1.1.0 and v2.0.1 is kept, and timeouts and errors count as failures.
- Verification compilers were 3.14.6 and 3.15.0b2.

Run with `run_shared_v1_benchmark.sh` at commit 381a6e8: triplesec1 ran the 3.14 files and
triplesec2 the 3.15 files, with both arms side by side.

None of the eval sources appears in the tokenizer, backbone or adapter corpora: the raw-source
SHA-512 overlap is 0 in every unit. The full set is therefore also the non-overlap set.

## Full-file accuracy

| Release | Python | No LoRA | With LoRA | Newly correct | Regressed | Net | Gain, pp |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | 3.14 | 4009/4991 = 80.32% | 4064/4991 = 81.43% | 75 | 20 | +55 | +1.10 |
| v1 | 3.15 | 3259/4991 = 65.30% | 3816/4991 = 76.46% | 566 | 9 | +557 | +11.16 |
| v2 | 3.14 | 630/1915 = 32.90% | 656/1915 = 34.26% | 40 | 14 | +26 | +1.36 |
| v2 | 3.15 | 485/2078 = 23.34% | 692/2078 = 33.30% | 215 | 8 | +207 | +9.96 |

Every unit had exactly one timeout, and it was the same source (`a8f152dd…`) in all eight units. No
unit had any other errors.

## Against October (100k backbone, rank-8 LoRAs on statement and segmentation)

| Release | Python | October, with LoRAs | Now, with LoRA | October, no LoRA | Now, no LoRA |
|---|---|---:|---:|---:|---:|
| v1 | 3.14 | 77.44% | 81.43% | 76.84% | 80.32% |
| v1 | 3.15 | 70.55% | 76.46% | — | 65.30% |
| v2 | 3.14 | 19.95% | 34.26% | 19.01% | 32.90% |
| v2 | 3.15 | 15.35% | 33.30% | — | 23.34% |

This comparison is not controlled. Between the two runs, several things changed at once:
- the decompiler runtime (upstream dev was merged after the October benchmark);
- the backbone corpus (50k instead of 100k) and the tokenizer;
- LoRA rank (64 instead of 8);
- segmentation LoRAs (not used now);
- the GPU (RTX 4090 instead of A6000).

The size of the jump on v2 points mostly to the runtime. Only the paired LoRA comparisons above
isolate a single factor.

## Reading

- The 3.15 LoRA gives the backbone a 3.15 capability it lacks (+11.2 pp on v1, +10.0 pp on v2), and
  it almost never regresses a file (9 and 8 files).
- The 3.14 LoRA adds about 1 pp on a backbone that already saw 3.14 in training. That is about the
  same as October's rank-8 gain of +0.60 / +0.94 pp.
- These results come from one training seed, and there is no ordinary per-version PyLingual control.

## Artifacts

- Local copy of the per-file journals, selections, runtime configs, provenance and logs:
  `~/docs/pylingual-experiments/shared-training-v1-20261028/bench-results/`
- Combined report: `bench-results/results/report.json`
- Host copies, including decompiled sources: `~/pylingual-bench/results` on triplesec1 and triplesec2.
