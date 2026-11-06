# Shared v3: all Python versions — 2026-11-06

The approved extension to **Python 3.6–3.13 is complete**: **51,144/51,144 evaluations** across
16 base-only units. Independent host and workstation audits found no missing or duplicate
rows, no model-load failures, and zero pooled raw-source overlap. No training or tokenizer
fitting was performed. The original 3.14/3.15 evaluation was not rerun.

## Protocol and immutable artifacts

- Frozen universal statement backbone: `syssec-utd/pylingual-shared-statement-v3` at
  `a20bc4d28fe8641df0f90aa3665431368930c329`.
- Frozen universal segmenter: `syssec-utd/pylingual-shared-segmentation-v3` at
  `37cfee8f8e7bc16c4050056ffcdead83f714c7cf`.
- Frozen tokenizer: `syssec-utd/pylingual-shared-tokenizer-v1` at
  `d23233a4bb237dcc3b1660e46a42f9189eaf0f6c`.
- **No statement or segmentation LoRAs** for any of the eight new versions.
- Strict full-file bytecode equivalence, FP32, 300 seconds/file, four torch threads,
  `TORCH_DISABLE_NATIVE_JIT=1`; all committed listed files retained and failures counted.
- Eval v1.1.0 at `a60d3b95748283ba0fbcd5e94da307d5bd5df870`; eval v2.0.1 at
  `258bc0380f7c8550fa28f294b896cf168a5e4055`.
- Scoring source `84c65f938ee45c6175beb3b9b385c9550f2f0f3a`. Seven scoring/model-building
  functions are AST-identical to the earlier `197c449` source; only version/checkout/report
  scope was extended. Original 3.14/3.15 defaults are unchanged.
- Triplesec1 evaluated 3.6–3.9, triplesec2 evaluated 3.10–3.13; one full unit per GPU.
  Separate two-file smokes are excluded from the official counts.
- Native verification compilers: 3.6.15, 3.7.17, 3.8.20, 3.9.25, 3.10.20, 3.11.15,
  3.12.13 and 3.13.14. Earlier 3.14/3.15 results used 3.14.6 and 3.15.0b2.

The original frozen training plan is unchanged (SHA-256
`0567ed821c5605e850fc3e0be8ff4667d25b2cfc89e888e4125e4121cc1cab63`). Its 150k backbone
covered every version from 3.6 through 3.15. These are **in-distribution version evaluations**,
not unseen-version transfer.

## Full-file accuracy: same base-only v3 family

| Python | Eval v1 exact files | Eval v1 | Eval v2 exact files | Eval v2 |
|---|---:|---:|---:|---:|
| 3.6 | 4114/4780 | 86.07% | 316/1106 | 28.57% |
| 3.7 | 4258/4949 | 86.04% | 361/1198 | 30.13% |
| 3.8 | 4286/4987 | 85.94% | 383/1284 | 29.83% |
| 3.9 | 4301/4987 | 86.24% | 354/1201 | 29.48% |
| 3.10 | 4250/4991 | 85.15% | 431/1261 | 34.18% |
| 3.11 | 4334/4991 | 86.84% | 406/1335 | 30.41% |
| 3.12 | 4288/4962 | 86.42% | 489/1362 | 35.90% |
| 3.13 | 4183/4962 | 84.30% | 1734/2788 | 62.20% |
| 3.14 | 4137/4991 | 82.89% | 711/1915 | 37.13% |
| 3.15 | 4124/4991 | 82.63% | 846/2078 | 40.71% |

3.14/3.15 rows are the already published no-LoRA results, not new evaluations.
Each Python version has a different official input list, and eval v1/v2 differ as well.
In particular, 3.13 has 2,788 eval-v2 files versus 1,106–1,362 for the other new versions.
The percentages are **not a paired cross-version difficulty comparison**; the higher 3.13
eval-v2 score does not by itself show a better model for that version.

## Original 3.14/3.15 statement-LoRA comparisons (separate completed experiment)

| Python | Eval v1 base → LoRA | Eval v2 base → LoRA |
|---|---:|---:|
| 3.14 | 82.89% → 83.25% | 37.13% → 37.44% |
| 3.15 | 82.63% → 82.93% | 40.71% → 40.86% |

No equivalent LoRA arm exists for 3.6–3.13; none was trained or implied by this extension.
See [the original v3 report](SHARED_V3_BENCHMARK_20261105.md) for paired gains/regressions.

## Failure accounting

| Eval | Python | Equal | CompileError | Different | Error | Timeout |
|---|---|---:|---:|---:|---:|---:|
| v1 | 3.6 | 4114 | 232 | 434 | 0 | 0 |
| v1 | 3.7 | 4258 | 257 | 434 | 0 | 0 |
| v1 | 3.8 | 4286 | 271 | 430 | 0 | 0 |
| v1 | 3.9 | 4301 | 259 | 427 | 0 | 0 |
| v1 | 3.10 | 4250 | 244 | 497 | 0 | 0 |
| v1 | 3.11 | 4334 | 302 | 355 | 0 | 0 |
| v1 | 3.12 | 4288 | 308 | 365 | 0 | 1 |
| v1 | 3.13 | 4183 | 337 | 442 | 0 | 0 |
| v2 | 3.6 | 316 | 351 | 439 | 0 | 0 |
| v2 | 3.7 | 361 | 387 | 450 | 0 | 0 |
| v2 | 3.8 | 383 | 436 | 465 | 0 | 0 |
| v2 | 3.9 | 354 | 387 | 460 | 0 | 0 |
| v2 | 3.10 | 431 | 373 | 457 | 0 | 0 |
| v2 | 3.11 | 406 | 558 | 371 | 0 | 0 |
| v2 | 3.12 | 489 | 534 | 338 | 0 | 1 |
| v2 | 3.13 | 1734 | 575 | 479 | 0 | 0 |

**38,488 Equal, 5,811 CompileError, 6,843 Different and 2 Timeout outcomes** sum to all
51,144 files. There were zero runtime `Error` outcomes and zero model-load failures.
Both timeouts occurred in Python 3.12 (one per release), remained failures, and were not
discarded or rerun. Full paths, identities and traces remain in the journals and manifest.

## Independent verification and retained evidence

The final read-only host audits rehashed all 18 reused model/identity files, every selected
PYC and committed list, and all 69 runtime source files. Every PYC was also checked against
its committed Git blob, including Unicode filenames. The workstation then matched the
snapshots and merged journals to the host digests, independently reconstructed outcomes,
and verified FP32 parameter byte counts, disabled adapters, unchanged family identities,
compiler versions/binary hashes, and the seven unchanged scoring/model functions.

The source pool was independently rebuilt from 552,000 accepted journal records to exactly
1,052,604 identifiers. All evaluated raw source identifiers are absent. This is raw
source/identifier membership, not a rederivation of canonical equivalence or project-disjointness.
The official reporter exactly reproduced the independent fixed counts, successes and categories.
Nine applicable unit tests passed; six audit negative controls rejected dropped/duplicate rows,
changed PYC identities, fake successes, model-load failures and falsely claimed overlap.

- Machine-readable results/provenance:
  [`experiments/shared-v3-other-versions-benchmark-2026-11-05.json`](experiments/shared-v3-other-versions-benchmark-2026-11-05.json).
- Local evidence root: `/home/henry/docs/pylingual-experiments/shared-training-v3-150k-20261102/other-versions-benchmark`.
- Receipts: `completion-verified-20261106.json`, `final-host-audit-triplesec1-gpu.json` and
  `final-host-audit-triplesec2-gpu.json`; complete source fingerprints and script hashes are recorded.
- Host-specific, nondeleting backups: `hosts/{triplesec1-gpu,triplesec2-gpu}`. Only the new
  disjoint-version result directories were merged into `results`; original results are untouched.
- Host2 staged inputs/results lived in `/dev/shm` and were copied to workstation storage after
  both lanes completed. No unrelated data, caches, models or services were modified or deleted.
- Both GPUs are idle. The dedicated native-compiler/SSH-forward service was stopped after
  both complete snapshots were backed up. Model/identity symlinks remained read-only.

One training seed; no ordinary per-version control models, no controlled performance claim,
and no production-correctness guarantee are established by these bytecode-equivalence results.
