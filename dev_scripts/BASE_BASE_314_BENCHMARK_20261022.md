# Universal-only Python 3.14: full official file benchmarks

Completed October 22, 2026. Universal statement translator + universal segmenter; **both LoRAs disabled**. This is the no-adapter arm, not an ordinary per-version PyLingual checkpoint.

| Dataset release | Full official set | No pooled-source overlap | Known pooled-source overlap |
|---|---:|---:|---:|
| v1.1.0 | 3835/4991 = **76.84%** | 3530/4665 = **75.67%** | 305/326 = **93.56%** |
| v2.0.1 | 364/1915 = **19.01%** | 355/1892 = **18.76%** | 9/23 = **39.13%** |

## Protocol and caveats

- Strict **full-file bytecode equivalence**: every verification check must pass. These are not statement exact-match percentages.
- Complete official Python 3.14 lists: no sampling, error-row replacement, or excluded failures. FP32 inference and a 300-second per-file timeout. Verification compiler: Python 3.14.6.
- Unique PYC identities and SHA-256 values were independently checked against pinned selections after copying raw journals, summaries, runtime, provenance, model statistics and validation reports locally. Both releases have identical runtime/weight fingerprints and protocol hash.
- Non-overlap means absence of the raw source SHA-512 from the **pooled study** train/validation corpus. It is conservative and is not proof of canonical source independence. Full official sets contain overlap; they are not all held out.
- Training: completed 100k-file universal runs, two epochs each, with balanced Python 3.6–3.14 inputs and no 3.15 backbone/tokenizer training. The 3.14/3.15 adapters were also trained but are disabled here.
- This corrected-runtime full rerun is separate from earlier provisional attempts. No results were mixed between attempts.
- The remaining independent `formats.pyc` NetworkX error is retained as a failure, not dropped. Earlier reconstruction bookkeeping and decoder repairs are documented separately.
- No ordinary per-version control was retrained. Adapted-model comparisons and two-version resident VRAM/performance measurements are still running.

## Failure counts

| Release | CompileError | Different | Timeout | Other error |
|---|---:|---:|---:|---:|
| v1 | 502 | 651 | 3 | 0 |
| v2 | 836 | 713 | 1 | 1 |

## Artifacts

- Machine-readable report: `/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/full-retrain-20261020/base-base-3.14.json`
- Per-file journals and pinned artifacts: `/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/full-retrain-20261020/base-base-3.14-20261021-retryfix`
- Transfer audits: `v1-artifact-transfer-audit.json` and `v2-no_adapter-3.14-artifact-transfer-audit.json`.
- Runtime repair evidence: `RECONSTRUCTION_RETRY_REPAIR_20261021.md`.
