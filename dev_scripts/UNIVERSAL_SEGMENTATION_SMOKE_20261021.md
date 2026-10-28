# Universal-only segmentation smoke: 2026-10-21

## Scope

- One fixed universal segmentation checkpoint and tokenizer; no adapters.
- Official evaluation dataset v1.1.0, Python 3.6–3.15.
- The same 1,000 selected source identities were attempted for each version.
- Zero known training/validation overlap by the recorded source-identity audit.
- GPUs 2 and 3 ran inference; ongoing statement-LoRA training on GPUs 0 and 1 was not interrupted.
- These are segmentation label/boundary metrics, **not end-to-end decompilation success rates**.

## Results

| Python | Files scored / attempted | Instruction-label accuracy | Boundary F1 |
|---|---:|---:|---:|
| 3.6 | 999 / 1000 | 99.503% | 98.994% |
| 3.7 | 999 / 1000 | 99.505% | 99.002% |
| 3.8 | 1000 / 1000 | 99.794% | 99.648% |
| 3.9 | 1000 / 1000 | 99.795% | 99.656% |
| 3.10 | 1000 / 1000 | 99.731% | 99.544% |
| 3.11 | 1000 / 1000 | 99.773% | 99.625% |
| 3.12 | 1000 / 1000 | 99.689% | 99.506% |
| 3.13 | 1000 / 1000 | 99.686% | 99.508% |
| 3.14 | 999 / 1000 | 99.661% | 99.391% |
| 3.15 | 999 / 1000 | 99.010% | 97.959% |

## Caveats

The same `_inputs.pyc` source exceeded the fixed 10-second CPU preparation limit for 3.6, 3.7, 3.14, and 3.15. Those versions scored 999 files; the remaining versions scored all 1,000. Failed preparation attempts are recorded, not silently counted as successful predictions. Consequently, the scored source sets are not exactly identical across all versions; their intersection is 999 sources.

The checkpoint and source identities were verified equal across both worker manifests. This is an encouraging segmentation-generalization check, not a replacement for the queued adapted 3.14/3.15 full-decompiler benchmarks.

## Artifacts

- Aggregated JSON: `/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/full-retrain-20261020/universal-segmentation-all-versions-v1-1000.json`
- Remote JSON: `/home/henryh/pylingual-tokenizer-100k-20261019/full-retrain-20261020/universal-segmentation-all-versions-v1-1000-20261021/report.json`
- Worker metadata: `/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/full-retrain-20261020/universal-segmentation-all-versions.job.json`
- Dataset commit: `a60d3b95748283ba0fbcd5e94da307d5bd5df870`
- Backbone SHA-256: `cd8459125f3829663d68ccd94496d46276dbb95dbab15cc6dd50f31ecd5ab3be`

## Reproduction

Use `dev_scripts/smoke_universal_segmentation_versions.py` with the same family, source audit inputs, dataset commit, and selection seed recorded in the JSON. The default selects 1,000 common eligible sources across all ten versions. `--versions` partitions inference only; source selection remains common across all versions. `--skip-checkout` is only safe after all selected blobs have been fetched.

```sh
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /home/henryh/pylingual-tokenizer-100k-20261019/repo/.venv/bin/python -u -m dev_scripts.smoke_universal_segmentation_versions \
  --run /home/henryh/pylingual-tokenizer-100k-20261019/full-retrain-20261020 \
  --output /path/to/new-results --files 1000
```
