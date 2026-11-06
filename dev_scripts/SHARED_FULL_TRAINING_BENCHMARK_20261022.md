# Shared PyLingual: full training and benchmark results — 2026-10-22

## Outcome

Training and all six official full-file quality passes are complete and independently audited. Both inference components share a single backbone across versions. LoRAs improve Python 3.14 modestly: **+0.60108 percentage points on v1**, **+0.93995 points on v2**. Python 3.15 adaptation is usable but substantially weaker in absolute file accuracy; there is no matched ordinary-PyLingual or no-LoRA 3.15 control.

## Exact full-file accuracy

A file passes only when all recompilation/bytecode-equivalence checks pass, including extra-code-object checks. This is not statement-string accuracy, teacher-forced token accuracy, or executable behavioral testing. Inference is FP32 with a 300-second per-file deadline. Every official listed file is retained; timeouts and control-flow errors are failures.

| Eval release | Python | Models | Full exact files | Full % | Non-overlap exact files | Non-overlap % | Pooled-overlap exact files |
|---|---|---|---:|---:|---:|---:|---:|
| v1 | 3.14 | Universal, no LoRAs | 3835/4991 | 76.8383 | 3530/4665 | 75.6699 | 305/326 |
| v1 | 3.14 | Universal + both LoRAs | 3865/4991 | 77.4394 | 3559/4665 | 76.2915 | 306/326 |
| v1 | 3.15 | Universal + both LoRAs | 3521/4991 | 70.5470 | 3229/4665 | 69.2176 | 292/326 |
| v2 | 3.14 | Universal, no LoRAs | 364/1915 | 19.0078 | 355/1892 | 18.7632 | 9/23 |
| v2 | 3.14 | Universal + both LoRAs | 382/1915 | 19.9478 | 373/1892 | 19.7146 | 9/23 |
| v2 | 3.15 | Universal + both LoRAs | 319/2078 | 15.3513 | 313/2053 | 15.2460 | 6/25 |

“Non-overlap” excludes conservative raw-source SHA-512 membership in the pooled study corpus, including sources not necessarily exposed to a particular arm. It does **not** establish canonical source equivalence, project-disjointness, or that every official file is held out. Pooled overlaps: v1 326/version; v2 23 for 3.14 and 25 for 3.15. Release tags are **v1.1.0** and **v2.0.1**. Python builds are **3.14.6** and **3.15.0b2**, not final 3.15. The v2 version lists differ in size, so cross-version accuracy differences do not isolate a model effect.

## Paired 3.14 ablation

The following comparisons use identical PYC identities, runtime code and frozen backbone artifacts, with only the two version LoRAs enabled/disabled. They are one trained seed/run, not replicated training experiments.

| Release | Newly correct | Regressed | Net correct | Full gain, pp |
|---|---:|---:|---:|---:|
| v1 | 46 | 16 | +30 | +0.60108 |
| v2 | 29 | 11 | +18 | +0.93995 |

The non-overlap gains are +0.62165 pp on v1 and +0.95137 pp on v2. Do not interpret these as parity or superiority to ordinary PyLingual: all six ordinary control jobs were cancelled before their first attempt, as requested. External deployed scores are not matched controls.

## Training and coverage

- Universal translator initialized from generic `Salesforce/codet5-base`, not a PyLingual checkpoint. Segmentation used a fresh RoBERTa MLM stage followed by segmentation fine-tuning.
- Universal/tokenizer corpus: **100,000 files**, 90,000 PyPI + 10,000 CSN, balanced across Python 3.6–3.14; **3.15 excluded**. All three universal training stages completed two epochs.
- Only 3.14 and 3.15 adapters trained: **54,500 training + 500 validation files per target**, 90% PyPI / 10% CSN, two epochs. 3.15 is out of the universal training distribution, **not** out of adapter training.
- Custom rank-8 LoRA, alpha 16, dropout 0.05. Training used FP16 for translation/segmentation/adapters and FP32 for MLM; inference measurements here are FP32.
- Statement adapters: 884,736 parameters/version; segmentation: 294,912/version. Both versions retain the same frozen backbone hash per component.
- Actual reserved opcode-slot usage was **zero for both targets**. New-opcode embedding behavior was tested synthetically, not exercised by this real-data ablation.
- Invalid earlier pilot/checkpoint runs are excluded. This report uses only the completed full-run families and the complete corrected-runtime benchmark attempts.

## Resident storage and runtime

With **both versions' adapters resident simultaneously** on an NVIDIA RTX A6000:

- CUDA allocated after model loading: **1.1979 GiB**; CUDA reserved: **1.3320 GiB**.
- Unique model parameter/buffer storage: **1.1971 GiB**.
- LoRAs cost **4.5 MiB per version** (both components), **9 MiB for both versions** in FP32.
- Shared statement and segmentation model-object identities were both verified.
- Small-batch microbenchmark peak allocated: **1.3051 GiB**. This is **not full-file decompilation peak VRAM**; CUDA context/driver allocations and process RSS are outside these PyTorch counters.
- Duplicating these exact backbones per version would use 2.3854 GiB of model storage instead of 1.1971 GiB (49.82% less). This is **parameter/buffer arithmetic**, not a measured stock-PyLingual VRAM comparison.

The following are synchronized model-call timings on pretokenized CUDA inputs, batch size 8, four CPU threads, two warmups and ten measured repetitions. They include Python/model dispatch and GPU work, not just CUDA kernel execution, and exclude parsing/reconstruction, tokenization, equivalence compilation and output-cache hits. Version workloads differ in lengths. No comparative speedup is established.

| Python | Component | Input shape | Median batch ms | Min–max ms | Examples/s | Generated IDs/s | Peak allocated GiB |
|---|---|---|---:|---:|---:|---:|---:|
| 3.14 | statement | [8, 135] | 336.031 | 331.141–347.376 | 23.81 | 322.03 | 1.3014 |
| 3.14 | segmentation | [8, 389] | 44.804 | 44.622–45.170 | 178.56 | n/a | 1.3051 |
| 3.15 | statement | [8, 122] | 295.968 | 294.115–315.398 | 27.03 | 302.56 | 1.2930 |
| 3.15 | segmentation | [8, 378] | 42.698 | 42.543–42.937 | 187.36 | n/a | 1.3023 |

Raw timings are available only as median/min/max aggregates in the original measurement output; individual repetition durations were not retained. No p95/p99 is inferred from those ten samples. Generated-ID throughput counts nonpadding generated IDs, including control IDs. Model-object loading was 1.668 s for the first version and 0.069 s for the second; CUDA initialization was excluded and OS page-cache state was uncontrolled. This is not process cold-start or isolated adapter-switch latency.

## Failure handling and correctness

All 20,881 file evaluations across the six passes were audited for uniqueness, input hashes, counts, success categories and overlap strata. No rows were dropped. There were zero model-load failures. The same 54 runtime/evaluator source-file fingerprints were used in all arms, and the five frozen-backbone artifacts matched between base and adapted families.

`formats.pyc` retains its independent `NetworkXError` in all three v2 passes and counts as failure. CompileError, Different and Timeout categories remain in the denominators. The preceding reconstruction-bookkeeping repair had 75 passing focused tests, seven negative-control regressions, and nine causally demonstrated real-GPU repairs. Those fixes changed runtime bookkeeping, not weights/tokenizers; all final arms used the same fixed source. Earlier provisional attempts are archived separately and never combined with these passes.

## Limits and interpretation

Sharing is demonstrated by actual model identity and storage measurements. Accuracy gains from adding LoRAs to this universal pair are positive but small; v2 full-file accuracy remains low. The high gold-boundary statement smoke accuracy from earlier work must not be substituted for these full-file results. No ordinary-PyLingual quality/speed comparison, training-seed uncertainty, novel-opcode real-data transfer result, or full-file peak VRAM measurement is established. File wall-time distributions in the JSON are descriptive, not controlled speedup comparisons: different failure subsets, cache states, GPU assignment and concurrent jobs affect them.

## Artifacts

- Local workspace: `/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/full-retrain-20261020`
- Remote workspace on syssec-gpu4: `/home/henryh/pylingual-tokenizer-100k-20261019/full-retrain-20261020`
- Consolidated audited result: `benchmark/shared-study-results.json`
- Original scheduler result: `benchmark/report.json`
- Per-file results, selections, runtime configs, fingerprints, model stats and transfer audits: `benchmark/{v1,v2}/adapted/{3.14,3.15}/`
- Corrected universal-only baseline: `base-base-3.14-20261021-retryfix/`; consolidated baseline `base-base-3.14.json`
- Paired comparisons: `benchmark/v1-paired-comparison.json`, `benchmark/v2-paired-comparison.json`
- Runtime captures and configs: `benchmark/performance/adapted/`
- Hardware/software inventory: `benchmark/environment.json`
- Detailed performance scope: `benchmark/performance-report.md`
- Training completion audit: `full-training-completion-audit.json`; published family metadata `published-*-family.json`
- Final completion/fingerprint audit: `full-study-completion-audit.json`
