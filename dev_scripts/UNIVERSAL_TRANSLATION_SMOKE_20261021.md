# Universal statement translation smoke — 2026-10-21

## Scope and result

User requested Python **3.14 only**, universal statement backbone **without LoRA**, on GPUs 2 and 3 of `syssec-gpu4`. No other versions were tested. Training on GPUs 0/1 was not interrupted.

- All **40,223** prepared statement examples from **500** target-validation files (450 PyPI / 50 CSN).
- Canonical source identities disjoint from the 100,000-file universal backbone/tokenizer pool and the 54,500-file 3.14 adapter training pool.
- **11,203 distinct masked bytecode inputs**; deterministic duplicate inputs generated once, then scored against every original reference row. No teacher forcing.
- Normal `VersionedTranslator(use_adapter=False)` path, FP32, greedy generation, maximum generation length 512. Ground-truth statement boundaries, not predicted segmentation.
- Compiler build: **CPython 3.14.6**.

| Population | Statements | Exact reference match |
|---|---:|---:|
| All | 40,223 | **99.018%** (39,828) |
| Input and target each within training limit of 256 tokens | 39,347 | **99.438%** (39,126) |
| Input or target exceeds 256 tokens | 876 | **80.137%** (702) |

No empty predictions, generation exceptions, or runtime fallback outputs after repair. Exact lexical-token agreement equals exact text agreement here. Across distinct bytecode inputs, **97.706%** matched at least one observed reference for that input. Some masked bytecode inputs have multiple source references, so that last metric is explicitly any-reference, not single-reference accuracy.

**Interpretation:** the universal translator looks strong on ordinary 3.14 statements. Long statements are a clear weak point. This is statement translation with gold boundaries, **not full-file decompilation accuracy or compiled-bytecode equivalence**. Many statement errors are real; some strict reference mismatches such as `return` versus `return None` can be equivalent.

## Decoder defect found and repaired

The initial run exposed a runtime bug, not evidence that the model generated empty code. Raw generated IDs decoded to correct Python, but `VersionedTranslator` removed every `tokenizer.all_special_ids` ID. The fitted BPE reports **863** IDs as special, including ordinary Python keywords, punctuation and masked identifiers. The filter therefore erased valid output.

`pylingual/models.py` now discards only explicit tokenizer control roles (BOS/EOS/PAD/UNK/MASK/CLS/SEP) and input-only opcode/reserved IDs. It preserves ordinary source tokens regardless of their special flag. Tokenizers, weights, model architecture, precision and training data were not changed. This repairs both adapted and unadapted translation paths; future benchmark subprocesses load the repaired module.

The regression reproduced the old failure (`return x` became `x`), passed after repair, and now also covers masked identifiers/punctuation and both adapter enablement settings. **29 tests passed** across embedding integration, tokenizer/opcode registry, and full-benchmark tests. Three stale benchmark-selection test fixtures were updated to the already-approved official release layout and overlap-reporting policy; production selection policy was not changed.

## Reproduction and artifacts

Driver: `dev_scripts/smoke_universal_translation.py`, phases `prepare`, `worker`, `combine`. Set `CUDA_VISIBLE_DEVICES=2` / `3` for two workers with `--shard 0` / `1`, `--shards 2`; all phases require `--run` and `--output`. Preparation audits source disjointness and saves a fixed plan; workers load and verify the same immutable family; combination requires every input and both terminal worker records.

Remote run:
`/home/henryh/pylingual-tokenizer-100k-20261019/full-retrain-20261020`

Successful report:
`/home/henryh/pylingual-tokenizer-100k-20261019/full-retrain-20261020/universal-translation-3.14-20261021/report.json`

Local report:
`/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/full-retrain-20261020/universal-translation-3.14.json`

Broken-decoder predictions were preserved separately under the remote directory suffix `-before-decoder-fix`; do not mix them with successful results. The report includes model/runtime fingerprints, ambiguity counts, length strata, worker timing/memory and mismatch examples. The generation script fingerprint describes the worker revision; the separately recorded analysis fingerprint describes the combination revision. No local raw source corpus staging.
