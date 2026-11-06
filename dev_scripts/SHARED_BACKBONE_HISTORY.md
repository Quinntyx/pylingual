# Shared-backbone and LoRA experiment: historical snapshot

This branch preserves the shared-backbone experiments, training tools, immutable public artifact
references, and completed benchmarks. It is a **historical experiment, not the proposed runtime
implementation to merge into `dev`**. Keep the original code, manifests and measurements intact.

## Decision — 2026-11-06

Proceed with one base segmentation model and one base statement-translation model, without LoRAs.
The successor starts from the fork's current `dev`, incorporates upstream `dev`, and implements
only the runtime changes needed for the two base models. It must not merge this experimental
branch wholesale or carry its adapter banks, embedding updates and training orchestration forward.

The paired v3 statement-LoRA results added only **0.14–0.36 percentage points** of full-file accuracy,
with regressions as well as newly correct files. This motivated the simpler implementation choice;
it is not proof that adapters are never useful. All Python versions 3.6–3.15 were present in v3
backbone training, and these measurements use one training seed.

## Preserved evidence

- [Original v3 3.14/3.15 benchmark](SHARED_V3_BENCHMARK_20261105.md): 27,950 evaluations,
  matched base/statement-LoRA arms, paired gains/regressions, and failure accounting.
- [Base-only evaluation across Python 3.6–3.15](SHARED_V3_OTHER_VERSIONS_BENCHMARK_20261106.md):
  51,144 additional evaluations for 3.6–3.13, with the earlier 3.14/3.15 base scores for reference.
- [Frozen v3 training plan](experiments/shared-v3-main-150k-2026-11-02.json): approved 150k-file
  backbone and adapter pools, the unchanged tokenizer, and final public checkpoint revisions.
- [Original benchmark provenance](experiments/shared-v3-benchmark-2026-11-05.json).
- [Remaining-version benchmark provenance](experiments/shared-v3-other-versions-benchmark-2026-11-05.json).

Both benchmark reports retain failures and timeouts. The independent audits checked fixed counts,
input identities, model/runtime fingerprints, FP32 inference and zero pooled raw-source overlap.
Different Python versions have different official input lists; cross-version percentages are not
paired comparisons. Raw-source overlap checks do not establish project-disjointness.

## Base artifacts for the clean successor

| Artifact | Public Hugging Face repository | Immutable revision |
|---|---|---|
| Statement translation | `syssec-utd/pylingual-shared-statement-v3` | `a20bc4d28fe8641df0f90aa3665431368930c329` |
| Segmentation | `syssec-utd/pylingual-shared-segmentation-v3` | `37cfee8f8e7bc16c4050056ffcdead83f714c7cf` |
| Tokenizer family | `syssec-utd/pylingual-shared-tokenizer-v1` | `d23233a4bb237dcc3b1660e46a42f9189eaf0f6c` |

These are the trained universal base models, not unfine-tuned generic pretrained models. Their
input preprocessing, tokenizer alignment and version prefixes remain compatibility requirements;
removing adapters is not permission to change the trained model's input representation.

No new training, tokenizer fitting or benchmark reruns are part of archiving this work. The
completed run's compiler services and scheduled monitoring are stopped; raw journals and audit
receipts remain in the experiment workspace identified by the reports.

The older [pilot runtime guide](SHARED_MODELS.md) and
[tokenizer/adapter design notes](SHARED_TOKENIZER_TRAINING.md) remain as historical implementation
records, not instructions to restore LoRAs in the clean successor.
