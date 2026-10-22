# Shared-backbone PyLingual ablation

> **Implementation history:** this guide describes the existing pilot scripts.
> The [corrected training design](SHARED_TOKENIZER_TRAINING.md) supersedes the
> training choices below: pooled bytecode-trained tokenizers, normal training
> settings, and 16 reserved opcode slots with LoRA + new-token embedding updates.
> The existing commands do not yet implement the new binding/embedding format.

PyLingual has two neural tasks: statement translation (T5) and instruction-boundary
segmentation (RoBERTa). This implementation shares **one backbone per task** across
versions, with small, unmerged query/value LoRAs. It also supports sharing only one
task while keeping the other task's ordinary per-version models.

## V1 study

* Translation starts from **`Salesforce/codet5-base`**, the upstream starting point
  used by ordinary PyLingual, **not** a fine-tuned PyLingual translator.
* Mixed-backbone training uses equal successful file counts for **3.6–3.14**.
  **No 3.15 data updates the backbone or its tokenizer.**
* Train only **3.14 and 3.15 adapters**: 3.14 is in-distribution for backbone
  training; 3.15 is excluded from backbone training but has adapter training data.
  This is adaptation to an unseen backbone version, **not zero-shot 3.15**.
* The optional shared segmenter warm-starts from the existing 3.9 segmenter, so
  the segmentation experiment is not a from-generic-pretraining ablation.
* For the translation ablation, keep the ordinary per-version segmenters fixed.
  Compare specialist translation, shared translation without LoRA, and shared
  translation with the matching LoRA on exactly the same files. The fully shared
  pipeline is an additional, separate feasibility measurement.
* All translators remain base-sized (~223M parameters). This does not test larger
  backbones. Deployed specialists are useful references, not matched-data or
  matched-compute controls: their training data and budgets differ.

## Runtime

```yaml
SHARED_BACKBONES:
  SEGMENTATION_MODEL: models/segmentation
  STATEMENT_MODEL: models/statement
```

Paths are relative to the YAML file. Existing configurations still work. Either
shared task may be omitted if ordinary `v3.14`, `v3.15`, etc. entries provide that
task. For the translation-only ablation, omit `SEGMENTATION_MODEL` from this section
and retain the standard per-version entries. Add `USE_ADAPTERS: false` to bypass
LoRAs without changing the backbone or the requested version's translation prefix.

Pass the configuration to `decompile(config_file=...)` or the CLI's config option.
A missing requested adapter is an error, not a fallback to another version.

Each family contains `backbone/`, `tokenizer/`, `adapters/<version>/`, and
`family.json`. The registry shares a family once per process/device. Separate worker
processes still have separate copies. Families are immutable compatibility units;
refreshing weights requires a new identity and retraining/revalidating adapters.

A per-backbone reentrant lock covers adapter selection and complete inference,
including lazy segmentation iterators, generation retries, and per-version output
caches. This is correctness-first serialization, not concurrent multi-adapter
batching. Checkpoints check backbone identity, layers and shapes before loading.
The project-local PyTorch LoRA format requires no PEFT dependency and is not
PEFT-compatible. Adapters are never merged into the shared weights.

## Sample existing archives

Use an existing `pypi-scraper-new` run; no scraping or downloaded package execution
is needed. Installed interpreters must exist for every requested version.

```sh
python -m dev_scripts.sample_shared_dataset \
  --archives /path/to/2026-07-02-14 --output /path/to/sample

python -m dev_scripts.sample_shared_dataset \
  --archives /path/to/2026-07-02-14 --output /path/to/sample-adapter315 \
  --versions 3.15
```

Defaults: 64 successful training files and 16 validation files per target version.
The first command defaults to 3.6–3.14, excluding 3.15. The manifest records quotas,
archive/member provenance, hashes, seed, and conversion failures. Each file supplies
at most 64 statement rows and eight segmentation rows. Uniform file sampling and
round-robin version selection prevent large files or versions dominating training.
The global canonical-source split keeps equivalent parsed sources on the same side
of the train/validation split, including the separate 3.15 run with the same seed.
Keep raw data, weights, credentials and caches outside Git.

## Train

Use the existing project training environment with CUDA. No Hub uploads are made.

```sh
python -m dev_scripts.shared_training \
  --data /path/to/sample --adapter-data /path/to/sample-adapter315 \
  --task statement --output /path/to/models/statement \
  --steps 6000 --adapter-steps 600 --adapter-versions 3.14 3.15

python -m dev_scripts.shared_training \
  --data /path/to/sample --adapter-data /path/to/sample-adapter315 \
  --task segmentation --output /path/to/models/segmentation \
  --steps 1500 --adapter-steps 600 --adapter-versions 3.14 3.15 \
  --batch-size 8 --accumulation 1
```

Backbone training never consumes the additional adapter-only rows. After saving the
backbone and fixed tokenizer, freeze both and train only the requested adapters.
Family metadata records backbone versus adapter training versions separately,
per-version file counts, dropped long rows, unknown input tokens, hyperparameters,
manifest hashes and training history. The family becomes loadable only when every
requested adapter is saved. Outputs must not exist; no destructive resume occurs.

To continue backbone training, use `--statement-init <family>/backbone` and
`--statement-tokenizer <family>/tokenizer` (or the segmentation equivalents), a new
output, and `--backbone-lineage <family>/family.json`. Use `--steps 0` to reuse an
already trained unwrapped backbone and train just the requested adapters. This
publishes a new identity; it is not the identity-preserving update command below.

### Future adapter-only updates

```sh
python -m dev_scripts.train_shared_adapter \
  --family /path/to/models/statement \
  --data /path/to/new-version-statement.jsonl --version 3.15 \
  --output /path/to/models/statement-adapter-update --steps 600
```

The command copies artifacts unchanged into a new family directory, preserves the
backbone identity and trains only the selected adapter. It never changes tokenizer
vocabulary. It reports unknown input tokens: distinct opcodes collapsed to `[UNK]`
cannot acquire distinct input representations through LoRA alone.

## Evaluate

Use **pylingualv2 `.pyc` files** from the local `pylingual_eval_dataset` checkout,
not the similarly named Hugging Face statement/segmentation training repositories.

```sh
python -m dev_scripts.evaluate_shared \
  --dataset /path/to/pylingual_eval_dataset --data /path/to/sample \
  --adapter-data /path/to/sample-adapter315 --versions 3.14 3.15 \
  --output /path/to/results/specialists --files-per-version 8 --timeout 30

python -m dev_scripts.evaluate_shared \
  --dataset /path/to/pylingual_eval_dataset --data /path/to/sample \
  --adapter-data /path/to/sample-adapter315 --versions 3.14 3.15 \
  --families /path/to/models --config /path/to/translation-only.yaml \
  --selection /path/to/results/specialists/selection.json \
  --output /path/to/results/shared --files-per-version 8 --timeout 30
```

For no-LoRA component metrics pass `--disable-adapters`; the runtime config must
also set `USE_ADAPTERS: false`. Do not reuse external prediction caches.

Selection is seeded, balanced and reusable across configurations. The evaluator
rejects source-identifier overlap with either data manifest, changed bytecode hashes,
paths outside the dataset, and incorrect version/file quotas. Without v2 source
text this cannot exclude differently formatted or near-duplicate source. Overlap
with the deployed specialists' original training data is not established.

Metrics:

* **File accuracy:** every equivalence check succeeds and at least one exists.
  Errors and timeouts count as failures; categories are reported separately.
* **Code-object counts:** equal/total objects among files reaching verification;
  these conditional counts are **not** error-inclusive code-object accuracy.
* **Statement exact match:** up to 64 sampled-source validation statements/version,
  ignoring surrounding whitespace; not semantic accuracy and not the v2 file score.
* **Segmentation token accuracy/boundary F1:** sampled-source validation rows.

`file_results.jsonl` flushes each file. `summary.json`, `selection.json`, and
`component_metrics.json` preserve results. Tiny training data and eight v2 files
per version cannot establish production parity. Training loss is not accuracy.
The evaluator counts timeouts explicitly and excludes compiler diagnostics from
code-object counts. The compiler helper accepts successful prerelease-interpreter
warnings, fails on unsuccessful compilation, and strips wrapper traceback line
numbers so error correction sees the actual generated-source line.

## Tests

```sh
python -m unittest test.test_shared_models -v
```

No Hub downloads. Tests cover frozen backbone gradients, adapter independence,
checkpoint/compatibility guards, concurrent lazy pipeline selection, real tiny T5
generation and RoBERTa training, generation retries/deadlines, shared/legacy routing,
no-LoRA version prefixes, rejection of 3.15 in default backbone data groups, and
compiler success/failure/diagnostic handling for prerelease Python interpreters.

## First pilot results

The [2026-10-16 v1 report](experiments/shared-v1-2026-10-16.md) records the exact
protocol, component checks, strict v2 file outcomes, and model-weight footprint.
The architecture saves weight memory, but this deliberately tiny run does not
match deployed 3.14 whole-file accuracy. It is not a production replacement.
