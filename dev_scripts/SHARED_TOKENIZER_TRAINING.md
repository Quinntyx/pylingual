# Shared tokenizers, version LoRAs, and future-opcode embeddings

## Status and scope

This is the agreed design and retraining checklist for the shared-backbone study.
The 100k-file tokenizers are fitted and independently verified. Production
LoRA-plus-new-token input-embedding training, adapter-only checkpoint resume,
family assembly, and normal model-loader/runtime integration are now implemented
and GPU-tested (section 12). These are engineering acceptance tests, not completed
54,500-file adapter runs or decompilation-quality benchmarks.

The user accepted the transient FP16 startup warning as non-blocking; retain the
normal precision settings. Non-finite losses/parameters or backbone mutation still
fail. Full retraining was authorized on 2026-10-20 and is now running on GPU4
(section 13); completed accuracy benchmarks are still pending.

PyLingual has two tasks: CodeT5 statement translation and RoBERTa instruction-boundary
segmentation. Share one backbone and one fixed tokenizer **per task**, rather than
trying to use one tokenizer or architecture for both tasks. Each Python version gets
its own LoRA and, when necessary, small new-token embedding updates.

The stock-CodeT5-tokenizer experiment and its checkpoints are not the corrected
experiment. Restart the model training from the normal initializations; do not
continue those weights. Previously selected source files may be reused, but extract
all examples and rebuild tokenized datasets under the corrected tokenizers.

## 1. Data contracts

- The current tokenizer stage uses **100,000 backbone-only training files**, not
  statement rows (increased from 50,000 on 2026-10-19). Mix Python **3.6–3.14**,
  with successful file counts differing by at most one because 100,000 is not
  divisible by nine. Extract all rows produced by the normal dataset converter,
  then apply the normal tokenization/windowing policies. Model training remains
  separately gated; this authorizes tokenizer training, not training model weights.
- Use **90% PyPI / 10% CSN by source-file count**: 90,000 PyPI and 10,000 CSN
  files for the current tokenizer stage. Record both file and resulting
  example counts per version and corpus; equal file counts do not guarantee equal
  row counts.
- Train generic tokenizers only on this backbone training partition. Do not include
  validation, test, adapter-only, or **3.15** data in vocabulary learning.
- For 3.14 adaptation, use **50,000 additional files** disjoint from backbone and
  held-out sources. The requested expansion also retains the existing 4,500 3.14
  training files, for **54,500 adapter training files**; build the new backbone
  partition so these retained files are not admitted there either. Count held-out
  files separately from these training budgets.
- V1 has only 3.14 and 3.15 adapters, each with **54,500 training files**.
  Future per-version datasets will also contain 54,500 training files when collected;
  this does not authorize training additional version adapters now. Keep held-out
  evaluation files separate. A trained 3.15 LoRA is unseen-version **adaptation**,
  not zero-shot evaluation.
- Deduplicate globally using raw hashes and a version-independent canonical-source
  hash. Split by source file, never by statement or code object. Preserve the
  benchmark exclusions across tokenizer, backbone, and adapter preparation.
- Read selected files directly on **`syssec-gpu4`** from the mounted NAS root
  `/syssec-nas1/pyc-research/datasets`. This replaces the unavailable
  `syssec-adwaith1:/data2` source and needs no source-host SSH transfer. Use only
  training directories for candidates and exclude identities found in any NAS
  validation/test directory, including 3.15. Do not stage the source corpus locally.
  Use existing versioned compile servers, recording interpreter versions,
  bytecode magic, flags, and transport checksums.

## 2. Train generic tokenizers before training models

Ordinary PyLingual learns tokenizers separately from each version's bytecode data.
The shared variant must instead learn each task's tokenizer from the pooled,
version-balanced backbone training files and then freeze its vocabulary and IDs.

### Statement tokenizer

Follow the template and bytecode-training procedure in
[`statement/train_tokenizer_auto.py`](statement/train_tokenizer_auto.py): bytecode
BPE, with the normal 30,000-token vocabulary budget. Do not silently substitute the
stock `Salesforce/codet5-base` tokenizer. Preserve and record the template's
normalization, pre-tokenization, special tokens, and decoding rules.

Give every known opcode in the training-version union an explicit atomic
representation, and allocate the 16 future-opcode slots below. Specify whether the
vocabulary budget includes these tokens; the recommended experiment keeps the total
budget fixed and includes them. Verify that all assigned IDs fit the model's
embedding table and do not collide with existing special tokens. Do not assume a
particular numeric range is free.

Use the normal statement preprocessing and input/target truncation. Do not silently
introduce a version prefix or change source masking. Any departure from the normal
pipeline must be recorded as an explicit experimental factor.

### Segmentation tokenizer

Follow [`segmentation/train_tokenizer.py`](segmentation/train_tokenizer.py): train
WordPiece on pooled backbone bytecode, preserving its normalizer, `<SEP>` instruction
splitting, structural tokens, and post-processor. Allocate its own known-opcode
representations and 16 reserved IDs; the numeric IDs need not match CodeT5's.

**Atomic opcode tokens must not accidentally become structural special tokens.**
In particular, do not exclude all opcodes from masked-language-model masking or
ignore their segmentation labels merely to make them atomic. Preserve an explicit
mapping from output tokens to original instruction indices when introducing the
opcode bindings; test this against the normal B/I/E label alignment.

Save both tokenizers, their training manifests, vocabulary/ID hashes, and opcode
registries before model training. Never train or replace a tokenizer independently
for a LoRA attached to the same backbone.

## 3. Reserve 16 future-opcode slots

Allocate exactly **16 unused opcode IDs per task** during initial tokenizer
construction, for example with registry names `<future_opcode_00>` through
`<future_opcode_15>`. Their actual integer IDs are recorded in the tokenizer
manifest. No current instruction or synthetic instruction meaning is assigned to
these slots during backbone training.

For a future Python version:

1. Enumerate its instruction identities as exposed by PyLingual's normal bytecode
   conversion and compare them with the backbone's known opcode registry. Use the
   actual compiler/disassembler and the same filtering/normalization as training,
   not a remembered opcode count or an unfiltered table of internal pseudo-ops.
   Count 3.15's unseen instructions before finalizing this capacity.
2. Map each unseen instruction to one reserved slot in that version's binding file.
   Instruction identity is not CPython's numeric opcode value; numbers can be reused
   and meanings can change across releases.
3. At the bytecode lexer/preprocessing boundary, map the **opcode field** to the
   fixed ID or placeholder. Do not globally replace mnemonic-looking strings in
   operands, Python identifiers, string literals, or generated source.
4. Tokenize operands normally. The opcode itself must produce its assigned atomic
   token, without `[UNK]` or subword fallback.

Prefer this binding layer over editing the trained BPE merges or asking a tokenizer
API to append a new vocabulary entry: appending normally allocates a new ID rather
than binding an existing reserved ID. The base tokenizer artifacts remain immutable.

Bindings are adapter-specific. Two different Python versions may reuse a slot for
different instructions because each has its own bindings and embedding updates.
Within a version, assignments are one-to-one and stable. Pin old binding revisions
so an adapter can always reproduce its original tokenization.

The capacity is **16 instructions absent from the backbone vocabulary in any one
supported version**, not 16 new instructions per release or 16 additions over the
family's lifetime. Surviving instructions introduced in earlier releases still count
if the backbone never learned them. If a version needs more than 16, fail clearly:
a larger vocabulary/new backbone family requires an explicit revision, not silent
slot collisions, resizing, or fallback.

## 4. Train LoRA plus adapter-local embedding updates

A reserved ID has no learned positive opcode meaning. A LoRA might learn around a
fixed unused embedding, but it should not have to do so. For every newly bound slot,
train a small version-specific input-embedding correction alongside the LoRA:

```text
embedding(version, token_id) = frozen_base_embedding(token_id)
                            + version_delta[token_id]  # only bound reserved IDs
```

- Keep the entire backbone, original embedding table, and tokenizer frozen.
- Store an independent trainable tensor containing only the bound rows for that
  version. A zero correction starts from the base row; alternatively, initialize
  the correction so the effective embedding equals a related known instruction's
  embedding. Record the initialization, rather than claiming equivalence between
  these choices.
- Pass only LoRA parameters and the small correction tensor to the optimizer.
  Do **not** optimize the full embedding matrix with a gradient mask: weight decay
  or optimizer state can still modify supposedly frozen rows.
- Start from the matched training settings below. A separate embedding learning
  rate, different LoRA learning rate, rank, dropout, or early stopping is another
  declared experiment, not an undocumented default.
- These are **input-side** opcode updates. CodeT5 can share embedding weights across
  components; do not mutate a shared table or accidentally alter the output head.
  Specify the wrapper's encoder/decoder behavior and exclude opcode-only tokens
  from source-code generation. Segmentation updates must preserve instruction-label
  alignment.
- Switching adapters must switch the LoRA, opcode bindings, and embedding corrections
  together. Never apply corrections by overwriting the shared base weights.

Storage is tiny: 16 rows × 768 dimensions × 2 bytes is **24 KiB per task/version**
in FP16, at most; only bound rows need to be saved. Training optimizer state uses
additional memory. This is explicitly **LoRA + new-token embeddings**, not pure LoRA.

## 5. Match the normal training pipeline

The verified reference configuration is the saved `313_v3_statement.json` and
`313_v3_segmentation.json` under `/home/henryh/pylingual-2/dev_scripts/` on
`pylingual-train`, corroborated by the `dev` training code at revision `9914e52`.
This does not establish
that every historical per-version training run used identical settings. Copy and
hash the reference configuration into each new run's provenance.

| Setting | Statement | Segmentation MLM | Segmentation classification |
| --- | --- | --- | --- |
| Initialization | Generic `Salesforce/codet5-base` | Normal fresh RoBERTa configuration | The newly trained MLM |
| Tokenizer | Generic bytecode BPE | Generic bytecode WordPiece | Same frozen WordPiece |
| Epochs | 2 | 2 | 2 |
| Per-device batch | 24 | 28 | 28 |
| Gradient accumulation | 1 | 1 | 1 |
| Learning rate | `2e-5` | `5e-5` | `2e-5` |
| Precision | FP16 | FP32, as in normal MLM arguments | FP16 |
| Weight decay | `0.01` | `0`, normal Trainer default | `0.01` |
| Token limit | 256 for input and target | 1024 | 1024 |

Segmentation's normal windowing uses a **512-token budget and 128 stride** before
its configured token limit; preserve the actual implementation rather than
interpreting this as a 1,024-instruction window. Use the normal RoBERTa architecture,
including positional capacity, and run its MLM pretraining stage before boundary
classification. Do not warm-start from an existing PyLingual segmenter and call
that a generic-initialization ablation.

Use Hugging Face `Seq2SeqTrainer`/`Trainer`, the standard collators and Arrow-backed
datasets, rather than the previous custom training loop. Handle installed-library
API differences explicitly without changing optimization semantics. Record GPU
count, effective global batch size, package versions, seeds, save/evaluation
schedules, and any necessary memory-driven changes. Do not silently change batch
size, precision, sequence limits, or accumulation on an out-of-memory error.

For adapters, start with the same task-level optimization settings unless a
separately documented ablation changes them. Only parameter selection and
version-local conditioning should change. The reserved-token mechanism and pooled
vocabulary are intentional shared-model design differences, not claims that the
normal per-version tokenizers are identical to the shared one.

## 6. Retraining and future-version procedure

### Fresh universal family

1. Freeze the experiment definition, file partitions, normal reference settings,
   and opcode-registry rules. Verify 3.15 is absent from backbone/tokenizer inputs.
2. Build raw examples from the selected sources using normal conversion. Preserve
   every file's provenance and all produced rows; remove old one-row-per-file caps
   and stock-tokenizer length filters.
3. Train both generic tokenizers, including the 16-slot allocations; freeze and hash
   them. Rebuild tokenized datasets, invalidating old tokenizer-dependent caches.
4. Run a real smoke test using the production Trainer path and identical settings:
   encode, collate, forward, backward, optimize, save, reload, and infer. Check
   finite losses, vocabulary bounds, segmentation alignment, and a reserved-slot
   binding with a nonzero embedding gradient.
5. Train the universal statement backbone from generic CodeT5. Train segmentation's
   fresh MLM and boundary model. Save immutable backbone/tokenizer identities.
6. Train only the requested 3.14/3.15 adapter packages, freezing the backbones and
   training only LoRAs and any bound-slot correction rows.
7. Verify and benchmark before declaring the run complete.

### Add or retrain a version later

1. Load the exact frozen family; check its hashes and format support. Do not refit
   the tokenizer, remap known IDs, or load an adapter from another backbone.
2. Generate the new version's bytecode from its own training partition, preserving
   the benchmark exclusions. Determine unseen instructions and allocate at most
   16 slots. A version with no unseen instructions needs no embedding corrections.
3. Save the binding revision and initialize that version's LoRA/corrections. A
   continuation must load **both** the previous LoRA and corrections with the same
   binding revision; training from scratch is a different operation.
4. Train only those parameters. Use a batch containing bound instructions to verify
   that corrections receive gradients and actually update; hash the backbone
   before and after and require equality.
5. Save a new adapter revision, reload it in a fresh process, and test switching
   between old and new versions, including versions that reuse slot IDs.
6. Repeat held-out evaluation and attach the results to that adapter revision.

## 7. Artifact and runtime requirements

The following is a **proposed extension**, not a format currently accepted by the
loader. Introduce explicit format/version support before creating loadable families:

```text
family/
  family.json                 # task, backbone/tokenizer identity, registry, versions
  backbone/
  tokenizer/                  # immutable trained tokenizer
  opcode_registry.json        # known instruction IDs and exactly 16 reserved IDs
  adapters/<version>/
    adapter.json              # LoRA configuration and compatibility metadata
    adapter.pt                # LoRA tensors
    opcode_bindings.json      # instruction identity -> reserved ID, binding revision
    new_token_embeddings.pt   # only correction rows plus their explicit IDs
    training.json             # input manifests, settings, initialization, metrics
```

Bind an adapter package to the task, exact backbone and tokenizer hashes, opcode
registry, Python version, binding revision, embedding dimension, dtype, and row IDs.
Reject mismatches, unknown/out-of-range/duplicate IDs, wrong shapes, incomplete
packages, and unsupported formats. Never silently load only the LoRA and omit its
required embedding updates. Read tensor files with the existing safe tensor-loading
policy, such as `torch.load(..., weights_only=True)` for trusted local checkpoints.

A shared runtime must select all version-specific state under the same lock, covering
lazy inference and generation retries, and include the binding/adapter revision in
prediction-cache identities. Disabling adapters must also disable their embedding
corrections. For a pure frozen-backbone control, unseen opcodes still use the same
registered slot IDs but receive no version-trained corrections; record that mapping
as evaluation metadata and label the control accurately.

The current `pylingual/lora.py` format-1 checkpoints save LoRA matrices only. Neither
[`train_shared_adapter.py`](train_shared_adapter.py) nor the continuation scripts
currently train, save, or restore these correction rows or bindings. Extending their
checkpoint and runtime support is required before using this workflow.

## 8. Acceptance checks and honest comparisons

Before a full run, require:

- Exact file budgets, corpus ratios, version counts, and canonical/raw overlap checks.
- Immutable tokenizer IDs and byte-for-byte unchanged shared backbone after adapter
  training; only declared LoRA/correction parameters are trainable.
- Each newly bound opcode encodes atomically to its documented ID; operands, source
  strings, normal special tokens, instruction windows, and labels remain correct.
- Save/reload parity for LoRA **and** embedding updates; no cross-version effects
  when slots are reused. The 17th unseen instruction produces a clear error.
- Recorded real training throughput and GPU memory, separating data preparation,
  MLM pretraining, backbone training, adapter training, and evaluation times.

Compare the same held-out files with: ordinary per-version models; shared backbones
without adapters; shared backbones with pure LoRA; and shared backbones with **LoRA +
new-token embedding updates**. Report file-level decompilation equivalence, statement
metrics, boundary metrics, failures/timeouts, runtime, and deployed memory. A pure
LoRA control uses the same slot bindings but no trainable embedding corrections.

Ordinary per-version models should retain their ordinary per-version tokenizer
training for a faithful baseline. To isolate the benefit of sharing model weights
from the effect of the generic tokenizer, add a version-only model using the shared
tokenizer as a separately labeled control. Do not present deployed specialists as
matched-data/compute controls without establishing their data and budgets.


## 9. GPU4-local preparation (100k-file tokenizer-only stage)

The current entrypoint is `dev_scripts/run_gpu4_tokenizer_preparation.py`, run
**on GPU4**. NAS reads, normalization, version-specific compilation, conversion,
and tokenizer fitting stay on GPU4. Compilation uses 18 persistent Docker workers
(two per version); requests use host loopback and local stdin/stdout, **not an SSH
compiler tunnel**. SSH is only used to launch and monitor the job.

The pinned compiler images are `python:<version>-slim` for 3.6.15, 3.7.17, 3.8.20,
3.9.25, 3.10.20, 3.11.15, 3.12.13, 3.13.14, and 3.14.6. Docker access and cached
images are prerequisites; the runner uses `--pull=never` and pins the local image
ID. Containers have no network, a read-only repository mount/filesystem, dropped
capabilities and a 512 MiB memory limit. Sampled code is compiled, never executed.

```bash
# Run on GPU4 from the staged repository, under the background-job workflow:
cd /home/henryh/pylingual-tokenizer-100k-20261019/repo
.venv/bin/python -u -m dev_scripts.run_gpu4_tokenizer_preparation \
  --root /home/henryh/pylingual-tokenizer-100k-20261019
```

Use `--smoke` first for an isolated 90-file/2048-token fit, or `--preflight-only`
for compiler checks only. Do not run these concurrently with production or reuse
a completed experiment root: tokenizer families and their provenance are immutable.

The switch preserved 47,425 accepted files rather than resampling. Preflight
requires exact agreement with the previous Python patch versions, bytecode magic,
and opcode tables. `compiler-provenance.before-gpu4-docker.json` and
`compiler-transition.json` retain the old metadata and accepted-journal prefix
fingerprint; current provenance records image IDs and registry digests. A real
90-file test verified both tokenizer fits/reloads and converter semantic parity
(6,677 statement / 1,051 segmentation rows). Set display ordering can vary across
processes: five source cells and three segmentation cells differed only in set
ordering, so this is semantic parity, not byte-for-byte reproducibility.

### Legacy tunneled fallback

The older entrypoint is `dev_scripts/run_tokenizer_preparation.py`.
From the feature worktree on the coordinator host (existing native interpreters
3.6–3.14 are required; this entrypoint does not install software):

```bash
ssh-agent python3 -u -m dev_scripts.run_tokenizer_preparation \
  --repo /home/henry/Documents/src/pylingual/feat-shared-backbone-lora \
  --workspace /home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019 \
  --remote-root /home/henryh/pylingual-tokenizer-100k-20261019 \
  --training-host syssec-gpu4 \
  --nas-root /syssec-nas1/pyc-research/datasets \
  --old-reserved /home/henry/Documents/pylingual-experiments/shared-full-20261016/reserved-benchmark-sources.json
```

Launch this command using the tmux-jobs workflow, not a foreground session. Add
`--smoke` for a separate 90-file / 2048-token end-to-end test; its artifacts must
never be used as the production tokenizer family. The full run targets 90,000
PyPI plus 10,000 CSN unique accepted files. Version allocations differ by at
most one file. All converter rows are kept without old-tokenizer length filters.

`sample_nas_sources.py` reads source files directly from the NAS mount on GPU4.
Its cached catalog globally deduplicates original file identities and excludes
all NAS validation/test identities; 3.15 training directories are not candidates.
CSN uses numeric source IDs, so retain `nas_identity` (for example `csn-5241`) as
well as actual payload hashes. PyPI directory IDs preserve original SHA-512
identities even when the NAS file is an already-normalized copy. Exclusion metadata
includes previous CSN IDs so normalized copies cannot bypass adapter exclusions.

The local coordinator stores only exclusion/provenance metadata. Native compile
RPCs process normalized text transiently in RAM; no source corpus is staged on
local disk. Previous experiment sources, including adapter and held-out files,
are excluded by raw and canonical identity, with benchmark identities reserved.
The new corpus also excludes the deterministic canonical validation partition.

`prepare_tokenizer_corpus.py` produces the accepted-file journal, compressed
all-row chunks and an exact-quota manifest. `train_generic_tokenizers.py` fits
raw bytecode using the normal statement template / segmentation WordPiece
conventions, then adds atomic known-opcode markers and exactly16 non-special
reserved markers **within** the total30,000-token budget. Markers are input-side
only; `tokenizer_opcode_registry.py` binds instruction opcode fields without
changing operands. Segmentation uses offsets to preserve instruction alignment
when AddedTokens change the old word-ID layout. Opcode IDs must be excluded
from generated Python output. No neural model weights are trained here.

Actual completion requires both saved tokenizers plus a complete tokenizer
manifest, save/reload identity checks, atomic-slot checks and real-data alignment
checks. Script existence or a launched job is not proof of completion. The GPU4
dependency check and 31 registry/NAS/quota/compiler unit tests pass. The source-host SSH blocker
has been replaced by mounted-NAS preparation; no successful production fit is
claimed until the complete manifests exist. See the workspace `status.json` for
the latest observed state. Embedding-delta model training remains a subsequent
stage, not something implemented or validated by this job.


## 10. Verified 100k-file tokenizer family (2026-10-20)

The tokenizer-only run on `syssec-gpu4` is **complete and independently audited**.
No neural model weights were trained. The source corpus stayed on GPU4/NAS; only
fitted tokenizer artifacts and JSON provenance were copied back locally.

- Pool: **100,000 unique files**, exactly **90,000 PyPI + 10,000 CSN**.
- Versions: Python **3.6–3.14**, 10,000 PyPI files per version; 1,111 CSN files
  per version, with one extra CSN file for 3.6. Python 3.15 was excluded.
- Full converter output: **9,073,374 statement rows** and **1,408,216 segmentation
  rows**, without old-tokenizer length filters or per-file row caps.
- Independent full audit: 100,000 unique raw identities and canonical identities;
  zero overlaps with excluded raw/canonical/adapter identities or the 8,607
  indexed NAS validation/test identities; reserved canonical validation bucket
  excluded. Compiler, journal, data and tokenizer fingerprints verified remotely.
- Both task tokenizers preserve **233 atomic known opcodes + 16 contiguous future
  slots**. Save/reload, operand preservation, future bindings, instruction
  alignment and overflow rejection checks passed.

| Task | Learned base vocabulary | Total with opcode/slot markers | Reserved IDs |
| --- | ---: | ---: | --- |
| Statement BPE | 15,192 | **15,441** | 15,425–15,440 |
| Segmentation WordPiece | 29,751 | **30,000** | 29,984–29,999 |

### Why statement BPE is smaller than 30,000

The configured vocabulary size is a **ceiling, not a required exact size**.
An independent read-only repeat fit used every statement row from all 100,000
files (1,900,326,609 characters) and raised the base ceiling from 29,751 to 60,000.
It still produced **15,192 base tokens and 14,189 merges**, with **identical token
IDs, vocabulary, merge set, normalizer, pre-tokenizer and post-processor**.
The trainer exhausted the available merges under the unchanged repository
bytecode template; this was not a partial iterator or an overly small ceiling.
Default BPE minimum frequency is 0 and no maximum token length was imposed.
The diagnostic took 55.26 seconds and did not modify the saved tokenizers.

The original independent auditor incorrectly required exactly 30,000 tokens.
Its corrected check requires a positive base vocabulary within its budget,
`total = base + known opcodes + reserved slots`, and total within the family
budget. Two valid budget cases and three invalid accounting/overflow cases were
checked, then the **entire 100k-file audit passed**. Do not pad the fitted tokenizer
or retrain it merely to reach the ceiling. The family and IDs are now frozen.

### Measured checks, not a model benchmark

Both tokenizer fits plus verification took **176.35 seconds** (statement 59.04 s;
segmentation 117.30 s). On the existing 1,000-row-per-task diagnostic sample:
statement had **0 unknown tokens / 33,785 tokens**; segmentation had
**230 unknown tokens / 441,970 tokens** (about 0.052%). These are corpus-sample
checks, **not held-out tokenizer-quality or decompilation benchmarks**. The
vocabulary-convergence result does not establish model performance.

### Artifact locations and subsequent training

- GPU4 family:
  `/home/henryh/pylingual-tokenizer-100k-20261019/tokenizers/{statement,segmentation}`
- Verified local copy:
  `/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/verified-artifacts/tokenizers/{statement,segmentation}`
- Audit/evidence files in the local `verified-artifacts` directory:
  `tokenizer-audit.json`, `statement-vocab-diagnostic.json`, `data-manifest.json`,
  and compiler provenance/transition JSON. Every copied task file and the raw
  data/compiler manifest hashes were verified after transfer.
- Machine-readable result:
  `/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/results.json`.

Future model training must load this immutable family, not refit it. Select the
**task-specific actual vocabulary size** when checking/resizing embeddings;
reserve/update only the documented per-task slot IDs for each version adapter.
The LoRA-plus-new-token-embedding design in earlier sections remains the plan;
this tokenizer-only stage does **not** implement or benchmark neural adapters.
Full neural backbone, adapter, control training and decompilation benchmarking
remain paused until separately authorized; the bounded smoke below is an exception.


## 11. Bounded GPU smoke (2026-10-20)

Entry point: `dev_scripts/smoke_shared_gpu.py`. This is a diagnostic harness, not
an unrestricted trainer. On `syssec-gpu4`, it completed the proposed **3+1** layout:

| Task | GPUs | Precision | Batch/device | Trainer steps | Peak allocated GiB |
| --- | --- | --- | --- | --- | --- |
| Generic CodeT5 statement training | 0, 1, 2 (DDP) | FP16 | 24 | 4 | 12.616 (rank 0) |
| Fresh RoBERTa masked-language pretraining | 3 | FP32 | 28 | 4 | 11.961 |
| MLM-initialized B/I/E classifier | 3, after MLM | FP16 | 28 | 4 | 5.851 |

These are short fixture measurements, not full-training throughput or worst-case
memory estimates. Statement inputs and targets are capped at 256 tokens in this
smoke. Segmentation uses the normal instruction-boundary sliding-window mapping.
Rows come from the already prepared backbone pool, not adapter or held-out data.

Verified:

- All three models complete forward/backward, save, and exact-logit reload checks
  under matched FP32 evaluation. MLM encoder weights transfer exactly to the
  boundary classifier before classifier training. Statement generation runs with
  opcode/reserved IDs forbidden in generated source.
- Both tasks train rank-8 LoRAs plus encoder-input-only embedding corrections using
  two **synthetic** version bindings. Each binding runs two optimizer steps;
  nonzero gradients reach both LoRAs and embedding deltas.
- Saving/reloading corrections and LoRAs reproduces logits exactly. Two bindings
  can reuse one reserved slot without changing the other binding's output.
  SHA-256 fingerprints prove original backbone parameters remain unchanged
  during these adapter tests, including tied input/output embedding weights.
- Tokenizer files still match every immutable hash in the family manifest.
- All jobs exit and all four GPUs return to idle. No long training was started.

**Historical smoke warning (accepted as non-blocking):** FP16 statement startup logged infinite gradient
norms on steps 1 and 3; boundary training logged one on step 1. Later norms and
losses were finite, checkpoint checks passed, and parameters remained usable.
This is not evidence of long-run numerical stability. The user subsequently
accepted this transient warning as non-blocking; do not change precision or gate
training solely on it. Persistent invalid losses/parameters still fail. The harness counts actual AdamW step calls
via an optimizer post-hook, not callback counters that can misreport skipped
steps. A step call alone does not prove a useful nonzero gradient update.

An earlier reload comparison incorrectly mixed Accelerate's retained FP16 forward
wrapper with an unwrapped FP32 model. The harness removes that wrapper before
strict equality checks; it does not loosen numerical tolerances to hide a mismatch.

Successful artifact directory on GPU4:

`/home/henryh/pylingual-tokenizer-100k-20261019/gpu-smoke-20261020-v3/`

Local compact evidence (metadata only):

`/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/gpu-smoke-20261020/results-v3.json`

`/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/gpu-smoke-20261020/verification-notes.json`

The saved synthetic corrections are explicitly marked **harness-only**. They do
not establish support for real 3.15 bindings, satisfy the 54,500-file version
budgets, or replace the production adapter bundle/loader work in sections 5–7.
Do not load them as production adapters. Full model training and benchmark results
remain future, separately authorized work.


## 12. Implemented production adapter workflow (2026-10-20)

### Components and safety properties

- `pylingual/opcodes.py`: validates the immutable opcode registry, token IDs,
  16-slot capacity and version-local bindings; rewrites only instruction opcode
  fields. Operands, jump targets and future-annotation suffixes stay intact.
- `pylingual/embedding_updates.py`: sparse, adapter-local **encoder-input**
  corrections. CodeT5's shared embedding table, decoder input embeddings and output
  head are not overwritten. An adapter with no new opcodes has no correction rows.
- `pylingual/lora.py`: switches LoRAs, bindings and correction rows under one shared
  lock; version-local slot reuse is isolated. Format-2 adapter packages pin the
  backbone content ID, tokenizer/registry fingerprints, correction IDs and weight
  checksum. Invalid packages are rejected before changing the shared model.
- `pylingual/opcode_segmenter.py`, `pylingual/shared_models.py`,
  `pylingual/models.py`, and `pylingual/decompiler.py`: use atomic opcode inputs
  through normal model loading, segmentation windows and statement translation.
  Offset-based alignment avoids depending on AddedTokens' `word_ids()` behavior.
  Opcode-only IDs are forbidden in generated source. Backbone-only ablations retain
  the same input preprocessing, with LoRAs and correction rows disabled together.
- `dev_scripts/opcode_adapter_training.py`, dispatched by
  `dev_scripts/train_shared_adapter.py`: normal Hugging Face Trainer/Seq2SeqTrainer,
  Arrow-backed datasets and standard task collators. Statement input/target limits
  are **256/256**; segmentation windows are **512 tokens, stride 128**. It optimizes
  only the selected LoRA plus bound correction rows, checks finite losses/weights,
  and hashes frozen weights before/after. The entire base is audited even if a
  parameter were accidentally marked trainable.
- Checkpoints contain adapter/correction weights, Trainer optimizer/scheduler/RNG
  state and run provenance, not a duplicate trainable backbone. Resume requires
  identical data hashes, version, bindings, tokenizers, encoding limits, LoRA
  configuration and resolved optimizer settings.

### A. Prepare and train the shared backbones first

Keep the already fitted statement and segmentation tokenizers unchanged. Apply
`OpcodeRegistry.bind()` when preparing backbone bytecode inputs; for labeled rows,
`dev_scripts.opcode_adapter_training.encode_rows()` implements the task encoding.
The backbone pool remains the balanced 100,000-file 90% PyPI/10% CSN pool from
3.6–3.14. Never infer a new vocabulary from adapter/validation/test rows.

Train from generic CodeT5 and the normal fresh RoBERTa MLM initialization using
section 5 settings. Transfer that MLM to the B/I/E classifier before classifier
training. There is no separate inference-time MLM LoRA: the version adapters attach
to the statement model and the resulting boundary classifier.

The old `dev_scripts/shared_training.py` warm-start/custom-loop format-1 pilot is
**not** the corrected universal/MLM trainer. Do not accidentally reuse its defaults
or old checkpoints. Full-backbone orchestration and quality benchmarking are still
separate work; the publisher below packages a supplied checkpoint and does not
train it or certify that its claimed data lineage is correct.

After the actual backbones are trained, package each with its exact frozen tokenizer.
Template commands (checkpoint paths must refer to the new full runs, not smoke runs):

```bash
python -m dev_scripts.publish_shared_backbone \
  --task statement --checkpoint "$STATEMENT_CHECKPOINT" \
  --tokenizer "$EXPERIMENT_ROOT/tokenizers/statement" \
  --training-versions 3.6 3.7 3.8 3.9 3.10 3.11 3.12 3.13 3.14 \
  --held-out-version 3.15 --output "$MODEL_ROOT/statement-base"

python -m dev_scripts.publish_shared_backbone \
  --task segmentation --checkpoint "$SEGMENTATION_CHECKPOINT" \
  --tokenizer "$EXPERIMENT_ROOT/tokenizers/segmentation" \
  --training-versions 3.6 3.7 3.8 3.9 3.10 3.11 3.12 3.13 3.14 \
  --held-out-version 3.15 --output "$MODEL_ROOT/segmentation-base"
```

The publisher rejects vocabulary-size mismatches and incomplete/unwrapped model
loads. It never resizes a model. It records hashes of the saved backbone files and
uses their content hash as the backbone ID; normal loading verifies those hashes.
All outputs must be fresh directories; source families are not modified. For
explicitly bounded diagnostics only, pass `--smoke-only` and keep that designation.

### B. Build version-specific adaptation rows

Use **54,500 training source files per version**, for 3.14 and 3.15. Validation/test
files are additional and disjoint; retain the section 1 canonical/source-hash
exclusions from the backbone pool and across held-out partitions. Rebuild examples
using normal bytecode conversion and the new tokenizers, not old tokenized tensors.

The adapter input is JSONL, one normal converted example per row, with:

- `version`: `"3.14"` or `"3.15"`;
- `file`: stable canonical/source-hash identity used for partition checks;
- `bytecode`: the raw normal model-view string with ` <SEP> ` between instructions;
- `source`: the statement target (as emitted by the normal masked converter);
- `boundary`: segmentation-only B/I/E labels, either a list or its normal literal
  representation, with exactly one label per bytecode instruction.

The trainer performs binding/tokenization itself. Do not pre-replace source targets
or globally replace opcode-like words in operands. Do not truncate the training
file budget into a fixed number of randomly selected statement rows.

Bindings are inferred from **training rows only**, sorted by unseen mnemonic.
For explicit/reproducible assignment, use `--bindings bindings.json`, containing
`{"NEW_INSTRUCTION": 0}` (a slot index, not a vocabulary ID). Bindings may also be
specified from the target interpreter's instruction specification; never mine
held-out examples for them. Existing bindings cannot be changed on continuation.
If a validation/runtime instruction is unbound, fail clearly instead of silently
falling back to subwords. More than 16 distinct unseen instructions requires an
explicit new family/vocabulary revision.

### C. Train adapters independently, then assemble once

Run long jobs in tracked tmux sessions. Once the two backbones exist, four independent
jobs can occupy the four GPUs: statement/3.14, statement/3.15, segmentation/3.14,
and segmentation/3.15. Each job starts from its task's same immutable base family,
not from a merged adapter or another version's continued weights.

Example commands for one version (repeat with `VERSION=3.14`; assign a distinct GPU
to each concurrent job):

```bash
VERSION=3.15
CUDA_VISIBLE_DEVICES=0 python -m dev_scripts.train_shared_adapter \
  --family "$MODEL_ROOT/statement-base" --version "$VERSION" \
  --data "$ADAPTER_DATA/$VERSION/statement-train.jsonl" \
  --validation-data "$ADAPTER_DATA/$VERSION/statement-valid.jsonl" \
  --epochs 2 --batch-size 24 --accumulation 1 --learning-rate 2e-5 \
  --precision fp16 --rank 8 --checkpoint-steps 10000 \
  --output "$MODEL_ROOT/statement-$VERSION"

CUDA_VISIBLE_DEVICES=1 python -m dev_scripts.train_shared_adapter \
  --family "$MODEL_ROOT/segmentation-base" --version "$VERSION" \
  --data "$ADAPTER_DATA/$VERSION/segmentation-train.jsonl" \
  --validation-data "$ADAPTER_DATA/$VERSION/segmentation-valid.jsonl" \
  --epochs 2 --batch-size 28 --accumulation 1 --learning-rate 2e-5 \
  --precision fp16 --rank 8 --checkpoint-steps 10000 \
  --output "$MODEL_ROOT/segmentation-$VERSION"
```

Rank 8, alpha 16, dropout 0.05, zero-initialized input corrections, and a shared
LoRA/correction learning rate are explicit adaptation choices, not historical
PyLingual baseline settings. Weight decay is 0.01. Preserve matched initializations,
file budgets, precision and reference settings in version-only comparison models.

For interruption recovery, repeat the same command with a **fresh** output directory
and `--resume previous-output/training/checkpoint-N`. Use the same data, tokenizer,
bindings and optimizer arguments. The checkpoint restores LoRAs, correction rows,
optimizer, scheduler and RNG/data progress. `--steps N` is for bounded engineering
checks, not a substitute for the requested full two-epoch run.

Assemble the two independently trained adapters for each task:

```bash
python -m dev_scripts.assemble_shared_adapters \
  --family "$MODEL_ROOT/statement-base" \
  --adapter "$MODEL_ROOT/statement-3.14/adapters/3.14" \
  --adapter "$MODEL_ROOT/statement-3.15/adapters/3.15" \
  --output "$MODEL_ROOT/statement-family"

python -m dev_scripts.assemble_shared_adapters \
  --family "$MODEL_ROOT/segmentation-base" \
  --adapter "$MODEL_ROOT/segmentation-3.14/adapters/3.14" \
  --adapter "$MODEL_ROOT/segmentation-3.15/adapters/3.15" \
  --output "$MODEL_ROOT/segmentation-family"
```

Point the normal `--config-file` at a YAML/JSON config containing `SHARED_BACKBONES`
with `STATEMENT_MODEL` and `SEGMENTATION_MODEL` paths. Paths are relative to that
config file unless absolute. `USE_ADAPTERS: false` selects the universal-only
comparison; `USE_STATEMENT_ADAPTERS` and `USE_SEGMENTATION_ADAPTERS` can override
that independently. Each task's backbone is loaded once per worker/device and reused
across version switches. No personal dotfiles are needed or modified; diagnostic
configs are transient experiment artifacts, deliberately outside chezmoi.

### D. Acceptance evidence and limits

Actual GPU4 checks used the production publisher, Trainer, adapter package loader,
assembler and public `load_models()` path, not the earlier harness-only corrections:

- Statement and segmentation each completed four FP16 optimizer updates for each
  version-tagged fixture. Bound correction tensors became nonzero.
- Resuming each task from checkpoint 2 reproduced the uninterrupted checkpoint's
  final adapter tensors with **maximum absolute difference 0.0**.
- Normal version switching reused one backbone per task. The no-adapter comparison
  retained atomic inputs/alignment and disabled both LoRA and embedding updates.
- Locally generated synthetic `.pyc` fixtures compiled by native **3.14.6** and
  **3.15.0b2** passed real Decompiler masking, segmentation and translation stages
  on GPU4. This does not claim successful control-flow reconstruction, source
  equivalence or decompilation accuracy from a four-step model.
- 21 new integration/opcode tests and 17 legacy shared-model tests passed. Checks
  cover fixed IDs, strict bindings/capacity, target prefixes, sparse-input gradients,
  frozen/tied weights, version isolation, alignment, token limits, package reload,
  backbone identity and checksum rejection.

The new-token **training** fixture deliberately uses `SYNTHETIC_NEW_OP`; it is not a
real 3.15 dataset or a finished 54,500-file adapter. Native 3.15.0b2's ordinary
opcode names are already covered by this registry's 233-name union. Its remaining
table entries absent from the registry are `TRACE_RECORD` (internal/instrumented)
and seven compiler pseudo-operations (IDs >=256), not new ordinary `.pyc` opcodes.
Therefore actual 3.15 data may need a LoRA but **zero new embedding rows**. Do not
invent bindings or require nonempty corrections just to exercise the mechanism;
recheck against the exact interpreter/data used by the eventual full run.

GPU4 artifacts: `/home/henryh/pylingual-tokenizer-100k-20261019/embedding-integration-v5/`.
Local metadata-only evidence:
`/home/henry/Documents/pylingual-experiments/shared-tokenizer-100k-20261019/production-integration-20261020/`.
`dev_scripts/verify_opcode_adapters.py` and `dev_scripts/verify_shared_loader.py`
reproduce bounded diagnostics against prepared fixtures. All diagnostic families
remain marked `smoke_only`; none is a usable performance checkpoint.


## 13. Autonomous full retraining (2026-10-20)

**Status: running, not finished.** This supersedes earlier “not started” readiness
notes, not the historical smoke measurements. No smoke loss is a quality benchmark.

Run root on `syssec-gpu4`:
`/home/henryh/pylingual-tokenizer-100k-20261019/full-retrain-20261020/`.
The immutable `protocol.json` records the run's intended budgets and settings;
individual `provenance.json`, encoding manifests, and result files record actuals.

- Backbones: all **100,000** selected files, 90,000 PyPI / 10,000 CSN, balanced
  across 3.6–3.14, excluding 3.15. Include every normal converter row, not just the
  first statement/code object per file: **9,073,374 statements** and **1,408,216
  code-object rows**, yielding **3,033,955 segmentation windows**.
- Statement: generic CodeT5, two epochs, three-GPU DDP, 24 examples/rank,
  effective batch 72, LR 2e-5, FP16. The other GPU trains fresh RoBERTa MLM
  (two epochs, batch 28, LR 5e-5, FP32), then the B/I/E classifier (two epochs,
  batch 28, LR 2e-5, FP16) initialized from that new MLM.
- Each target version, 3.14 and 3.15, has **54,500 training files**
  (49,050 PyPI / 5,450 CSN), plus a separate 500-file validation selection.
  Reuse the verified earlier 3.14 selections; expand 3.15 using only sampled NAS
  files read on GPU4. No source-corpus staging occurs on the local workstation.
- Version LoRAs use rank 8, alpha 16, dropout .05, two epochs, LR 2e-5, FP16;
  statement batch 24 / accumulation 3, segmentation batch 28 / accumulation 1.
  Allocate reserved-slot bindings only for instruction names actually observed
  in the selected training input. Unused internal/pseudo operations do not force
  new embeddings. The real 3.15 run can legitimately require zero new rows.
- Ordinary controls train **version-specific** BPE/WordPiece tokenizers with the
  normal 30,000 vocabulary budget and no opcode placeholders, then generic CodeT5
  and fresh MLM→classifier on exactly the same version training files. Use the
  same effective batch, epochs, seed and ordinary task optimizer settings.

### Entry points and restart safety

- `prepare_full_training_data.py`: full-coverage Arrow encoding;
  `--scope universal|adapter|control`. Adapter mode accepts `--bindings-file`.
  Scope, source-journal hash, tokenizers, limits, source counts and all converter
  rows are tracked. HF's in-memory fingerprint can change on `load_from_disk`;
  compare the persisted `dataset/state.json` identity, not the restored view.
- `prepare_version_training_corpora.py`: validates retained source-transfer hashes,
  recompiles 3.14/3.15, and samples only the additional 3.15 files from the NAS.
  Retained files are already normalized: **do not normalize them a second time**.
  The normalizer is not idempotent. Preserve their pinned normalized bytes and
  canonical identities while rebuilding all normal bytecode/converter rows.
- `train_control_tokenizers.py`: normal per-version tokenizers fitted only on that
  version's complete training selection, not validation or the shared backbone pool.
- `prepare_training_variants.py`: waits for selected sources, infers genuine new
  opcode bindings, fits control tokenizers, and builds adapter/control train and
  validation Arrow caches. At most two four-process encoding jobs run concurrently.
- `train_universal_models.py`: two-epoch backbone or `--ordinary-version` control
  training. Non-finite loss/parameters are fatal; transient FP16 startup warnings
  are not. Explicit `--resume` retains checkpoint optimizer/scheduler state.
- `train_shared_adapter.py` additionally supports `--encoded-data` and
  `--encoded-validation`. In this mode, `--data` / `--validation-data` identify the
  matching **accepted source journals**, not expanded raw-row JSONL. Caches remain
  memory-mapped rather than holding millions of Python dictionaries in RAM.
  Registry, semantic tokenizer identity, bindings, version, source budget,
  partition, converter coverage, token limits and persisted Arrow state are checked.
- `run_full_gpu_lane.py`: starts/resumes the initial 3+1 lanes once data is ready.
- `run_full_training_study.py`: schedules independent version adapters and ordinary
  controls as GPUs become free; honors MLM→classifier dependencies, preserves
  failed attempts, resumes adapter-only checkpoints into fresh family directories,
  and assembles both versions against each task's one immutable backbone.
  It waits rather than taking a GPU occupied by another user/process.

These current orchestration scripts use the dated run subdirectory above. For a
new experiment, use a fresh root with the required source/tokenizer artifacts (or
make an explicit new run identifier); do not overwrite a previous run's family.
Model-family `complete` metadata is published only after successful training and
assembly. `training-complete-awaiting-benchmark` is **not** study completion.

### Acceptance evidence before the long adapter/control stages

The ordinary tokenizer→encoding→MLM→classifier→statement path completed bounded,
isolated real GPU checks. The memory-mapped adapter path also completed production
CLI training and validation for both tasks, with frozen-backbone checks and
adapter-only saves. These fixtures are explicitly `smoke_only`; they are not used
as initializations or performance results for the full study.

Monitor `statement-lane.json`, `segmentation-lane.json`,
`universal/*/progress.json`, `target-corpora/progress.json`,
`variant-preparation.json` and `training-supervisor.json`. Check actual optimizer
updates and full-epoch result files before launching held-out quality/VRAM tests.


### Full benchmark automation and interpretation

`run_full_retrain_benchmarks.py --run <dated-run-root>` waits for all universal,
adapter and ordinary-control training to finish. It then evaluates **every listed
file** in the independent benchmark, not a smoke subset: currently 1,915 Python
3.14 files and 2,078 Python 3.15 files. It uses otherwise-idle GPUs only, with
resumable per-file results and up to three attempts for failed benchmark jobs.

The four arms are ordinary per-version controls, shared backbones without adapters,
shared statement adapters only, and both shared task adapters. In the three shared
arms the backbone/tokenizer pair is identical; only adapter enablement changes.
Unlike an older pilot, the statement-only arm does **not** substitute an ordinary
segmenter, which would confound the ablation.

`benchmark_full_retrain.py` pins input listings, every model/runtime artifact hash,
training-source journals, native compiler provenance and generated runtime settings.
It checks full source budgets and normalized train/validation/backbone separation,
and rejects benchmark raw-source SHA-512 overlaps. The benchmark distribution
contains bytecode rather than original source, so benchmark canonical-AST overlap
is not rederived; the report states that limit explicitly.

- **Primary quality:** complete-file recompilation/bytecode equivalence through
  the actual decompiler, 30-second inference limit, all failures/timeouts retained.
- **Diagnostics:** teacher-forced metrics on every encoded validation row from
  the separate 500-file version validation sets. Tokenization/windowing differs
  between ordinary and shared models; token accuracy/loss is not decompilation
  accuracy and is not directly comparable across vocabularies.
- **Memory:** both Python versions kept resident in one fresh process per arm;
  report live CUDA allocation and deduplicated storage bytes, plus object-identity
  checks proving that shared arms actually share their two backbones.
- **Throughput:** model kernels bypass translator output caching, FP32 inference,
  eight-example batches, two warmups and ten timed repeats; input shapes and output
  token rates are retained. This is not a claim that tokenizer-specific workloads
  contain identical numbers of tokens or generated tokens.

Outputs are under `<dated-run-root>/benchmark/`; `report.json` is published only
when all quality and performance jobs succeed. `benchmark-supervisor.json` records
waiting/running/failed/completed state. No final accuracy exists while training is
still running. Native verification uses the recorded 3.14.6 and 3.15.0b2 compilers
via a separate RAM-only loopback/SSH gateway; the sampled corpus is never staged
on the local workstation. The ordinary loader also accepts a local tokenizer
export, so controls do not require uploading weights or tokenizers to the Hub.

The real preflight verified both native compiler RPC routes and the ordinary local
checkpoint+tokenizer loader through actual CPU statement generation/segmentation.
The 38-test adapter/data/benchmark/annotation suite passed. These checks are infrastructure
acceptance only, not full-model quality measurements. Current measured optimizer
rates imply a multi-day complete study; preserve checkpoints and monitor progress
rather than reducing epochs/data to force an overnight result.


#### Recovery: xdis common-constant annotation decoding

The full 3.15 source preparation exposed an xdis representation difference:
`LOAD_COMMON_CONSTANT.argval` can be the actual `NotImplementedError` class,
where the annotation recognizer previously expected integer table index `1`.
Annotation preamble matching now uses that instruction's encoded `arg` index;
other instruction values and the bytecode itself are unchanged. This supports
both old and new xdis representations without weakening the rest of the pattern.

The original failed file passed the real bytecode-to-training-rows conversion
(148 statement rows, 20 segmentation rows) and was accepted on resume. No retained
file was dropped or replaced, and backbone training was not interrupted. The
complete adapter/data/benchmark/annotation regression suite passed 38 tests.
Evidence and the resumed job handle are in `<dated-run-root>/annotation-repair.json`.
To resume preparation after a recoverable failure, rerun
`python -u -m dev_scripts.prepare_version_training_corpora --root <experiment-root> --workers 8`;
the accepted-file journal preserves already verified selections. Diagnose retained
conversion failures rather than silently skipping or replacing selected sources.


## Current production allocation: 2+2 (2026-10-20)

This supersedes the earlier 3+1 launch instructions for the active full retrain.
GPU 0–1 train the statement backbone with **36 examples/rank, global batch 72**.
GPU 2–3 train MLM and then segmentation with **14 examples/rank, global batch 28**.
Epoch count, learning rates, precision, tokenizers, datasets, and initial weights
are unchanged. `run_full_gpu_lane` now defaults to this allocation.

```bash
python -m dev_scripts.run_full_gpu_lane --root "$ROOT" --lane statement   --prepare-pid "$STATEMENT_PREP_PID" --gpus 0,1
python -m dev_scripts.run_full_gpu_lane --root "$ROOT" --lane segmentation   --prepare-pid "$SEGMENTATION_PREP_PID" --gpus 2,3 --original-world-size 1
```

The preparation PID is only consulted while the Arrow manifest is absent.
For existing runs, lane wrappers preserve original provenance and use explicit
`--resume-original-world-size` authorization when repartitioning batches.
Model, optimizer, scheduler, global step, and (where applicable) scaler are
restored by Trainer. An actual world-size transition deterministically reseeds
per-rank RNGs (`seed + rank`); it is not a bit-identical continuation. Subsequent
same-topology checkpoints restore their RNG normally. Execution details are
appended to `resume-history.jsonl`, with live `latest-execution.json` and
`resume-verification.json` recording observed restoration rather than assuming it.

### Important Transformers 5.20.2 loss-normalization correction

The installed T5 and RoBERTa forwards accept `**kwargs` but compute native **mean**
cross entropy without using `num_items_in_batch`. Trainer infers support and
multiplies that native mean by DDP world size. A real CPU/Gloo resume smoke showed
MLM loss and gradient norms doubling from one rank to two, before this correction.
The same model/Trainer inspection confirmed that the old three-rank statement
run was affected. This is separate from the waived transient FP16 warning.

`dev_scripts/loss_normalization.py:token_normalized_loss` now supplies summed
cross entropy divided by Trainer's global supervised-token count. Trainer's
DDP compensation is therefore correct. Single-, two-, and three-rank gradient
equivalence tests include uneven ignored-label counts; actual T5 3→2 and MLM 1→2
checkpoint smoke tests also pass. The correction applies to universal training
and version-only controls through their common trainer entrypoint.

The invalid early statement run was quarantined (not deleted) under
`full-retrain-20261020/universal/statement-before-ddp-loss-fix-20261020` and
**restarted from the same generic CodeT5 initialization and same data**.
It must not contribute to reported results. The valid single-GPU MLM run resumed
from checkpoint **10,000**, retaining its optimizer and scheduler at step 10,000.
Legacy distributed checkpoints with incompatible loss normalization are rejected
rather than silently resumed. No training source files were resampled for this
allocation change.

Migration and acceptance records: `repartition-state.json`,
`repartition-verification.json`, and updated `protocol.json` in the run directory.
The dependency supervisor was paused only during migration and is running again.
51 regression tests passed; both live tasks have subsequently performed hundreds
of optimizer updates on two ranks, with all four physical GPUs active.


### Completed target pools and reserved-slot coverage (2026-10-20)

The independently checked target pools contain **54,500 training files plus
500 validation files per version**, each split exactly 90% PyPI/10% CSN. All
110,000 canonical and source identities are unique and disjoint from the
100,000-file backbone/tokenizer pool. Audits are saved as
`target-source-independent-audit.json` and `compiler-vocabulary-independent-audit.json`.

The shared registry has **233 opcode names**, exactly the union of compiler
metadata from Python 3.6–3.14; no 3.15 compiler vocabulary was included. Both
current target pools use only already-known names, so **zero of the 16 reserved
slots are bound for either adapter**. The actual 3.15 compiler build is
**CPython 3.15.0b2** (3.14 is 3.14.6); final scores must name these builds.

Consequently this real-data study tests version adaptation/decompilation, but
does **not** measure learning entirely new instruction-token embeddings. The
reserved-slot mechanism is tested synthetically; it must not be presented as
empirically exercised by these target pools. Keep the agreed sampling and
backbone vocabulary unchanged rather than manufacture a new-token result.


## Active-run amendment — 2026-10-21: shared family only, official v1 + v2

This section supersedes earlier plans for freshly trained ordinary controls.
The user cancelled **all six version-only full-model training stages**, before
any of those stages started (all have zero attempts). Do not queue them unless
the user explicitly authorizes a new comparison-training run. Their prepared
input datasets are retained as historical artifacts, not training permission.

The current autonomous chain is:

1. Finish the existing approved 100,000-file universal statement and MLM →
   segmentation runs. Keep the fixed shared tokenizers and the 2+2 GPU layout.
2. Automatically train only four LoRAs: statement and segmentation for each of
   3.14 and 3.15, on the existing 54,500-file adaptation partitions.
3. Automatically benchmark the shared family using both official releases of
   https://github.com/syssec-utd/pylingual_eval_dataset, **only Python 3.14 and
   3.15**. The user will compare these measurements to existing deployed-model
   results; do not claim newly trained matched controls or matched-budget wins.

Release pins and complete requested manifest sizes:

| Dataset | Tag | Commit | 3.14 files | 3.15 files |
|---|---|---|---:|---:|
| v1 | `v1.1.0` | `a60d3b95748283ba0fbcd5e94da307d5bd5df870` | 4,991 | 4,991 |
| v2 | `v2.0.1` | `258bc0380f7c8550fa28f294b896cf168a5e4055` | 1,915 | 2,078 |

The source-identity audit found known training/validation overlap: 326 files
per version in v1; 23 files for 3.14 and 25 for 3.15 in v2. Keep every official
listed file for comparability, and report known-overlap and non-overlap strata
separately. **Do not describe the entire official evaluation as held-out.**
Canonical source equivalence is not rederived from bytecode, so the non-overlap
check is a raw-source SHA-512 identity check, not a proof against all semantic
duplicates. Each selected bytecode file and exact release commit is pinned.

The automatic benchmark evaluates **only the final system with both adapters**;
no extra component-ablation arms are automatically queued. It introduces no
additional model training. The queue contains four release/version quality jobs
and one inference runtime/VRAM job. The file timeout is the upstream 300 seconds. Reports are written beneath
`benchmark/v1/`, `benchmark/v2/`, and `benchmark/performance/`; final aggregation
is `benchmark/report.json`.

`run_full_training_study.py` defaults to adapter-only scheduling and archives
cancelled control entries separately. `run_full_retrain_benchmarks.py` requires
only the four completed adapters, not cancelled controls. The actual 2+2 GPU
reservation is tested so adapters cannot collide with an unfinished backbone.
The benchmark compiler route is pinned separately from compiler-build metadata:
GPU4 loopback port 45616 forwards to the existing native gateway port 46017.
The actual verification compilers remain 3.14.6 and 3.15.0b2 and their magic bytes
match the requested evaluation releases. The live two-version compiler preflight
and full requested-file audit passed; 11 focused queue/scoring regression tests
passed without installing any extra software.

Current durable runtime records (outside the Git worktree):
`shared-only-scope-amendment.json`, `shared-only-automation.jobs.json`,
`official-evaluation-release-audit.json`, `benchmark-compiler-route.json`,
`training-supervisor.json`, and `benchmark-supervisor.json` under the active
`full-retrain-20261020` workspace/run. These document a changed plan, **not
completed LoRA training or completed benchmark measurements**.
