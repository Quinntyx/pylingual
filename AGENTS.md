# Agent notes: shared-backbone + LoRA branch

## Persist training artifacts to Hugging Face

Training hosts die without warning. syssec-gpu4 went down on 2026-10-23 and took the only copy of
every trained backbone and adapter with it. Nothing that took more than an hour to produce may
exist only on one machine.

- **Push every dataset before training on it.** That covers the source sample, the converted
  rows, and the tokenized splits. Push them from the machine that built them.
- **Push checkpoints while training runs, not only at the end.** Use `hub_strategy="checkpoint"`
  (or an equivalent periodic upload) so that the latest resumable checkpoint, including optimizer,
  scheduler and RNG state, lives on the Hub. When training finishes, push the final model to the
  same repo.
- **Repos must be public.** The `syssec-utd` org is out of private storage. Never upload
  credentials, `.env` files, host lists, or anything that is not a dataset or a model.
- **Keep the namespace tidy.** `syssec-utd` already holds hundreds of repos:
  - Use one repo per artifact per version, named `syssec-utd/pylingual-shared-<artifact>-v<N>`.
    For example: `pylingual-shared-tokenizer-corpus-v1`, `pylingual-shared-tokenizer-v1`,
    `pylingual-shared-statement-v1`, `pylingual-shared-segmentation-v1`,
    `pylingual-shared-lora-statement-3.14-v1`.
  - Bump `<N>` only when the contents change meaning: new data, new tokenizer, or new
    hyperparameters. Resumed or continued runs push to the existing repo; they do not get a
    new one.
  - Do not create repos for smoke tests or throwaway runs. Keep those local.
  - Every repo's README records what it was built from: the upstream repo IDs and revisions, the
    git commit of this branch, the seed, and the hyperparameters.
- Record the Hub repo IDs and revisions of every artifact in the experiment's manifest JSON, so
  that a run can be rebuilt from the Hub alone.

## Ask before spending compute

- Never substitute or re-select the dataset the user specified for a training stage. Approval for
  one stage's data does not carry over to another stage.
- Do not queue training jobs the user has not approved, such as controls or comparison models.
  Before any multi-hour run, state the exact data, hyperparameters and GPU layout, and get a yes.

## Hosts (as of 2026-10-28)

- `syssec-gpu4`: dead for the foreseeable future.
- `triplesec1-gpu`, `triplesec2-gpu`: one RTX 4090 (24 GB) each, 8 cores, 30 GB RAM, about
  105 GB free disk. Neither has the NAS mounted.
- The user's local machine mounts `/syssec-nas1`. Run CPU-bound work there, such as data
  preparation and tokenizer training.
