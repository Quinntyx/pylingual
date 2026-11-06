"""Publish training inputs to public Hugging Face dataset repos before any GPU uses them (see AGENTS.md).

`adapter-corpus` uploads the per-version adapter pools from build_adapter_pools as Parquet (same
row schema as the tokenizer corpus plus a split column), with every journal and manifest the
trainers verify. `encoded` uploads the audited Arrow datasets from prepare_full_training_data as
saved, so a training host can load_from_disk them after a plain snapshot download.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from dev_scripts.hub_checkpoints import git_commit
from dev_scripts.publish_tokenizer_corpus import SCHEMA, record

SPLIT_SCHEMA = SCHEMA.append(pa.field("split", pa.string()))


def adapter_corpus(source, staging, repo_id, shard_files):
    rows = []
    for manifest_path in sorted(source.glob("3.*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        directory = manifest_path.parent
        entries = [json.loads(line) for line in (directory/"accepted.jsonl").read_text().splitlines()]
        if not manifest.get("complete") or manifest.get("smoke_only") or len(entries) != manifest["files"]:
            raise ValueError(f"{directory} is incomplete or smoke-only; refusing to publish")
        relative = directory.relative_to(source)
        (staging/relative).mkdir(parents=True)
        for name in ("accepted.jsonl", "manifest.json"):
            shutil.copy(directory/name, staging/relative/name)
        for index, start in enumerate(range(0, len(entries), shard_files)):
            table = [{**record(entry), "split": entry["split"]} for entry in entries[start:start+shard_files]]
            pq.write_table(pa.Table.from_pylist(table, schema=SPLIT_SCHEMA),
                           staging/relative/f"{relative.parts[1]}-{index:05d}.parquet", compression="zstd")
        rows.append(f"| {manifest['version']} | {manifest['split']} | {manifest['files']} | {manifest['corpora']['pypi']} | "
                    f"{manifest['corpora']['csn']} | {manifest['statement_rows']} | {manifest['compiler'].split()[0]} |")
        print(json.dumps({"staged": str(relative), "files": len(entries)}), flush=True)
    shutil.copy(source/"compiler-provenance.json", staging/"compiler-provenance.json")
    any_manifest = json.loads(next(source.glob("3.*/train/manifest.json")).read_text())
    table = "\n".join(rows)
    return f"""---
license: other
pretty_name: PyLingual shared-backbone adapter corpus
---
# {repo_id}

Per-version source pools for the Python 3.14 and 3.15 LoRAs of the shared-backbone PyLingual study.
Each row is one source file compiled for exactly one Python version, with every row the PyLingual
converter produced. Validation files are exactly the sources whose canonical hash falls in the
reserved bucket (`int(canonical_sha256[:8], 16) % 10 == 0`); training files never do.

- Disjoint from the backbone corpus (journal sha256 `{any_manifest['disjoint_from']['backbone_journal_sha256']}`),
  from every `pylingual_eval_dataset` v1/v2 source (exclusions sha256 `{any_manifest['disjoint_from']['exclusions_sha256']}`)
  and from each other, by raw SHA-512, canonical hash and NAS identity
- Seed `{any_manifest['seed']}`; sampled from `{any_manifest['nas_root']}`
- Code: https://github.com/quinntyx/pylingual branch `feat-shared-backbone-lora`, commit `{git_commit()}`

| Python | Split | Files | PyPI | CSN | Statement rows | Compiler |
|---|---|---:|---:|---:|---:|---|
{table}

`<version>/<split>/accepted.jsonl` is the exact journal the trainers hash-check.
"""


def encoded(source, staging, repo_id, sources):
    rows = []
    for manifest_path in sorted(source.glob("**/manifest.json")):
        if "build-cache" in manifest_path.parts:
            continue
        manifest = json.loads(manifest_path.read_text())
        directory = manifest_path.parent
        if not manifest.get("complete") or manifest.get("smoke_only"):
            raise ValueError(f"{directory} is incomplete or smoke-only; refusing to publish")
        relative = directory.relative_to(source)
        shutil.copytree(directory, staging/relative, ignore=shutil.ignore_patterns("build-cache"), copy_function=os.link)
        rows.append(f"| `{relative}` | {manifest['task']} | {manifest['scope']} | {manifest['split']} | {manifest['files']} | "
                    f"{manifest['encoded_rows']} | `{json.dumps(manifest['bindings'])}` |")
    table = "\n".join(rows)
    origins = "\n".join(f"- [`{hub['repo_id']}`](https://huggingface.co/{hub['repo_id'] if kind == 'tokenizer' else 'datasets/'+hub['repo_id']}) "
                        f"at revision `{hub['revision']}` ({kind})" for kind, hub in sources.items())
    return f"""---
license: other
pretty_name: PyLingual shared-backbone encoded training data
---
# {repo_id}

Tokenized Arrow datasets for the shared-backbone + per-version LoRA PyLingual study, exactly as the
trainers load them (`datasets.load_from_disk("<path>/dataset")`; `<path>/manifest.json` records the
journal, tokenizer and registry hashes the trainers verify). Statement inputs and targets are
truncated at 256 tokens; segmentation uses 512-token windows with stride 128.

Built from:
{origins}
- Code: https://github.com/quinntyx/pylingual branch `feat-shared-backbone-lora`, commit `{git_commit()}`

| Path | Task | Scope | Split | Source files | Encoded rows | Opcode bindings |
|---|---|---|---|---:|---:|---|
{table}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("adapter-corpus", "encoded"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--hub-record", type=Path, required=True, help="Where to write the published revision")
    parser.add_argument("--origin", action="append", default=[], metavar="KIND=HUB_JSON",
                        help="Upstream artifact records to cite in the card, e.g. tokenizer=tokenizer-hub.json")
    parser.add_argument("--shard-files", type=int, default=5000)
    args = parser.parse_args()
    staging = args.source.parent/(args.source.name+"-hf-staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    if args.kind == "adapter-corpus":
        card = adapter_corpus(args.source, staging, args.repo_id, args.shard_files)
    else:
        sources = {kind: json.loads(Path(path).read_text()) for kind, path in (item.split("=", 1) for item in args.origin)}
        card = encoded(args.source, staging, args.repo_id, sources)
    (staging/"README.md").write_text(card)
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=staging)
    info = api.dataset_info(args.repo_id)
    args.hub_record.write_text(json.dumps({"repo_id": args.repo_id, "revision": info.sha, "git_commit": git_commit()}, indent=2)+"\n")
    print(json.dumps({"published": args.repo_id, "revision": info.sha}), flush=True)


if __name__ == "__main__":
    main()
