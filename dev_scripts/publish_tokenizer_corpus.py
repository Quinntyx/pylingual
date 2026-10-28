"""Publish a completed tokenizer corpus to a public Hugging Face dataset repo (see AGENTS.md).

Each row is one accepted source file: identities, normalized source, compiled pyc and every
converter row for both tasks. Smoke or partial corpora have no complete manifest and are refused.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

SCHEMA = pa.schema([
    ("source_sha512", pa.string()), ("payload_sha512", pa.string()), ("nas_identity", pa.string()),
    ("corpus", pa.string()), ("version", pa.string()), ("canonical_sha256", pa.string()),
    ("source", pa.string()), ("pyc", pa.binary()),
    ("statement", pa.list_(pa.struct([("source", pa.string()), ("bytecode", pa.string())]))),
    ("segmentation", pa.list_(pa.struct([("source", pa.string()), ("bytecode", pa.string()), ("boundary", pa.string())]))),
])


def record(entry):
    with gzip.open(entry["chunk"], "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    directory = Path(entry["chunk"]).parent
    return {"source_sha512": entry["source_sha512"], "payload_sha512": entry.get("payload_sha512"),
            "nas_identity": entry.get("nas_identity"), "corpus": entry["corpus"], "version": entry["version"],
            "canonical_sha256": entry["canonical_sha256"], "source": (directory/"source.py").read_text(),
            "pyc": (directory/"source.pyc").read_bytes(),
            "statement": [{"source": s, "bytecode": b} for s, b in rows["statement"]],
            "segmentation": [{"source": s, "bytecode": b, "boundary": str(y)} for s, b, y in rows["segmentation"]]}


def card(repo_id, manifest, commit):
    counts = "\n".join(f"| {v} | {c['pypi']} | {c['csn']} |" for v, c in manifest["counts"].items())
    return f"""---
license: other
pretty_name: PyLingual shared tokenizer corpus
---
# {repo_id}

Version-balanced corpus used to fit the shared-backbone PyLingual tokenizers. Each row is one
source file compiled for exactly one Python version, with every row the PyLingual converter
produced for the statement and segmentation tasks.

- Files: **{manifest['files']}** ({manifest['statement_rows']} statement rows, {manifest['segmentation_rows']} segmentation rows)
- Seed: `{manifest['seed']}`; Python 3.15 held out: `{manifest['heldout_315']}`
- Excluded: NAS validation/test identities and every `pylingual_eval_dataset` v1/v2 source
  (exclusions sha256 `{manifest['exclusions_sha256']}`)
- Built from `{manifest['nas_root']}` with branch `feat-shared-backbone-lora` at commit `{commit}`
  of https://github.com/quinntyx/pylingual

| Python | PyPI files | CSN files |
|---|---:|---:|
{counts}

`manifest.json`, `accepted.jsonl`, `compiler-provenance.json` and `excluded-source-identities.json`
record the exact provenance.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--shard-files", type=int, default=5000)
    args = parser.parse_args()
    data = args.workspace/"data"
    manifest = json.loads((data/"manifest.json").read_text())
    entries = [json.loads(line) for line in (data/"accepted.jsonl").read_text().splitlines()]
    if not manifest.get("complete") or len(entries) != manifest["files"]:
        raise ValueError("Corpus is incomplete; refusing to publish")
    staging = args.workspace/"hf-staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging/"data").mkdir(parents=True)
    for index, start in enumerate(range(0, len(entries), args.shard_files)):
        rows = [record(entry) for entry in entries[start:start+args.shard_files]]
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), staging/"data"/f"train-{index:05d}.parquet", compression="zstd")
        print(json.dumps({"shard": index, "files": start+len(rows)}), flush=True)
    for name in ("manifest.json", "accepted.jsonl"):
        shutil.copy(data/name, staging/name)
    for name in ("compiler-provenance.json", "excluded-source-identities.json"):
        shutil.copy(args.workspace/name, staging/name)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                            cwd=Path(__file__).resolve().parent).stdout.strip()
    (staging/"README.md").write_text(card(args.repo_id, manifest, commit))
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=staging)
    info = api.dataset_info(args.repo_id)
    (args.workspace/"hub.json").write_text(json.dumps({"repo_id": args.repo_id, "revision": info.sha, "git_commit": commit}, indent=2)+"\n")
    print(json.dumps({"published": args.repo_id, "revision": info.sha}), flush=True)


if __name__ == "__main__":
    main()
