"""Publish a completed shared tokenizer family to a public Hugging Face model repo (see AGENTS.md).

The repo holds statement/ and segmentation/ tokenizers with their opcode registries plus the
fitting manifest. Its card pins the corpus dataset revision recorded by publish_tokenizer_corpus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from huggingface_hub import HfApi

from pylingual.opcodes import RESERVED_SLOT_COUNT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    tokenizers = args.workspace/"tokenizers"
    manifest = json.loads((tokenizers/"manifest.json").read_text())
    corpus = json.loads((args.workspace/"hub.json").read_text())
    if not manifest.get("complete") or manifest["reserved_slots"] != RESERVED_SLOT_COUNT:
        raise ValueError("Tokenizer family is incomplete or has the wrong slot count; refusing to publish")
    staging = args.workspace/"hf-tokenizer-staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(tokenizers, staging)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                            cwd=Path(__file__).resolve().parent).stdout.strip()
    rows = "\n".join(f"| {task} | {t['base_vocab_size']} | {t['total_vocab_size']} | {t['checks']['reserved_ids'][0]}–{t['checks']['reserved_ids'][-1]} |"
                     for task, t in manifest["tasks"].items())
    (staging/"README.md").write_text(f"""---
license: other
---
# {args.repo_id}

Shared (all-version) PyLingual tokenizers for the universal backbone + per-version LoRA study.
Each task has one frozen tokenizer; every known opcode is a single atomic token and
{RESERVED_SLOT_COUNT} contiguous `<future_opcode_NN>` slots are reserved for instructions from
future Python versions, bound per adapter.

- Fitted on dataset [`{corpus['repo_id']}`](https://huggingface.co/datasets/{corpus['repo_id']}) at revision `{corpus['revision']}`
- Total vocabulary budget {manifest['total_vocab_budget']} (opcode markers and reserved slots included); {manifest['known_opcode_count']} known opcodes
- Code: https://github.com/quinntyx/pylingual branch `feat-shared-backbone-lora`, commit `{commit}`

| Task | Learned base vocab | Total vocab | Reserved IDs |
|---|---:|---:|---|
{rows}

Load a task with `AutoTokenizer.from_pretrained("{args.repo_id}", subfolder="statement")`.
""")
    api = HfApi()
    api.create_repo(args.repo_id, private=False, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, folder_path=staging, commit_message="Shared tokenizer family v1")
    info = api.model_info(args.repo_id)
    (args.workspace/"tokenizer-hub.json").write_text(json.dumps({"repo_id": args.repo_id, "revision": info.sha, "git_commit": commit}, indent=2)+"\n")
    print(json.dumps({"published": args.repo_id, "revision": info.sha}), flush=True)


if __name__ == "__main__":
    main()
