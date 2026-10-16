"""Publish a new model family by training only one adapter against a frozen backbone.

The original family is never modified. Existing version adapters and the exact
backbone/tokenizer are copied unchanged, retaining their compatibility identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from dev_scripts.shared_training import encode_row, read_rows, train_steps
from pylingual.shared_models import load_family
from pylingual.utils.version import PythonVersion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="JSONL rows for one version, with file/source/bytecode and segmentation boundary fields")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New family directory; must not exist")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    version = str(PythonVersion(args.version))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite family {args.output}")
    if args.family.resolve() in args.output.resolve().parents:
        raise ValueError("Output family cannot be nested inside its source")
    if not torch.cuda.is_available():
        raise RuntimeError("Adapter training requires a CUDA GPU")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    metadata = json.loads((args.family / "family.json").read_text())
    task = metadata["task"]
    bank, tokenizer, _, _ = load_family(args.family, task, torch.device("cuda:0"))
    by_file = defaultdict(list)
    dropped = 0
    unknown_tokens = input_tokens = 0
    for row in read_rows(args.data):
        if row["version"] != version:
            raise ValueError(f"Adapter input contains version {row['version']}, expected {version}")
        encoded = encode_row(task, tokenizer, row)
        if encoded is None:
            dropped += 1
            continue
        unknown_tokens += encoded["input_ids"].count(tokenizer.unk_token_id)
        input_tokens += len(encoded["input_ids"])
        by_file[row["file"]].append(encoded)
    if not by_file:
        raise ValueError("No usable adapter training examples")
    if version in metadata["adapters"]:
        bank.load(version, args.family / metadata["adapters"][version])
    else:
        bank.add(version, rank=args.rank, alpha=2 * args.rank, dropout=0.05)
    # Copy unchanged artifacts, never merge a LoRA or resize the shared tokenizer.
    shutil.copytree(args.family, args.output, ignore=shutil.ignore_patterns("family.json"))
    updated = dict(metadata)
    updated["complete"] = False
    (args.output / "family.json").write_text(json.dumps(updated, indent=2) + "\n")
    parameters = bank.trainable(version)
    bank.model.config.use_cache = False
    history = train_steps(bank.model, tokenizer, {version: list(by_file.values())}, (version,), args.steps, args.batch_size, args.accumulation, 3e-4, args.seed, parameters)
    relative = f"adapters/{version}"
    bank.save(version, args.output / relative)
    updated["versions"] = sorted(set(metadata["versions"]) | {version}, key=lambda text: PythonVersion(text).as_tuple())
    updated["adapters"] = {**metadata["adapters"], version: relative}
    updated["adapter_updates"] = {**metadata.get("adapter_updates", {}), version: {"seed": args.seed, "source_family": str(args.family.resolve()), "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(), "files": len(by_file), "dropped_rows": dropped, "unknown_input_tokens": unknown_tokens, "input_tokens": input_tokens, "parameters": sum(p.numel() for p in parameters), "history": history}}
    updated["complete"] = True
    (args.output / "family.json").write_text(json.dumps(updated, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output), "backbone_id": bank.backbone_id, "version": version, "unknown_input_tokens": unknown_tokens, "input_tokens": input_tokens}), flush=True)


if __name__ == "__main__":
    main()
