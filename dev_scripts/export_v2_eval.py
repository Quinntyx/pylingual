"""Export file-diverse TEST rows from HF model datasets, not the v2 .pyc benchmark.

These similarly named Hugging Face datasets are useful for component checks only.
Use evaluate_shared.py with pylingual_eval_dataset for v2 whole-file accuracy.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

VERSIONS = tuple(f"3.{minor}" for minor in range(6, 15))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--statement-rows", type=int, default=256)
    parser.add_argument("--segmentation-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.training_manifest.read_text())
    training_hashes = {row["source_sha512"] for row in manifest["files"] if row["split"] == "train"}
    receipts = []
    for task, target, per_file in (("statement", args.statement_rows, 4), ("segmentation", args.segmentation_rows, 1)):
        for version in VERSIONS:
            repo = f"syssec-utd/{task}-py{version.replace('.', '')}-pylingual-v2"
            info = HfApi().dataset_info(repo, token=True)
            parquet_files = [item.rfilename for item in info.siblings if item.rfilename.startswith("data/test-") and item.rfilename.endswith(".parquet")]
            if not parquet_files:
                raise ValueError(f"No explicit test parquet files in {repo}")
            local_files = [hf_hub_download(repo, file, repo_type="dataset", revision=info.sha, token=True) for file in parquet_files]
            dataset = load_dataset("parquet", data_files={"test": local_files}, split="test", streaming=True).shuffle(seed=args.seed, buffer_size=2048)
            rows = []
            counts = Counter()
            excluded = 0
            for row in dataset:
                file = str(row.get("file", ""))
                if not file:
                    raise ValueError(f"Missing file provenance in {repo}")
                hashes = set(re.findall(r"[0-9a-f]{128}", file))
                if hashes & training_hashes:
                    excluded += 1
                    continue
                if counts[file] >= per_file:
                    continue
                rows.append({**row, "version": version, "eval_repo": repo, "eval_split": "test"})
                counts[file] += 1
                if len(rows) >= target:
                    break
            if len(rows) != target:
                raise ValueError(f"Only {len(rows)} file-diverse rows available for {repo}, wanted {target}")
            out = args.output / f"{task}-{version}.jsonl"
            out.write_text("".join(json.dumps(row) + "\n" for row in rows))
            receipt = {"task": task, "version": version, "repo": repo, "split": "test", "revision": info.sha, "rows": len(rows), "files": len(counts), "excluded_train_hash_matches": excluded, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "first_file": rows[0]["file"]}
            receipts.append(receipt)
            (args.output / "manifest.json").write_text(json.dumps({"seed": args.seed, "selection": "streaming shuffled test split; per-file cap; source-hash overlap filter", "hash_overlap_limit": "Only hashes exposed by the v2 file column can be checked; upstream canonical-source overlap is not available.", "datasets": receipts}, indent=2) + "\n")
            print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
