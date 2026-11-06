"""Pin, download, and index complete PyLingual corpora without row/file caps.

Artifacts stay outside Git. Network credentials use the existing Hugging Face
credential store; no uploads are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

HASH = re.compile(r"(?<![0-9a-f])[0-9a-f]{128}(?![0-9a-f])")


def source_id(file):
    hashes = set(HASH.findall(str(file)))
    if len(hashes) != 1:
        raise ValueError(f"Expected one source SHA-512 in file provenance: {file!r}")
    return hashes.pop()


def index_repository(pin, output):
    destination = output / "indexes" / f"{pin['task']}-{pin['version']}.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["pin"] != pin:
            raise ValueError(f"Pinned input changed: {destination}")
        return saved
    started = time.monotonic()
    files = []
    by_split = {}
    for name in pin["files"]:
        local = hf_hub_download(pin["repo"], name, repo_type="dataset", revision=pin["revision"], token=True, cache_dir=str(output / "hub"))
        parquet = pq.ParquetFile(local)
        split = Path(name).name.split("-")[0]
        rows = Counter()
        example = None
        for batch in parquet.iter_batches(batch_size=32768, columns=["file"]):
            for file in batch.column(0).to_pylist():
                rows[source_id(file)] += 1
                example = example or file
        state = by_split.setdefault(split, {"rows": 0, "files": {}})
        state["rows"] += sum(rows.values())
        for key, value in rows.items():
            state["files"][key] = state["files"].get(key, 0) + value
        files.append({"name": name, "path": local, "split": split, "rows": parquet.metadata.num_rows, "bytes": Path(local).stat().st_size})
        print(json.dumps({"event": "indexed_shard", "repo": pin["repo"], "file": name, "rows": parquet.metadata.num_rows}), flush=True)
    result = {"pin": pin, "files": files, "splits": by_split, "seconds": time.monotonic() - started}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result) + "\n")
    temporary.replace(destination)
    print(json.dumps({"event": "indexed_repository", "repo": pin["repo"], "splits": {s: {"rows": v["rows"], "files": len(v["files"])} for s, v in by_split.items()}, "seconds": result["seconds"]}), flush=True)
    return result


def inventory(pins_path, output, workers=4):
    pins = json.loads(pins_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        indexes = list(pool.map(lambda pin: index_repository(pin, output), pins["datasets"]))
    summary = {"pins_sha256": hashlib.sha256(pins_path.read_bytes()).hexdigest(), "datasets": [{"repo": r["pin"]["repo"], "revision": r["pin"]["revision"], "task": r["pin"]["task"], "version": r["pin"]["version"], "bytes": sum(f["bytes"] for f in r["files"]), "splits": {s: {"rows": v["rows"], "files": len(v["files"])} for s, v in r["splits"].items()}} for r in indexes]}
    (output / "inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "inventory_complete", **summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    inventory(args.pins, args.output, args.workers)


if __name__ == "__main__":
    main()
