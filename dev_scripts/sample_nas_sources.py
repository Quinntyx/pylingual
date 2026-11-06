"""Read selected source files directly from GPU4's read-only dataset NAS mount.

Catalog only code/{train,valid,test} child names; never recurse over the NAS.
All validation/test source identities across snapshots are excluded globally.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

DATASET = re.compile(r"py3(\d+)-pylingual-v([0-9.]+)$")
IDENTITY = re.compile(r"(pypi|csn)-([a-f0-9]{128}|[0-9]+)$")


def catalog(root: Path, cache: Path):
    if cache.exists():
        with gzip.open(cache, "rt") as stream:
            saved = json.load(stream)
        if saved["root"] != str(root) or not saved.get("complete") or saved.get("format") != 2:
            raise ValueError("NAS catalog does not match requested root")
        return saved
    snapshots = []
    for entry in os.scandir(root):
        match = DATASET.fullmatch(entry.name)
        if match and 6 <= int(match.group(1)) <= 15:
            snapshots.append((int(match.group(1)), tuple(int(x) for x in match.group(2).split(".")), entry.name))
    snapshots.sort(reverse=True)
    heldout, candidates, counts = set(), {}, {}
    def scan_snapshot(snapshot):
        minor, revision, name = snapshot
        started = time.monotonic()
        count, local_heldout, local_candidates = Counter(), set(), {}
        for split in ("valid", "test", "train"):
            directory = root/name/"code"/split
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        match = IDENTITY.fullmatch(entry.name)
                        if not match:
                            continue
                        corpus, identity = match.groups()
                        count[split+"_"+corpus] += 1
                        if split != "train":
                            local_heldout.add(identity)
                        elif minor <= 14:
                            local_candidates.setdefault(identity, [entry.path, corpus, f"3.{minor}"])
            except FileNotFoundError:
                continue
        print(json.dumps({"nas_index_snapshot": name, "counts": count, "seconds": round(time.monotonic()-started, 2)}), file=sys.stderr, flush=True)
        return name, dict(count), local_heldout, local_candidates
    with ThreadPoolExecutor(max_workers=4) as executor:
        for name, count, local_heldout, local_candidates in executor.map(scan_snapshot, snapshots):
            counts[name] = count
            heldout.update(local_heldout)
            for identity, item in local_candidates.items():
                candidates.setdefault(identity, item)
    kept = [[identity, *item] for identity, item in candidates.items() if identity not in heldout]
    saved = {"format": 2, "root": str(root), "complete": True, "snapshots": counts, "heldout_identities": sorted(heldout),
             "candidates": kept, "training_candidates_before_heldout_exclusion": len(candidates)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix+".pending")
    with gzip.open(temporary, "wt", compresslevel=1) as stream:
        json.dump(saved, stream, separators=(",", ":"))
    temporary.replace(cache)
    return saved


def read_candidate(item):
    identity, directory, corpus, source_version = item
    try:
        paths = []
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name.endswith(".py"):
                    paths.append(entry.path)
        if len(paths) != 1:
            return {"failure": "ambiguous-source-file"}
        path = Path(paths[0])
        with path.open("rb") as stream:
            raw = stream.read(1000001)
        if not 100 <= len(raw) <= 100000:
            return {"failure": "file-size"}
        digest = hashlib.sha512(raw).hexdigest()
        return {"corpus": corpus, "source_id": identity, "source_path": str(path),
                "source_sha512": identity if len(identity) == 128 else digest,
                "nas_identity": corpus+"-"+identity,
                "payload_sha512": digest, "nas_source_version": source_version,
                "raw_b64": base64.b64encode(raw).decode("ascii")}
    except OSError as error:
        return {"failure": type(error).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--resume-offset", type=int, default=0)
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    saved = catalog(args.root, args.catalog)
    pools = {name: [] for name in ("pypi", "csn")}
    reserved = set(json.loads(args.reserved.read_text()))
    for item in saved["candidates"]:
        if item[0] not in reserved and item[2]+"-"+item[0] not in reserved:
            pools[item[2]].append(item)
    rng = random.Random(args.seed+args.resume_offset)
    for pool in pools.values():
        rng.shuffle(pool)
    print(json.dumps({"nas_catalog_ready": True, "candidates": {key: len(value) for key, value in pools.items()},
                      "heldout_source_identities": len(saved["heldout_identities"]),
                      "catalog": str(args.catalog)}), file=sys.stderr, flush=True)
    if args.index_only:
        return
    seen, failures = set(), Counter()
    with ThreadPoolExecutor(max_workers=args.read_workers) as executor:
        for line in sys.stdin:
            request = json.loads(line)
            corpus, wanted = request["corpus"], int(request["count"])
            if corpus not in pools or not 1 <= wanted <= 2000:
                raise ValueError("Invalid NAS source request")
            seen.update(request.get("exclude_sha512", ()))
            emitted = 0
            while emitted < wanted and pools[corpus]:
                batch = [pools[corpus].pop() for _ in range(min(wanted-emitted, len(pools[corpus])))]
                for record in executor.map(read_candidate, batch):
                    if "failure" in record:
                        failures[record["failure"]] += 1
                        continue
                    if record["source_sha512"] in reserved or record["source_sha512"] in seen or record["payload_sha512"] in seen:
                        failures["reserved-or-duplicate"] += 1
                        continue
                    seen.add(record["source_sha512"])
                    seen.add(record["payload_sha512"])
                    print(json.dumps(record), flush=True)
                    emitted += 1
            print(json.dumps({"batch_complete": True, "emitted": emitted, "remaining": len(pools[corpus]), "failures": failures}), flush=True)


if __name__ == "__main__":
    main()
