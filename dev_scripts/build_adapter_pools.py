"""Sample fresh per-version adapter pools on this machine from the locally mounted NAS.

Each version gets 54,500 training files (49,050 PyPI + 5,450 CSN) and 500 validation files
(450 + 50). Validation files are exactly those whose canonical hash falls in the reserved
bucket, so the two splits never share a source. Every pool is disjoint from the backbone corpus,
from every eval-dataset source and from the other version's pool, by raw SHA-512, canonical hash
and NAS identity. Output matches prepare_full_training_data's --scope adapter input.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

from dev_scripts.prepare_tokenizer_corpus import extract
from dev_scripts.run_local_tokenizer_preparation import PINNED, executable
from dev_scripts.run_remote_ablation import Compiler, Gateway, Handler

TARGETS = {"train": {"pypi": 49050, "csn": 5450}, "validation": {"pypi": 450, "csn": 50}}
PATCHES = {**PINNED, "3.15": "3.15.0b2"}


def compiler_path(version):
    if version == "3.15":
        return shutil.which("python3.15")
    return executable(version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True, help="Backbone corpus data directory (accepted.jsonl)")
    parser.add_argument("--exclusions", type=Path, required=True, help="excluded-source-identities.json of the backbone corpus")
    parser.add_argument("--eval-reserved", type=Path, required=True, help="reserved-benchmark-sources.json of the backbone corpus")
    parser.add_argument("--nas-root", type=Path, default=Path("/syssec-nas1/pyc-research/datasets"))
    parser.add_argument("--nas-catalog", type=Path, required=True)
    parser.add_argument("--versions", nargs="+", choices=["3.14", "3.15"], default=["3.14", "3.15"])
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--compile-copies", type=int, default=4)
    parser.add_argument("--smoke-scale", type=int, default=0, help="Divide every quota by this; smoke only, never publish")
    args = parser.parse_args()
    targets = {split: {corpus: max(1, count//args.smoke_scale) if args.smoke_scale else count for corpus, count in quota.items()}
               for split, quota in TARGETS.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    backbone = [json.loads(line) for line in (args.backbone/"accepted.jsonl").open()]
    excluded = json.loads(args.exclusions.read_text())
    forbidden_raw = set(excluded["source_sha512"]) | {row["source_sha512"] for row in backbone} | {row["payload_sha512"] for row in backbone}
    forbidden_canonical = set(excluded["canonical_sha256"]) | {row["canonical_sha256"] for row in backbone}
    reserved = set(json.loads(args.eval_reserved.read_text())) | {row["nas_identity"] for row in backbone} | {row["source_sha512"] for row in backbone}
    reserved_path = args.output/"source-reserved.json"
    reserved_path.write_text(json.dumps(sorted(reserved))+"\n")
    journal = args.output/"accepted.jsonl"
    accepted = [json.loads(line) for line in journal.open()] if journal.exists() else []
    counts = Counter((row["version"], row["split"], row["corpus"]) for row in accepted)
    seen_raw = {row["source_sha512"] for row in accepted} | {row["payload_sha512"] for row in accepted}
    seen_canonical = {row["canonical_sha256"] for row in accepted}
    if seen_raw & forbidden_raw or seen_canonical & forbidden_canonical or len(seen_canonical) != len(accepted):
        raise ValueError("Journal overlaps the backbone/eval exclusions or contains duplicates")
    failures, started = Counter(), time.monotonic()
    compilers, pools, metadata, sampler = [], {}, {}, None
    try:
        for version in args.versions:
            pools[version] = queue.Queue()
            for _ in range(args.compile_copies):
                compiler = Compiler(compiler_path(version), str(args.repo/"dev_scripts/compile_source_server.py"))
                compilers.append(compiler)
                if compiler.metadata["version"].split()[0] != PATCHES[version]:
                    raise RuntimeError(f"{version} compiler is {compiler.metadata['version']}, expected {PATCHES[version]}")
                pools[version].put(compiler)
            metadata[version] = {"executable": compiler_path(version), **compiler.metadata, "servers": args.compile_copies}
        (args.output/"compiler-provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
        sampler = subprocess.Popen([sys.executable, "-u", str(args.repo/"dev_scripts/sample_nas_sources.py"),
                                    "--root", str(args.nas_root), "--catalog", str(args.nas_catalog),
                                    "--reserved", str(reserved_path), "--seed", str(args.seed),
                                    "--resume-offset", str(len(accepted))],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        primed = False

        def sample(corpus, count):
            nonlocal primed
            request = {"corpus": corpus, "count": count}
            if not primed:
                request["exclude_sha512"] = sorted(seen_raw)
                primed = True
            sampler.stdin.write(json.dumps(request)+"\n")
            sampler.stdin.flush()
            records = []
            while True:
                line = sampler.stdout.readline()
                if not line:
                    raise ConnectionError("NAS sampler exited")
                record = json.loads(line)
                if record.get("batch_complete"):
                    if not record["emitted"]:
                        raise RuntimeError("NAS source pool exhausted")
                    return records
                records.append(record)

        with Gateway(("127.0.0.1", 0), Handler) as gateway, journal.open("a") as stream, \
                ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as workers:
            gateway.compiler_pools = pools
            threading.Thread(target=gateway.serve_forever, daemon=True).start()
            port = gateway.server_address[1]

            def accept(result, split):
                row = result["provenance"]
                if row["source_sha512"] in forbidden_raw or row.get("payload_sha512") in forbidden_raw \
                        or row["canonical_sha256"] in forbidden_canonical:
                    failures["excluded"] += 1
                    return
                if row["source_sha512"] in seen_raw or row["canonical_sha256"] in seen_canonical:
                    failures["duplicate"] += 1
                    return
                row["split"] = split
                seen_raw.update((row["source_sha512"], row["payload_sha512"]))
                seen_canonical.add(row["canonical_sha256"])
                counts[(row["version"], split, row["corpus"])] += 1
                accepted.append(row)
                stream.write(json.dumps(row, separators=(",", ":"))+"\n")

            for version in args.versions:
                for corpus in ("pypi", "csn"):
                    def missing(split):
                        return targets[split][corpus]-counts[(version, split, corpus)]
                    while missing("train") > 0 or missing("validation") > 0:
                        # Oversample while validation is open: only ~10% of sources land in its bucket.
                        wanted = min(args.workers*4, max(missing("train"), 10*missing("validation")))
                        records = [r for r in sample(corpus, wanted) if r["source_sha512"] not in forbidden_raw
                                   and r["payload_sha512"] not in forbidden_raw]
                        jobs = [(record, version, str(args.output), port, "train") for record in records]
                        retry = []
                        for job, result in zip(jobs, workers.map(extract, jobs)):
                            if result.get("failure") == "reserved-validation-partition":
                                retry.append(job[:4]+("validation",))
                            elif "failure" in result:
                                failures[result["failure"]] += 1
                            elif missing("train") > 0:
                                accept(result, "train")
                            else:
                                failures["filled-quota"] += 1
                        retry = retry[:max(0, missing("validation"))]
                        for result in workers.map(extract, retry):
                            if "failure" in result:
                                failures[result["failure"]] += 1
                            elif missing("validation") > 0:
                                accept(result, "validation")
                        stream.flush()
                        report = {"files": len(accepted), "counts": [[*key, value] for key, value in sorted(counts.items())],
                                  "failures": dict(failures), "elapsed_seconds": round(time.monotonic()-started, 1),
                                  "smoke_scale": args.smoke_scale}
                        (args.output/"progress.json").write_text(json.dumps(report, indent=2)+"\n")
                        print(json.dumps(report), flush=True)
            gateway.shutdown()
        for version in args.versions:
            for split in ("train", "validation"):
                chosen = [row for row in accepted if row["version"] == version and row["split"] == split]
                corpora = dict(Counter(row["corpus"] for row in chosen))
                if corpora != targets[split]:
                    raise ValueError(f"Wrong final quotas for {version}/{split}: {corpora}")
                directory = args.output/version/split
                directory.mkdir(parents=True, exist_ok=True)
                (directory/"accepted.jsonl").write_text("".join(json.dumps(row, separators=(",", ":"))+"\n" for row in chosen))
                manifest = {"complete": True, "files": len(chosen), "version": version, "split": split, "corpora": corpora,
                            "smoke_only": bool(args.smoke_scale), "seed": args.seed,
                            "compiler": metadata[version]["version"],
                            "disjoint_from": {"backbone_journal_sha256": hashlib.sha256((args.backbone/"accepted.jsonl").read_bytes()).hexdigest(),
                                              "exclusions_sha256": hashlib.sha256(args.exclusions.read_bytes()).hexdigest(),
                                              "other_adapter_pools": True},
                            "nas_root": str(args.nas_root),
                            "nas_catalog_sha256": hashlib.sha256(args.nas_catalog.read_bytes()).hexdigest(),
                            "all_normal_converter_rows": True,
                            "statement_rows": sum(row["statement_rows"] for row in chosen),
                            "segmentation_rows": sum(row["segmentation_rows"] for row in chosen),
                            "observed_opcodes": sorted({op for row in chosen for op in row["observed_opcodes"]}),
                            "failures": dict(failures), "elapsed_seconds": round(time.monotonic()-started, 1)}
                (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        (args.output/"complete.json").write_text(json.dumps({"complete": True, "files": len(accepted),
                                                             "smoke_only": bool(args.smoke_scale)})+"\n")
    finally:
        if sampler is not None:
            sampler.stdin.close()
            try:
                sampler.wait(timeout=15)
            except subprocess.TimeoutExpired:
                sampler.terminate()
                sampler.wait(timeout=5)
        for compiler in compilers:
            compiler.close()


if __name__ == "__main__":
    main()
