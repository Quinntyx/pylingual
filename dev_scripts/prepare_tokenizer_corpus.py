"""Prepare raw all-row tokenizer corpus on the training host, never fit models."""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import tokenize
import warnings

from dev_scripts.tokenizer_opcode_registry import opcode_names


def extract(job):
    record, version, root, port, *options = job
    partition = options[0] if options else "train"
    if partition not in ("train", "validation"):
        raise ValueError("Invalid source partition")
    from dev_scripts.dataset_generation.normalize_source import normalize_source
    from dev_scripts.dataset_generation.bytecode2csv import bytecode2csv_exception_wrapper
    from pylingual.masking.ast_masker import add_dummy_decorators
    from pylingual.editable_bytecode import PYCFile
    try:
        warnings.simplefilter("ignore", SyntaxWarning)
        raw = base64.b64decode(record["raw_b64"], validate=True)
        if hashlib.sha512(raw).hexdigest() != record.get("payload_sha512", record["source_sha512"]):
            raise ValueError("Source transport checksum mismatch")
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding)
        if record.get("nas_source_version") in ("3.6", "3.7"):
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if hasattr(node, "decorator_list"):
                    node.decorator_list = [item for item in node.decorator_list
                        if not (isinstance(item, ast.Name) and item.id.startswith("PYLINGUAL_DUMMY_DECORATOR_"))]
            text = ast.unparse(tree)
        retained = record.get("retained_normalized", False)
        if retained:
            canonical = record["retained_canonical_sha256"]
            if len(canonical) != 64 or any(c not in "0123456789abcdef" for c in canonical):
                raise ValueError("Invalid retained canonical identity")
            # normalize_source is not idempotent. These selected bytes were already
            # normalized and independently verified against the retained archive.
        else:
            canonical = hashlib.sha256(normalize_source(text, version=(3, 14), replace_docstrings=True).encode()).hexdigest()
        if (int(canonical[:8], 16) % 10 == 0) != (partition == "validation"):
            return {"failure": "reserved-validation-partition" if partition == "train" else "wrong-canonical-partition"}
        minor = int(version.split(".")[1])
        normalized = text if retained else normalize_source(text, version=(3, minor), replace_docstrings=True)
        if minor <= 7:
            normalized = add_dummy_decorators(normalized)
        directory = Path(root)/"sources"/version/record["source_sha512"][:2]/record["source_sha512"]
        directory.mkdir(parents=True, exist_ok=True)
        py = directory/"source.py"
        pyc = directory/"source.pyc"
        py.write_text(normalized)
        with socket.create_connection(("127.0.0.1", port), timeout=40) as connection:
            stream = connection.makefile("rwb")
            stream.write((json.dumps({"version": version, "source": normalized, "filename": str(py)})+"\n").encode())
            stream.flush()
            response = json.loads(stream.readline(16000000))
        if not response.get("success"):
            return {"failure": "compile-"+version, "detail": response.get("error", "")[:200]}
        pyc.write_bytes(base64.b64decode(response["pyc_b64"], validate=True))
        if tuple(PYCFile(str(pyc)).version) != (3, minor):
            raise ValueError("Compiler bytecode version mismatch")
        rows = bytecode2csv_exception_wrapper((py, pyc))
        if isinstance(rows, Exception):
            return {"failure": "extract-"+version, "detail": str(rows)[:220]}
        segments, statements = rows
        if not segments or not statements:
            return {"failure": "no-rows-"+version}
        content = {"statement": [[row[0], row[1]] for row in statements],
                   "segmentation": [[row[0], row[1], row[2]] for row in segments]}
        observed = set()
        for task in content:
            for row in content[task]:
                observed.update(opcode_names(row[1]))
        chunk = directory/"examples.json.gz"
        with gzip.open(chunk, "wt", encoding="utf-8", compresslevel=1) as output:
            json.dump(content, output, separators=(",", ":"))
        provenance = {key: record[key] for key in ("source_id", "source_path", "source_sha512", "corpus")}
        if "payload_sha512" in record:
            provenance["payload_sha512"] = record["payload_sha512"]
            provenance["nas_source_version"] = record["nas_source_version"]
            provenance["nas_identity"] = record["nas_identity"]
        if retained:
            provenance["source_normalization"] = "verified-retained-normalized-bytes"
        provenance.update(version=version, canonical_sha256=canonical, file=str(py),
                          chunk=str(chunk), statement_rows=len(statements),
                          segmentation_rows=len(segments), observed_opcodes=sorted(observed))
        return {"provenance": provenance}
    except Exception as error:
        return {"failure": type(error).__name__+"-"+version, "detail": str(error)[:200]}
    finally:
        signal.alarm(0)


def quotas(total):
    if total % 10:
        raise ValueError("Total must allow an exact90/10 corpus split")
    versions = [f"3.{minor}" for minor in range(6, 15)]
    result = {version: {} for version in versions}
    for corpus, count in (("pypi", total*9//10), ("csn", total//10)):
        for index, version in enumerate(versions):
            result[version][corpus] = count//9 + int(index < count % 9)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--source-script", required=True)
    parser.add_argument("--source-reserved", required=True)
    parser.add_argument("--known-hosts")
    parser.add_argument("--nas-root", type=Path)
    parser.add_argument("--nas-catalog", type=Path)
    parser.add_argument("--gateway-port", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--files", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--sample-limit", type=int, default=0, help="Explicit smoke run, not a completed production corpus")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    excluded = json.loads(args.exclusions.read_text())
    forbidden_raw = set(excluded["source_sha512"])
    forbidden_canonical = set(excluded["canonical_sha256"])
    wanted = quotas(args.files)
    journal = args.output/"accepted.jsonl"
    accepted = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    counts = {version: Counter() for version in wanted}
    seen_raw, seen_canonical = set(), set()
    for p in accepted:
        if p["source_sha512"] in forbidden_raw or p["canonical_sha256"] in forbidden_canonical:
            raise ValueError("Journal contains excluded source")
        if p["source_sha512"] in seen_raw or p["canonical_sha256"] in seen_canonical:
            raise ValueError("Journal contains duplicate source")
        if not Path(p["chunk"]).is_file():
            raise ValueError("Accepted chunk is missing")
        counts[p["version"]][p["corpus"]] += 1
        seen_raw.add(p["source_sha512"])
        seen_canonical.add(p["canonical_sha256"])
    command = ["ssh", "-p", "2227", "-l", "henryh@syssec.utdallas.edu", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
               "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile="+(args.known_hosts or ""),
               "syssec-docker0.utdallas.edu", "python3", "-u", args.source_script,
               "--reserved", args.source_reserved, "--seed", str(args.seed), "--resume-offset", str(len(accepted))]
    if args.nas_root:
        if not args.nas_catalog:
            raise ValueError("NAS sampling requires a cached catalog path")
        command = [sys.executable, "-u", args.source_script, "--root", str(args.nas_root),
                   "--catalog", str(args.nas_catalog), "--reserved", args.source_reserved,
                   "--seed", str(args.seed), "--resume-offset", str(len(accepted))]
    process = None
    primed = False
    failures = Counter()
    attempts_by_version = Counter()
    examples = {}
    start = time.time()

    def sample(corpus, count):
        nonlocal process, primed
        for attempt in range(6):
            try:
                if process is None or process.poll() is not None:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
                    primed = False
                request = {"corpus": corpus, "count": count}
                if not primed:
                    request["exclude_sha512"] = sorted(forbidden_raw | seen_raw)
                process.stdin.write(json.dumps(request)+"\n")
                process.stdin.flush()
                primed = True
                records = []
                while True:
                    line = process.stdout.readline()
                    if not line:
                        raise ConnectionError("Source sampler disconnected")
                    record = json.loads(line)
                    if record.get("batch_complete"):
                        if not record["emitted"]:
                            raise RuntimeError("Source corpus exhausted")
                        return records
                    records.append(record)
            except (OSError, ConnectionError, json.JSONDecodeError):
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                process = None
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 10))

    try:
        with journal.open("a") as stream, ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
            while len(accepted) < args.files:
                made_progress = False
                for corpus in ("pypi", "csn"):
                    allocation = []
                    remaining = {version: wanted[version][corpus]-counts[version][corpus] for version in wanted}
                    while len(allocation) < args.workers*4 and any(value > 0 for value in remaining.values()):
                        for version in wanted:
                            if remaining[version] > 0 and len(allocation) < args.workers*4:
                                allocation.append(version)
                                remaining[version] -= 1
                    if not allocation:
                        continue
                    records = sample(corpus, len(allocation))
                    jobs = [(record, version, str(args.output), args.gateway_port) for record, version in zip(records, allocation)]
                    for job, result in zip(jobs, pool.map(extract, jobs)):
                        attempts_by_version[job[1]] += 1
                        if "failure" in result:
                            key = result["failure"]
                            failures[key] += 1
                            examples.setdefault(key, result.get("detail", ""))
                            continue
                        p = result["provenance"]
                        if p["source_sha512"] in forbidden_raw or p["canonical_sha256"] in forbidden_canonical:
                            failures["excluded"] += 1
                            continue
                        if p["source_sha512"] in seen_raw or p["canonical_sha256"] in seen_canonical:
                            failures["duplicate"] += 1
                            continue
                        if counts[p["version"]][corpus] >= wanted[p["version"]][corpus]:
                            failures["filled-quota"] += 1
                            continue
                        seen_raw.add(p["source_sha512"])
                        seen_canonical.add(p["canonical_sha256"])
                        counts[p["version"]][corpus] += 1
                        accepted.append(p)
                        stream.write(json.dumps(p)+"\n")
                        made_progress = True
                    stream.flush()
                    progress = {"accepted": len(accepted), "target": args.files, "counts": counts,
                                "elapsed_seconds": round(time.time()-start, 2), "failures": failures,
                                "failure_examples": examples, "complete": False}
                    (args.output/"progress.json").write_text(json.dumps(progress, indent=2)+"\n")
                    print(json.dumps({key: progress[key] for key in ("accepted", "target", "elapsed_seconds", "failures")}), flush=True)
                    if args.sample_limit and len(accepted) >= args.sample_limit:
                        return
                for version, attempts in attempts_by_version.items():
                    if attempts >= 100 and not sum(counts[version].values()):
                        raise RuntimeError("No accepted files for "+version+" after "+str(attempts)
                                           +" attempts; inspect failure_examples before continuing")
                if not made_progress and sum(failures.values()) > 5000 and not accepted:
                    raise RuntimeError("No accepted files; inspect failure_examples before continuing")
        if counts != wanted or len(accepted) != args.files:
            raise ValueError("Final quotas did not match")
        manifest = {"complete": True, "files": len(accepted), "versions": list(wanted), "counts": counts,
                    "seed": args.seed, "mix_unit": "unique source files", "heldout_315": True,
                    "exclusions_sha256": hashlib.sha256(args.exclusions.read_bytes()).hexdigest(),
                    "source_host": socket.gethostname() if args.nas_root else "syssec-adwaith1",
                    "nas_root": str(args.nas_root) if args.nas_root else None,
                    "nas_catalog_sha256": hashlib.sha256(args.nas_catalog.read_bytes()).hexdigest() if args.nas_root else None,
                    "training_host": socket.gethostname(),
                    "sampling": "all converter rows per file; no old tokenizer filters or row caps",
                    "statement_rows": sum(p["statement_rows"] for p in accepted),
                    "segmentation_rows": sum(p["segmentation_rows"] for p in accepted),
                    "failures": failures, "elapsed_seconds": round(time.time()-start, 2)}
        (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps(manifest), flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
