"""Build the tokenizer corpus on this machine from the locally mounted NAS; never fits models.

Native compile servers run here behind a localhost gateway, and prepare_tokenizer_corpus
reads NAS sources directly. Every eval-dataset source (v1 and v2, all versions) is excluded
by NAS identity, raw SHA-512 and, where the source is available, canonical hash.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading

from dev_scripts.run_remote_ablation import Compiler, Gateway, Handler

# Same patch releases as the 2026-10-19 family, so bytecode matches across corpora.
PINNED = {"3.6": "3.6.15", "3.7": "3.7.17", "3.8": "3.8.20", "3.9": "3.9.25", "3.10": "3.10.20",
          "3.11": "3.11.15", "3.12": "3.12.13", "3.13": "3.13.14", "3.14": "3.14.6"}
IDENTITY = re.compile(r"[0-9a-f]{128}|[0-9]+")


def canonical(path):
    import tokenize
    import warnings
    from dev_scripts.dataset_generation.normalize_source import normalize_source
    try:
        warnings.simplefilter("ignore", SyntaxWarning)
        with tokenize.open(path) as stream:
            text = stream.read()
        return hashlib.sha256(normalize_source(text, version=(3, 14), replace_docstrings=True).encode()).hexdigest()
    except Exception:
        return None


def eval_exclusions(eval_root: Path, workers: int):
    identities, sources = set(), {}
    for release in ("pylingualv1", "pylingualv2"):
        for version_dir in (eval_root/release).glob("python-3.*"):
            for entry in version_dir.iterdir():
                if entry.is_dir() and IDENTITY.fullmatch(entry.name):
                    identities.add(entry.name)
                    for py in entry.glob("*.py"):
                        sources.setdefault(hashlib.sha512(py.read_bytes()).hexdigest(), py)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        canon = {value for value in pool.map(canonical, sources.values(), chunksize=64) if value}
    pypi = sorted(i for i in identities if len(i) == 128)
    csn = sorted("csn-"+i for i in identities if len(i) != 128)
    exclusions = {"source_sha512": sorted(set(pypi) | set(sources)), "canonical_sha256": sorted(canon),
                  "nas_identities": csn, "origin": str(eval_root)}
    return exclusions, sorted(set(pypi) | set(csn) | set(sources))


def executable(version):
    pyenv = Path.home()/".pyenv/versions"/PINNED[version]/"bin/python"
    return str(pyenv) if pyenv.is_file() else shutil.which("python"+version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--nas-root", type=Path, default=Path("/syssec-nas1/pyc-research/datasets"))
    parser.add_argument("--eval-root", type=Path, default=Path.home()/"docs/src/pylingual_eval_dataset")
    parser.add_argument("--files", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--compile-copies", type=int, default=2)
    parser.add_argument("--sample-limit", type=int, default=0, help="Smoke runs only; never publish their output")
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    exclusion_path, reserved_path = args.workspace/"excluded-source-identities.json", args.workspace/"reserved-benchmark-sources.json"
    if not exclusion_path.exists():
        exclusions, reserved = eval_exclusions(args.eval_root, args.workers)
        exclusion_path.write_text(json.dumps(exclusions)+"\n")
        reserved_path.write_text(json.dumps(reserved)+"\n")
    compilers, pools, metadata = [], {}, {}
    try:
        for version, patch in PINNED.items():
            path = executable(version)
            pools[version] = queue.Queue()
            for _ in range(args.compile_copies):
                compiler = Compiler(path, str(args.repo/"dev_scripts/compile_source_server.py"))
                compilers.append(compiler)
                if compiler.metadata["version"].split()[0] != patch:
                    raise RuntimeError(f"{path} is {compiler.metadata['version'].split()[0]}, expected {patch}")
                if not compiler.request({"source": "x = 1\n", "filename": "<preflight>"}).get("success"):
                    raise RuntimeError("Compiler preflight failed: "+version)
                pools[version].put(compiler)
            metadata[version] = {"executable": path, **compiler.metadata, "servers": args.compile_copies}
        (args.workspace/"compiler-provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
        with Gateway(("127.0.0.1", 0), Handler) as gateway:
            gateway.compiler_pools = pools
            threading.Thread(target=gateway.serve_forever, daemon=True).start()
            command = [sys.executable, "-u", "-m", "dev_scripts.prepare_tokenizer_corpus",
                       "--output", args.workspace/("data-smoke" if args.sample_limit else "data"),
                       "--exclusions", exclusion_path, "--source-script", args.repo/"dev_scripts/sample_nas_sources.py",
                       "--source-reserved", reserved_path, "--nas-root", args.nas_root,
                       "--nas-catalog", args.workspace/"nas-catalog.json.gz", "--gateway-port", gateway.server_address[1],
                       "--workers", args.workers, "--files", args.files, "--seed", args.seed,
                       "--sample-limit", args.sample_limit]
            environment = {**os.environ, "PYTHONPATH": str(args.repo), "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
            try:
                subprocess.run([str(value) for value in command], check=True, cwd=args.repo, env=environment)
            finally:
                gateway.shutdown()
    finally:
        for compiler in compilers:
            compiler.close()


if __name__ == "__main__":
    main()
