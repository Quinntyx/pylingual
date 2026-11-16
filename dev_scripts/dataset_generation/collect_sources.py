# /// script
# requires-python = ">= 3.12"
# dependencies = [
#   "pylingual",
# ]
# [tool.uv.sources]
# pylingual = { path = "../../", editable = true }
# ///
"""
The collect stage: fill the source path of every data request with a `corpus` from the dataset NAS.

The NAS holds pylingual datasets as <nas_root>/py3X-pylingual-vN/code/{train,valid,test}/<pypi|csn>-<identity>/. A
source is a candidate if some train split has it and no valid or test split does, and no excluded eval dataset release
has it (by NAS identity, raw SHA-512 or normalized hash). Candidates are shuffled with the seed and kept in that order
if they are min_bytes to max_bytes, are no duplicate (raw or normalized), and normalize and compile with the dataset's
version, until each request has num_train + num_test + num_valid sources. The generate stage then loses none of them.
This is how the shared-backbone training corpus was sampled, and its NAS index can be reused.

Writes <source_path>/<identity>/<file>.py, and the collect dir of the dataset gets one JSONL manifest per request.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import tqdm

from .DatasetDescription import CollectDescription, DatasetDescription, get_dataset_description_from_arg_json
from .normalize_source import normalize_source
from pylingual.utils.generate_bytecode import UV_VERSIONS, _uv_python
from pylingual.utils.get_logger import get_logger
from pylingual.utils.version import PythonVersion

SNAPSHOT = re.compile(r"py3(\d+)-pylingual-v([0-9.]+)$")
NAS_ENTRY = re.compile(r"(pypi|csn)-([a-f0-9]{128}|[0-9]+)$")
EVAL_ENTRY = re.compile(r"[0-9a-f]{128}|[0-9]+")  # eval dataset dirs: a PyPI identity, or a CSN number


def nas_index(root: Path, cache: Path) -> dict:
    """Train candidates (identity, directory, corpus, snapshot version) of every NAS snapshot, without held-out ones."""
    if cache.exists():
        with gzip.open(cache, "rt") as stream:
            saved = json.load(stream)
        if saved["root"] != str(root) or not saved.get("complete") or saved.get("format") != 2:
            raise ValueError(f"{cache} is not an index of {root}")
        return saved
    snapshots = sorted(((int(m[1]), e.name) for e in os.scandir(root) if (m := SNAPSHOT.fullmatch(e.name)) and 6 <= int(m[1]) <= 15), reverse=True)

    def scan(snapshot):
        minor, name = snapshot
        counts, heldout, candidates = Counter(), set(), {}
        for split in ("valid", "test", "train"):
            try:
                with os.scandir(root / name / "code" / split) as entries:
                    for entry in entries:
                        if match := NAS_ENTRY.fullmatch(entry.name):
                            corpus, identity = match.groups()
                            counts[f"{split}_{corpus}"] += 1
                            if split != "train":
                                heldout.add(identity)
                            elif minor <= 14:
                                candidates.setdefault(identity, [entry.path, corpus, f"3.{minor}"])
            except FileNotFoundError:
                continue
        return name, dict(counts), heldout, candidates

    counts, heldout, candidates = {}, set(), {}
    with ThreadPoolExecutor(max_workers=4) as executor:  # one snapshot at a time per thread; never recurse
        for name, count, held, found in executor.map(scan, snapshots):
            counts[name] = count
            heldout |= held
            for identity, item in found.items():
                candidates.setdefault(identity, item)
    saved = {"format": 2, "root": str(root), "complete": True, "snapshots": counts, "heldout_identities": sorted(heldout),
             "candidates": [[identity, *item] for identity, item in candidates.items() if identity not in heldout],
             "training_candidates_before_heldout_exclusion": len(candidates)}  # fmt: skip
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache.with_suffix(".pending"), "wt", compresslevel=1) as stream:
        json.dump(saved, stream, separators=(",", ":"))
    cache.with_suffix(".pending").replace(cache)
    return saved


def canonical(text: str) -> str | None:
    """Hash of the source normalized for 3.14, the identity the shared-backbone corpus deduplicated by."""
    try:
        return hashlib.sha256(normalize_source(text, version=(3, 14), replace_docstrings=True).encode()).hexdigest()
    except Exception:
        return None


def eval_sources(collect: CollectDescription, checkout_dir: Path, logger: logging.Logger) -> tuple[set, set, set]:
    """NAS identities, raw SHA-512 and canonical hashes of every source in the excluded eval dataset releases."""
    identities, raw, canon = set(), set(), set()
    for release in collect.exclude:
        checkout = checkout_dir / release["ref"]
        if not checkout.exists():
            logger.info(f"Cloning {release['repo']} at {release['ref']}...")
            subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", release["ref"], release["repo"], str(checkout)], check=True)
        for version_dir in checkout.glob("python-3.*"):
            for entry in version_dir.iterdir():
                if entry.is_dir() and EVAL_ENTRY.fullmatch(entry.name):
                    identities.add(entry.name if len(entry.name) == 128 else "csn-" + entry.name)
                    for py in entry.glob("*.py"):
                        data = py.read_bytes()
                        raw.add(hashlib.sha512(data).hexdigest())
                        canon.add(canonical(data.decode("utf-8", "replace")))
    canon.discard(None)
    return identities, raw, canon


def interpreter(version: PythonVersion) -> str:
    if version == sys.version_info[:2]:
        return sys.executable
    if version in UV_VERSIONS and (python := _uv_python(version)):
        return python
    prefix = subprocess.run(["pyenv", "prefix", version.as_str()], capture_output=True, text=True) if shutil.which("pyenv") else None
    if prefix is None or prefix.returncode != 0:
        raise RuntimeError(f"no Python {version} to compile the collected sources with")
    return str(Path(prefix.stdout.strip().split(":")[0]) / "bin" / "python")


_compiler: subprocess.Popen | None = None
_settings: tuple = ()


def _init(python: str, version: tuple[int, int], min_bytes: int, max_bytes: int):
    global _compiler, _settings
    _settings = (version, min_bytes, max_bytes)
    _compiler = subprocess.Popen([python, "-I", str(Path(__file__).with_name("compile_check.py"))], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


def read_candidate(item: list) -> dict | str:
    """A candidate's record with its source, or why it is dropped."""
    identity, directory, corpus, _ = item
    version, min_bytes, max_bytes = _settings
    try:
        paths = [p for p in Path(directory).iterdir() if p.name.endswith(".py")]
        if len(paths) != 1:
            return "ambiguous-source-file"
        raw = paths[0].read_bytes()
        if not min_bytes <= len(raw) <= max_bytes:
            return "file-size"
        text = raw.decode("utf-8")
        normalized = normalize_source(text, version=version, replace_docstrings=True)
    except (OSError, UnicodeDecodeError) as error:
        return type(error).__name__
    except Exception:
        return "normalize"
    _compiler.stdin.write(json.dumps(normalized) + "\n")
    _compiler.stdin.flush()
    if (result := _compiler.stdout.readline().strip()) != "ok":
        return "compile-" + result
    return {"identity": identity, "corpus": corpus, "nas_path": str(paths[0]), "sha512": hashlib.sha512(raw).hexdigest(), "canonical_sha256": canonical(text), "bytes": len(raw), "source": text}


def collect_sources(description: DatasetDescription, logger: logging.Logger):
    collect = description.collect
    requests = [r for r in description.data_requests if r.corpus]
    description.collect_dir.mkdir(parents=True, exist_ok=True)
    index = nas_index(collect.nas_root, collect.index)
    identities, forbidden_raw, forbidden_canonical = eval_sources(collect, description.collect_dir / "eval-checkouts", logger)

    pools = {}
    for corpus in sorted({r.corpus for r in requests}):
        pools[corpus] = sorted(item for item in index["candidates"] if item[2] == corpus and item[0] not in identities and f"{corpus}-{item[0]}" not in identities)
        random.Random(f"{collect.seed}-{corpus}").shuffle(pools[corpus])
    logger.info(f"Candidates: { {c: len(p) for c, p in pools.items()} }; excluding {len(identities)} eval identities")

    seen_raw, seen_canonical, failures, summary = set(), set(), Counter(), {}
    python = interpreter(PythonVersion(description.version))
    with multiprocessing.Pool(initializer=_init, initargs=(python, description.version, collect.min_bytes, collect.max_bytes)) as pool:
        for request in requests:
            manifest = description.collect_dir / f"{request.name}.jsonl"
            if manifest.exists() and sum(1 for _ in manifest.open()) == request.total_files:
                logger.info(f"{request.name} is already collected")
                for record in map(json.loads, manifest.open()):
                    seen_raw.add(record["sha512"])
                    seen_canonical.add(record["canonical_sha256"])
                summary[request.name] = request.total_files
                continue
            if request.source_path.exists() and any(request.source_path.iterdir()):
                raise FileExistsError(f"{request.source_path} has sources but no complete manifest")
            candidates, taken = pools[request.corpus], 0
            with manifest.open("w") as out, tqdm.tqdm(total=request.total_files, desc=f"collect {request.name}") as bar:
                while taken < request.total_files and candidates:
                    batch = [candidates.pop() for _ in range(min(len(candidates), 2 * (request.total_files - taken) + 64))]
                    for record in pool.imap(read_candidate, batch, chunksize=8):
                        if isinstance(record, str):
                            failures[record] += 1
                        elif record["sha512"] in forbidden_raw or record["canonical_sha256"] in forbidden_canonical:
                            failures["eval-source"] += 1
                        elif record["sha512"] in seen_raw or (record["canonical_sha256"] and record["canonical_sha256"] in seen_canonical):
                            failures["duplicate"] += 1
                        elif taken < request.total_files:
                            seen_raw.add(record["sha512"])
                            seen_canonical.add(record["canonical_sha256"])
                            target = request.source_path / record["identity"] / Path(record["nas_path"]).name
                            target.parent.mkdir(parents=True)
                            target.write_text(record.pop("source"))
                            out.write(json.dumps(record) + "\n")
                            taken += 1
                            bar.update()
            if taken < request.total_files:
                raise RuntimeError(f"only {taken} of {request.total_files} {request.corpus} sources for {request.name}")
            summary[request.name] = taken
    summary = {"collected": summary, "failures": failures, "seed": collect.seed, "interpreter": python, "nas_snapshots": sorted(index["snapshots"]),
               "excluded": {"releases": collect.exclude, "identities": len(identities), "raw_sha512": len(forbidden_raw), "canonical_sha256": len(forbidden_canonical)}}  # fmt: skip
    (description.collect_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(f"Collected {summary['collected']}, dropped {dict(failures)}")


@click.command(help="Collect the sources of the data requests with a corpus from the dataset NAS.")
@click.argument("json_path", type=str)
def main(json_path: str):
    logger = get_logger("collect-sources")
    description = get_dataset_description_from_arg_json(json_path, logger)
    if description.collect is None:
        raise click.UsageError(f"{json_path} has no collect section")
    collect_sources(description, logger)


if __name__ == "__main__":
    main()
