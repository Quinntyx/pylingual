"""Sample archived PyPI sources, normalize/compile per version, and generate rows.

Never executes package code or extracts archive paths. Equal successful *files*
per version, with global content-hash split assignment and recorded provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import subprocess
import tarfile
import tokenize
import zipfile
from collections import Counter
from pathlib import Path

from dev_scripts.dataset_generation.bytecode2csv import bytecode2csv_exception_wrapper
from dev_scripts.dataset_generation.normalize_source import normalize_source
from pylingual.masking.ast_masker import add_dummy_decorators

VERSIONS = tuple(f"3.{minor}" for minor in range(6, 15))


def source_members(archive):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as reader:
            for info in reader.infolist():
                if info.filename.endswith(".py") and 100 <= info.file_size <= 100_000:
                    yield info.filename, reader.read(info)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as reader:
            for info in reader:
                if info.isfile() and info.name.endswith(".py") and 100 <= info.size <= 100_000:
                    stream = reader.extractfile(info)
                    if stream is not None:
                        yield info.name, stream.read()


def interpreter(version):
    # Only use interpreters already installed; never download software.
    import shutil
    if version in ("3.6", "3.7"):
        patch = {"3.6": "3.6.15", "3.7": "3.7.17"}[version]
        path = Path.home() / ".pyenv/versions" / patch / "bin/python"
        if path.is_file():
            return str(path)
    path = shutil.which(f"python{version}")
    if not path:
        raise RuntimeError(f"Install Python {version} before sampling")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-files", type=int, default=64)
    parser.add_argument("--validation-files", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--versions", nargs="+", default=list(VERSIONS), help="Explicit target versions; default excludes 3.15. Adapter data may target 3.15 separately.")
    args = parser.parse_args()
    versions = tuple(args.versions)
    if not versions or len(set(versions)) != len(versions) or any(v not in tuple(f"3.{minor}" for minor in range(6, 16)) for v in versions):
        raise ValueError("Versions must be unique supported versions from 3.6 through 3.15")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    rng = random.Random(args.seed)
    interpreters = {v: interpreter(v) for v in versions}
    # A script file works under every target version; no package import/execution.
    compiler = args.output / "compile_source.py"
    compiler.write_text("import py_compile,sys\npy_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True,optimize=0)\n")
    archives = sorted(p for directory in args.archives.iterdir() if directory.is_dir() for p in directory.iterdir() if p.is_file())
    rng.shuffle(archives)
    counts = {v: Counter() for v in versions}
    wanted = {"train": args.train_files, "validation": args.validation_files}
    seen = set()
    failures = Counter()
    files = []
    handles = {(task, split): (args.output / f"{task}-{split}.jsonl").open("w") for task in ("statement", "segmentation") for split in wanted}
    try:
        for archive in archives:
            if all(counts[v][split] == wanted[split] for v in versions for split in wanted):
                break
            try:
                members = list(source_members(archive))
            except (OSError, tarfile.TarError, zipfile.BadZipFile, ValueError) as exc:
                failures[type(exc).__name__] += 1
                continue
            rng.shuffle(members)
            for member, raw in members:
                digest = hashlib.sha512(raw).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                try:
                    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
                    text = raw.decode(encoding)
                    # Split by canonical syntax, too: formatting duplicates cannot cross splits.
                    canonical = normalize_source(text, version=(3, 14), replace_docstrings=True)
                    canonical_hash = hashlib.sha256(canonical.encode()).hexdigest()
                except Exception as exc:
                    failures[type(exc).__name__] += 1
                    continue
                split = "validation" if int(canonical_hash[:8], 16) % 5 == 0 else "train"
                for version in versions:
                    if counts[version][split] >= wanted[split]:
                        continue
                    minor = int(version.split(".")[1])
                    try:
                        normalized = normalize_source(text, version=(3, minor), replace_docstrings=True)
                        if minor <= 7:
                            normalized = add_dummy_decorators(normalized)
                        directory = args.output / "files" / version / split / digest
                        directory.mkdir(parents=True, exist_ok=True)
                        py = directory / "source.py"
                        pyc = directory / "source.pyc"
                        py.write_text(normalized)
                        result = subprocess.run([interpreters[version], str(compiler), str(py), str(pyc)], capture_output=True, timeout=20)
                        if result.returncode:
                            failures[f"compile-{version}"] += 1
                            continue
                        rows = bytecode2csv_exception_wrapper((py, pyc))
                        if isinstance(rows, Exception):
                            failures[f"rows-{version}:{type(rows).__name__}"] += 1
                            continue
                        segmentation, statements = rows
                        if not statements or not segmentation:
                            failures[f"empty-{version}"] += 1
                            continue
                        provenance = {"version": version, "source_sha512": digest, "canonical_sha256": canonical_hash, "file": str(py), "archive": str(archive), "member": member}
                        # Cap within-file contribution; retain equal *file* counts, not fake row balancing.
                        rng.shuffle(statements)
                        rng.shuffle(segmentation)
                        for row in statements[:64]:
                            handles[("statement", split)].write(json.dumps({**provenance, "source": row[0], "bytecode": row[1]}) + "\n")
                        for row in segmentation[:8]:
                            handles[("segmentation", split)].write(json.dumps({**provenance, "source": row[0], "bytecode": row[1], "boundary": row[2]}) + "\n")
                        files.append({**provenance, "split": split, "statement_rows": min(len(statements), 64), "segmentation_rows": min(len(segmentation), 8)})
                        counts[version][split] += 1
                    except Exception as exc:
                        failures[f"{version}:{type(exc).__name__}"] += 1
                if len(files) % 36 == 0:
                    print(json.dumps({"files": len(files), "counts": counts}), flush=True)
                if all(counts[v][s] == wanted[s] for v in versions for s in wanted):
                    break
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {"seed": args.seed, "archives": str(args.archives.resolve()), "versions": versions, "excluded_versions": [v for v in ("3.15",) if v not in versions], "requested_files_per_version": wanted, "counts": counts, "failures": failures, "files": files}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": counts, "failures": failures}), flush=True)
    if any(counts[v][s] != wanted[s] for v in versions for s in wanted):
        raise RuntimeError("Insufficient successful files for exact per-version quotas")


if __name__ == "__main__":
    main()
