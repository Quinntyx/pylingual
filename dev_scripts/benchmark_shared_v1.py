"""Benchmark the shared v1 restart on the official eval releases, Python 3.14 and 3.15 files only.

Models come only from the pinned Hub revisions in the plan's results: the universal statement
backbone with both statement LoRAs, and the base-only universal segmenter. The two arms differ only
in whether the statement LoRA is enabled. Scoring follows the October benchmark
(SHARED_FULL_TRAINING_BENCHMARK_20261022.md): strict full-file bytecode equivalence, FP32, a
300-second limit per file, every listed file kept, timeouts and errors counted as failures.

Subcommands:
  identities  (local) pooled training-source identities, for the overlap stratum
  prepare     (host) build both model families from the Hub and check out the eval releases
  unit        (host) one release/arm/version; resumes from its per-file journal
  report      collect finished units into one report with paired LoRA comparisons
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from dev_scripts.benchmark_full_retrain import pin, sha, stratified_summary
from dev_scripts.hub_checkpoints import git_commit

REPO = Path(__file__).resolve().parents[1]
EVAL_REPO = "https://github.com/syssec-utd/pylingual_eval_dataset"
RELEASES = {"v1": ("v1.1.0", "a60d3b95748283ba0fbcd5e94da307d5bd5df870"),
            "v2": ("v2.0.1", "258bc0380f7c8550fa28f294b896cf168a5e4055")}
VERSIONS = ("3.14", "3.15")
ARMS = ("no_adapter", "statement_adapter")
TRAINING_VERSIONS = [f"3.{minor}" for minor in range(6, 15)]
IDENTITY_KEYS = ("source_sha512", "source_id", "payload_sha512")
# Must agree between every adapter and the backbone they are merged onto.
FAMILY_IDENTITY = ("backbone_id", "backbone_files_sha256", "tokenizer_sha256", "opcode_registry_sha256")


def identities(journals, output):
    found = set()
    for path in journals:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    found.update(row[key] for key in IDENTITY_KEYS if row.get(key))
    record = {"journals": {str(path): sha(path) for path in journals}, "keys": list(IDENTITY_KEYS),
              "identities": sorted(found)}
    output.write_text(json.dumps(record)+"\n")
    print(json.dumps({"identities": len(found), "output": str(output)}))


def download(root, source, name, **patterns):
    from huggingface_hub import snapshot_download
    local = root/"hub"/name
    snapshot_download(source["repo_id"], revision=source["revision"], local_dir=local, **patterns)
    return local


def statement_family(root, plan, destination):
    from dev_scripts.publish_shared_backbone import publish
    stages = plan["results"]["stages"]
    tokenizers = download(root, plan["tokenizer"], "tokenizer", allow_patterns=["statement/*", "segmentation/*"])
    final = download(root, stages["statement"]["hub"], "statement",
                     ignore_patterns=["last-checkpoint/*", "run/*", "README.md", ".gitattributes"])
    pending = destination.with_name(destination.name+".tmp")
    if pending.exists():
        shutil.rmtree(pending)
    publish(final, tokenizers/"statement", "statement", pending, training_versions=TRAINING_VERSIONS)
    metadata = json.loads((pending/"family.json").read_text())
    metadata["hub"] = {"statement": stages["statement"]["hub"], "tokenizer": plan["tokenizer"]}
    for version in VERSIONS:
        source = stages["lora-"+version]["hub"]
        lora = download(root, source, "lora-"+version, allow_patterns=[f"adapters/{version}/*", "family.json"])
        adapter = json.loads((lora/"family.json").read_text())
        mismatched = [key for key in FAMILY_IDENTITY if adapter[key] != metadata[key]]
        if mismatched:
            raise ValueError(f"{source['repo_id']} was trained on a different backbone: {mismatched}")
        shutil.copytree(lora/"adapters"/version, pending/"adapters"/version)
        metadata["versions"] = sorted(set(metadata["versions"]) | {version}, key=lambda text: tuple(map(int, text.split("."))))
        for key in ("adapters", "opcode_bindings", "adapter_updates"):
            metadata.setdefault(key, {})[version] = adapter[key][version]
        metadata["hub"]["lora-"+version] = source
    (pending/"family.json").write_text(json.dumps(metadata, indent=2)+"\n")
    pending.replace(destination)


def segmentation_family(root, plan, destination):
    from dev_scripts.publish_shared_backbone import publish
    source = plan["results"]["stages"]["segmentation"]["hub"]
    tokenizers = download(root, plan["tokenizer"], "tokenizer", allow_patterns=["statement/*", "segmentation/*"])
    final = download(root, source, "segmentation", ignore_patterns=["last-checkpoint/*", "run/*", "README.md", ".gitattributes"])
    pending = destination.with_name(destination.name+".tmp")
    if pending.exists():
        shutil.rmtree(pending)
    publish(final, tokenizers/"segmentation", "segmentation", pending, training_versions=TRAINING_VERSIONS)
    metadata = json.loads((pending/"family.json").read_text())
    metadata["hub"] = {"segmentation": source, "tokenizer": plan["tokenizer"]}
    (pending/"family.json").write_text(json.dumps(metadata, indent=2)+"\n")
    pending.replace(destination)


def checkout(root, release):
    tag, commit = RELEASES[release]
    destination = root/"eval"/tag
    if not destination.exists():
        pending = destination.with_name(tag+".tmp")
        if pending.exists():
            shutil.rmtree(pending)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", tag, "--filter=blob:none", "--sparse",
                        EVAL_REPO, str(pending)], check=True)
        subprocess.run(["git", "-C", str(pending), "sparse-checkout", "set", *("python-"+version for version in VERSIONS)], check=True)
        pending.replace(destination)
    head = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != commit:
        raise ValueError(f"{destination} is at {head}, expected {tag} = {commit}")
    return destination


def prepare(args, plan):
    root = args.root
    families = root/"families"
    families.mkdir(parents=True, exist_ok=True)
    if not (families/"statement").exists():
        statement_family(root, plan, families/"statement")
    if not (families/"segmentation").exists():
        segmentation_family(root, plan, families/"segmentation")
    for release in RELEASES:
        checkout(root, release)
    print(json.dumps({"prepared": str(root), "families": {task: sha(families/task/"family.json") for task in ("statement", "segmentation")}}), flush=True)


def selection(root, release, version):
    tag, commit = RELEASES[release]
    dataset = checkout(root, release)
    listing = dataset/(version.replace(".", "")+"-pyc-list.txt")
    pooled = root/"training-source-identities.json"
    excluded = set(json.loads(pooled.read_text())["identities"])
    paths = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate benchmark entries")
    files = []
    for relative in paths:
        file = (dataset/relative).resolve()
        if not file.is_relative_to(dataset.resolve()) or not file.is_file():
            raise ValueError("Missing or unsafe benchmark input: "+relative)
        files.append({"version": version, "relative": relative, "source_sha512": file.parent.name, "pyc_sha256": sha(file),
                      "known_training_or_validation_source_overlap": file.parent.name in excluded})
    return {"dataset": release, "release": tag, "commit": commit, "versions": [version], "files_per_version": len(files),
            "listing_sha256": sha(listing), "training_source_identities_sha256": sha(pooled),
            "source_hash_overlap": sum(row["known_training_or_validation_source_overlap"] for row in files),
            "scoring_policy": "all official files; the non-overlap stratum is reported separately", "files": files}


def runtime(root, arm):
    families = root/"families"
    return {"SHARED_BACKBONES": {"STATEMENT_MODEL": str(families/"statement"), "SEGMENTATION_MODEL": str(families/"segmentation"),
                                 "USE_ADAPTERS": arm == "statement_adapter", "USE_STATEMENT_ADAPTERS": arm == "statement_adapter",
                                 "USE_SEGMENTATION_ADAPTERS": False}}


def unit(args, plan):
    import torch
    from dev_scripts.evaluate_shared import file_metrics
    torch.set_num_threads(4)
    output = args.root/("smoke" if args.limit else "results")/args.release/args.arm/args.version
    output.mkdir(parents=True, exist_ok=True)
    selected = selection(args.root, args.release, args.version)
    if args.limit:
        selected["files"] = selected["files"][:args.limit]
        selected["smoke_limit"] = args.limit
    pin(output/"selection.json", selected)
    config = output/"runtime.json"
    pin(config, runtime(args.root, args.arm))
    families = args.root/"families"
    runtime_sources = sorted((REPO/"pylingual").rglob("*.py"))+[REPO/"dev_scripts"/name for name in ("benchmark_shared_v1.py", "evaluate_shared.py", "benchmark_full_retrain.py")]
    pin(output/"provenance.json", {
        "release": args.release, "arm": args.arm, "version": args.version, "file_timeout_seconds": args.timeout,
        "precision": "FP32 inference", "git_commit": git_commit(), "hub": plan["results"]["stages"],
        "families": {task: sha(families/task/"family.json") for task in ("statement", "segmentation")},
        "compilers": json.loads(args.compilers.read_text())["compilers"],
        "runtime_sha256": {str(path.relative_to(REPO)): sha(path) for path in runtime_sources}})
    settings = SimpleNamespace(output=output, config=config, versions=[args.version], dataset=checkout(args.root, args.release),
                               timeout=args.timeout, resume=True)
    if not (output/"complete.json").exists():
        file_metrics(settings, selected)
        rows = [json.loads(line) for line in (output/"file_results.jsonl").open()]
        if len(rows) != len(selected["files"]) or len({row["relative"] for row in rows}) != len(rows):
            raise RuntimeError("Incomplete full-file benchmark")
        if any(row.get("phase") == "model_load" for row in rows):
            raise RuntimeError("A model load failure invalidates this unit")
        (output/"summary.json").write_text(json.dumps(stratified_summary(rows, args.version), indent=2)+"\n")
        (output/"complete.json").write_text(json.dumps({"complete": True, "files": len(rows)})+"\n")
    print(json.dumps({"complete": True, "release": args.release, "arm": args.arm, "version": args.version,
                      "summary": json.loads((output/"summary.json").read_text())[args.version]}), flush=True)


def paired(base, adapted):
    before = {row["relative"]: bool(row["success"]) for row in base}
    after = {row["relative"]: bool(row["success"]) for row in adapted}
    if set(before) != set(after):
        raise ValueError("Arms scored different files")
    gained = sum(after[name] and not before[name] for name in before)
    lost = sum(before[name] and not after[name] for name in before)
    return {"files": len(before), "newly_correct": gained, "regressed": lost, "net": gained-lost,
            "gain_pp": 100*(gained-lost)/len(before)}


def report(args):
    results = args.root/"results"
    document = {"releases": {}}
    lines = ["| Release | Python | Arm | Exact files | % | Non-overlap exact | Non-overlap % |", "|---|---|---|---:|---:|---:|---:|"]
    for release, (tag, _) in RELEASES.items():
        document["releases"][release] = {"tag": tag, "versions": {}}
        for version in VERSIONS:
            entry = {}
            for arm in ARMS:
                output = results/release/arm/version
                if not (output/"complete.json").exists():
                    continue
                summary = json.loads((output/"summary.json").read_text())[version]
                held_out = summary["non_overlapping_source_files"]
                entry[arm] = summary
                held_out_percent = f"{100*held_out['file_accuracy']:.2f}" if held_out["files"] else "n/a"
                lines.append(f"| {release} | {version} | {arm} | {summary['equal_files']}/{summary['files']} | {100*summary['file_accuracy']:.2f} | "
                             f"{held_out['equal_files']}/{held_out['files']} | {held_out_percent} |")
            if len(entry) == len(ARMS):
                journals = [[json.loads(line) for line in (results/release/arm/version/"file_results.jsonl").open()] for arm in ARMS]
                entry["paired"] = paired(*journals)
                entry["categories"] = {arm: dict(Counter(row["category"] for row in rows)) for arm, rows in zip(ARMS, journals)}
            document["releases"][release]["versions"][version] = entry
    (results/"report.json").write_text(json.dumps(document, indent=2)+"\n")
    print("\n".join(lines))
    for release, record in document["releases"].items():
        for version, entry in record["versions"].items():
            if "paired" in entry:
                print(release, version, json.dumps(entry["paired"]), json.dumps(entry["categories"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("identities", "prepare", "compiler-port", "unit", "report"))
    parser.add_argument("--plan", type=Path, default=REPO/"dev_scripts/experiments/shared-v1-restart-2026-10-28.json")
    parser.add_argument("--root", type=Path, default=Path.home()/"pylingual-bench")
    parser.add_argument("--journals", type=Path, nargs="+", help="identities: accepted.jsonl journals of every training source")
    parser.add_argument("--output", type=Path, help="identities: output JSON")
    parser.add_argument("--compilers", type=Path, help="Metadata written by serve_benchmark_compilers")
    parser.add_argument("--limit", type=int, help="unit: smoke-check only the first N listed files (separate output)")
    parser.add_argument("--release", choices=RELEASES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--version", choices=VERSIONS)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.command == "identities":
        identities(args.journals, args.output)
    elif args.command == "prepare":
        prepare(args, plan)
    elif args.command == "compiler-port":
        print(json.loads(args.compilers.read_text())["port"])
    elif args.command == "unit":
        if not (args.release and args.arm and args.version and args.compilers):
            parser.error("unit needs --release, --arm, --version and --compilers")
        unit(args, plan)
    else:
        report(args)


if __name__ == "__main__":
    main()
