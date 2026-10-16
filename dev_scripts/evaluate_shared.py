"""Reproducible file-level v2 benchmark, plus shared-family held-out metrics.

No scraped training data or model weights are committed. Dataset and family paths
must be supplied explicitly. Run from the repository root with python -m.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import time
import traceback
from collections import Counter
from pathlib import Path

import torch

from pylingual.decompiler import decompile
from pylingual.models import load_models
from pylingual.utils.version import PythonVersion
from pylingual.shared_models import load_family
from dev_scripts.shared_training import VERSIONS, collate, encode_row, read_rows


def expired(signum, frame):
    raise TimeoutError("File-level evaluation wall-clock limit exceeded")


def select_files(args):
    training = json.loads((args.data / "manifest.json").read_text())
    excluded = {row["source_sha512"] for row in training["files"]}
    if args.adapter_data:
        adapter_manifest = json.loads((args.adapter_data / "manifest.json").read_text())
        excluded.update(row["source_sha512"] for row in adapter_manifest["files"])
    rng = random.Random(args.seed)
    selected = []
    for version in args.versions:
        listing = args.dataset / "pylingualv2" / f"{version.replace('.', '')}-pyc-list.txt"
        candidates = list(dict.fromkeys(line.strip() for line in listing.read_text().splitlines() if line.strip() and Path(line.strip()).parent.name not in excluded))
        if len(candidates) < args.files_per_version:
            raise ValueError(f"Insufficient non-training files in {listing}")
        for relative in rng.sample(candidates, args.files_per_version):
            file = args.dataset / relative
            if not file.is_file():
                raise FileNotFoundError(file)
            selected.append({"version": version, "relative": relative, "source_sha512": file.parent.name, "pyc_sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    return {"seed": args.seed, "files_per_version": args.files_per_version, "versions": args.versions, "backbone_training_versions": list(VERSIONS), "adapter_data_manifest_sha256": hashlib.sha256((args.adapter_data / "manifest.json").read_bytes()).hexdigest() if args.adapter_data else None, "source_overlap_with_training": 0, "training_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(), "files": selected}


def component_metrics(args):
    output = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for task in ("statement", "segmentation"):
        bank, tokenizer, metadata, _ = load_family(args.families / task, task, device)
        rows = read_rows(args.data / f"{task}-validation.jsonl")
        if args.adapter_data:
            rows.extend(read_rows(args.adapter_data / f"{task}-validation.jsonl"))
        per_version = {}
        for version in args.versions:
            if not args.disable_adapters and version not in bank.specs:
                bank.load(version, args.families / task / metadata["adapters"][version])
            candidates = [row for row in rows if row["version"] == version]
            rng = random.Random(args.seed)
            rng.shuffle(candidates)
            examples = []
            for row in candidates:
                encoded = encode_row(task, tokenizer, row)
                if encoded is not None:
                    examples.append((row, encoded))
                if len(examples) == args.component_rows:
                    break
            if not examples:
                raise ValueError(f"No usable validation examples for {task} {version}")
            correct = total = exact = 0
            boundary_tp = boundary_fp = boundary_fn = 0
            with bank.use(None if args.disable_adapters else version), torch.inference_mode():
                for start in range(0, len(examples), 8):
                    chunk = examples[start:start + 8]
                    batch = collate([encoded for _, encoded in chunk], tokenizer, device)
                    if task == "statement":
                        generated = bank.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=128)
                        for tokens, (row, _) in zip(generated, chunk):
                            decoded = tokenizer.decode([i for i in tokens.tolist() if i not in (0, 1, 2, 3, 4)], skip_special_tokens=False)
                            exact += decoded.strip() == row["source"].strip()
                            total += 1
                    else:
                        predictions = bank.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.argmax(-1)
                        labels = batch["labels"]
                        valid = labels != -100
                        correct += ((predictions == labels) & valid).sum().item()
                        total += valid.sum().item()
                        boundary_tp += ((predictions == 0) & (labels == 0) & valid).sum().item()
                        boundary_fp += ((predictions == 0) & (labels != 0) & valid).sum().item()
                        boundary_fn += ((predictions != 0) & (labels == 0) & valid).sum().item()
            result = {"examples": len(examples)}
            if task == "statement":
                result.update(exact=exact, exact_match=exact / total)
            else:
                result.update(tokens=total, correct=correct, token_accuracy=correct / total, boundary_tp=boundary_tp, boundary_fp=boundary_fp, boundary_fn=boundary_fn, boundary_f1=2 * boundary_tp / max(1, 2 * boundary_tp + boundary_fp + boundary_fn))
            per_version[version] = result
            print(json.dumps({"component": task, "version": version, **result}), flush=True)
        output[task] = per_version
    (args.output / "component_metrics.json").write_text(json.dumps(output, indent=2) + "\n")


def file_metrics(args, selection):
    rows = []
    model_stats = {}
    config = args.config or Path(__file__).resolve().parents[1] / "pylingual" / "decompiler_config.yaml"
    old_handler = signal.signal(signal.SIGALRM, expired)
    try:
        for item in selection["files"]:
            started = time.monotonic()
            result = {**item, "success": False, "code_objects": 0, "equal_code_objects": 0, "phase": "model_load"}
            try:
                if item["version"] not in model_stats:
                    segmenter, translator = load_models(config_file=config, version=PythonVersion(item["version"]))
                    # Loading/download time is not an inference timeout: production
                    # workers keep models resident. Count actual live parameters.
                    models = {"statement": translator.model, "segmentation": segmenter.model}
                    model_stats[item["version"]] = {task: {"parameters": sum(p.numel() for p in model.parameters()), "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()), "class": type(model).__name__} for task, model in models.items()}
                    model_stats[item["version"]]["allocated_cuda_bytes"] = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    (args.output / "model_stats.json").write_text(json.dumps(model_stats, indent=2) + "\n")
                    del models, segmenter, translator
                result["phase"] = "decompilation"
                inference_started = time.monotonic()
                signal.alarm(args.timeout)
                destination = args.output / "decompiled" / item["version"] / (item["source_sha512"] + ".py")
                destination.parent.mkdir(parents=True, exist_ok=True)
                answer = decompile(args.dataset / item["relative"], destination, config_file=config, timeout=args.timeout)
                checks = answer.equivalence_results
                object_checks = [check for check in checks if getattr(check, "bc_a", None) is not None]
                result.update(success=bool(checks) and all(check.success for check in checks), verification_checks=len(checks), code_objects=len(object_checks), equal_code_objects=sum(check.success for check in object_checks))
                result["verification_failures"] = dict(Counter(getattr(check, "message", str(check))[:400] for check in checks if not check.success))
                result["category"] = "Equal" if result["success"] else "CompileError" if any(type(check).__name__ == "CompileError" for check in checks) else "Different" if checks else "NoVerification"
                if time.monotonic() - inference_started >= args.timeout:
                    result.update(success=False, category="Timeout")
            except Exception as exc:
                result.update(category="Timeout" if isinstance(exc, TimeoutError) else "Error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            finally:
                signal.alarm(0)
            result["seconds"] = round(time.monotonic() - started, 3)
            if result["phase"] == "decompilation":
                result["inference_seconds"] = round(time.monotonic() - inference_started, 3)
            rows.append(result)
            with (args.output / "file_results.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps({"file": len(rows), "of": len(selection["files"]), **{key: value for key, value in result.items() if key != "traceback"}}), flush=True)
    finally:
        signal.signal(signal.SIGALRM, old_handler)
    summary = {}
    for version in args.versions:
        subset = [row for row in rows if row["version"] == version]
        succeeded = sum(row["success"] for row in subset)
        summary[version] = {"files": len(subset), "equal_files": succeeded, "file_accuracy": succeeded / len(subset), "code_objects": sum(row["code_objects"] for row in subset), "equal_code_objects": sum(row["equal_code_objects"] for row in subset), "categories": dict(Counter(row["category"] for row in subset)), "seconds": round(sum(row["seconds"] for row in subset), 3)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Local pylingual_eval_dataset checkout")
    parser.add_argument("--data", type=Path, required=True, help="Sampled training dataset directory")
    parser.add_argument("--versions", nargs="+", default=["3.14", "3.15"])
    parser.add_argument("--adapter-data", type=Path, help="Held-out backbone version's adapter-only dataset directory")
    parser.add_argument("--disable-adapters", action="store_true", help="Measure component metrics without LoRAs; runtime YAML must also set USE_ADAPTERS: false")
    parser.add_argument("--families", type=Path, help="Local directory containing statement/ and segmentation/")
    parser.add_argument("--config", type=Path, help="Decompiler configuration; omit for existing specialists")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="Previously saved selection manifest for identical baseline files")
    parser.add_argument("--files-per-version", type=int, default=8)
    parser.add_argument("--component-rows", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--components-only", action="store_true")
    args = parser.parse_args()
    if min(args.files_per_version, args.component_rows, args.timeout) < 1:
        parser.error("File counts, component row counts and timeout must be positive")
    if args.components_only and not args.families:
        parser.error("--components-only requires --families")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    selection = json.loads(args.selection.read_text()) if args.selection else select_files(args)
    if not args.versions or len(set(args.versions)) != len(args.versions) or {row["version"] for row in selection["files"]} != set(args.versions):
        raise ValueError("Evaluation selection must cover exactly the requested versions")
    expected_counts = Counter({version: args.files_per_version for version in args.versions})
    if Counter(row["version"] for row in selection["files"]) != expected_counts:
        raise ValueError("Evaluation selection must have equal requested file counts per version")
    if len({row["relative"] for row in selection["files"]}) != len(selection["files"]):
        raise ValueError("Evaluation selection contains duplicate bytecode paths")
    excluded = {row["source_sha512"] for row in json.loads((args.data / "manifest.json").read_text())["files"]}
    if args.adapter_data:
        excluded.update(row["source_sha512"] for row in json.loads((args.adapter_data / "manifest.json").read_text())["files"])
    if any(row["source_sha512"] in excluded for row in selection["files"]):
        raise ValueError("Evaluation selection overlaps training/adapter source identifiers")
    for row in selection["files"]:
        path = (args.dataset / row["relative"]).resolve()
        if not path.is_relative_to(args.dataset.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row["pyc_sha256"]:
            raise ValueError("Evaluation path escapes dataset or bytecode hash changed")
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    configuration = args.config or Path(__file__).resolve().parents[1] / "pylingual" / "decompiler_config.yaml"
    if args.families and not args.components_only:
        import yaml
        shared_config = yaml.safe_load(configuration.read_text()).get("SHARED_BACKBONES", {})
        if args.disable_adapters != (shared_config.get("USE_ADAPTERS", True) is False):
            raise ValueError("--disable-adapters and runtime USE_ADAPTERS must agree")
    provenance = {"arguments": vars(args), "config_sha256": hashlib.sha256(configuration.read_bytes()).hexdigest(), "training_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(), "torch_version": torch.__version__, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.families:
        provenance["families"] = {task: json.loads((args.families / task / "family.json").read_text()) for task in ("statement", "segmentation")}
    (args.output / "evaluation.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    if args.families:
        component_metrics(args)
    if not args.components_only:
        file_metrics(args, selection)


if __name__ == "__main__":
    main()
