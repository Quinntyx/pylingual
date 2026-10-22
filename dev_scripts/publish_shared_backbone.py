"""Package a trained, unwrapped backbone with its immutable shared opcode tokenizer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, T5ForConditionalGeneration
from pylingual.opcodes import OpcodeRegistry, tokenizer_fingerprint


def publish(checkpoint, tokenizer_directory, task, output, *, training_versions, held_out_version="3.15", smoke_only=False):
    checkpoint, tokenizer_directory, output = map(Path, (checkpoint, tokenizer_directory, output))
    if output.exists() or any(source.resolve() in output.resolve().parents for source in (checkpoint, tokenizer_directory)):
        raise ValueError("Output must be fresh and outside checkpoint/tokenizer directories")
    if not training_versions or held_out_version in training_versions:
        raise ValueError("Backbone training versions must be explicit and exclude the held-out version")
    if task not in ("statement", "segmentation"):
        raise ValueError("Unknown backbone task")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory)
    registry = OpcodeRegistry.load(tokenizer_directory / "opcode_registry.json", tokenizer)
    model_type = T5ForConditionalGeneration if task == "statement" else AutoModelForTokenClassification
    model, loading = model_type.from_pretrained(checkpoint, output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Checkpoint must contain the complete, unwrapped trained backbone")
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError("Backbone vocabulary differs from frozen tokenizer; never resize at adapter time")
    if task == "segmentation" and model.config.id2label != {0: "B", 1: "I", 2: "E"}:
        raise ValueError("Boundary classifier must use the normal B/I/E label order")
    metadata = {"format": 2, "complete": False, "task": task,
                "targets": ["q", "v"] if task == "statement" else ["query", "value"],
                "versions": sorted(set(training_versions), key=lambda version: tuple(map(int, version.split(".")))),
                "backbone_training_versions": list(training_versions), "held_out_version": held_out_version,
                "source_prefix": "", "adapters": {}, "opcode_bindings": {},
                "opcode_registry_sha256": registry.fingerprint,
                "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                "source_checkpoint": str(checkpoint.resolve()), "smoke_only": bool(smoke_only)}
    output.mkdir(parents=True)
    (output / "family.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model.save_pretrained(output / "backbone")
    tokenizer.save_pretrained(output / "tokenizer")
    (output / "tokenizer/opcode_registry.json").write_bytes((tokenizer_directory / "opcode_registry.json").read_bytes())
    # Content identity, not a user-selected label: adapters must name these exact base weights.
    files = {}
    for path in sorted((output / "backbone").iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata["backbone_files_sha256"] = files
    metadata["backbone_id"] = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata["complete"] = True
    pending = output / "family.json.tmp"
    pending.write_text(json.dumps(metadata, indent=2) + "\n")
    pending.replace(output / "family.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--task", choices=("statement", "segmentation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-versions", nargs="+", required=True)
    parser.add_argument("--held-out-version", default="3.15")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    print(publish(args.checkpoint, args.tokenizer, args.task, args.output, training_versions=args.training_versions,
                  held_out_version=args.held_out_version, smoke_only=args.smoke_only))
