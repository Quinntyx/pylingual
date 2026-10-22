"""Load each immutable local model family once per worker process."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from threading import RLock

import torch
import transformers

from pylingual.lora import AdapterBank, VersionedPipeline
from pylingual.opcodes import OpcodeRegistry, tokenizer_fingerprint
from pylingual.opcode_segmenter import OpcodeSegmenter

_registry = {}
_registry_lock = RLock()


def _verify_backbone(directory, metadata):
    expected = metadata.get("backbone_files_sha256")
    if not expected:
        if metadata.get("format") == 2 and not metadata.get("smoke_only"):
            raise ValueError("Production opcode family requires immutable backbone file fingerprints")
        return
    if not isinstance(expected, dict) or any(not name or Path(name).name != name for name in expected):
        raise ValueError("Invalid backbone fingerprint paths")
    root = directory / "backbone"
    if set(expected) != {path.name for path in root.iterdir() if path.is_file()}:
        raise ValueError("Backbone file inventory differs from its manifest")
    for name, digest in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Backbone path escapes its family")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Backbone file checksum mismatch")
    identity = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if identity != metadata.get("backbone_id"):
        raise ValueError("Backbone content identity mismatch")


def load_family(directory: Path, task: str, device):
    directory = Path(directory).resolve()
    key = (str(directory), task, str(device))
    with _registry_lock:
        if key in _registry:
            return _registry[key]
        metadata = json.loads((directory / "family.json").read_text())
        if metadata.get("format") not in (1, 2) or not metadata.get("complete") or metadata.get("task") != task:
            raise ValueError(f"Invalid or incomplete {task} model family: {directory}")
        _verify_backbone(directory, metadata)
        if task == "statement":
            model = transformers.T5ForConditionalGeneration.from_pretrained(directory / "backbone")
            tokenizer = transformers.AutoTokenizer.from_pretrained(directory / "tokenizer")
        elif task == "segmentation":
            model = transformers.AutoModelForTokenClassification.from_pretrained(directory / "backbone")
            tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(directory / "tokenizer")
        else:
            raise ValueError(f"Unknown task {task}")
        registry = None
        if metadata["format"] == 2:
            registry = OpcodeRegistry.load(directory / "tokenizer" / "opcode_registry.json", tokenizer)
            if registry.fingerprint != metadata.get("opcode_registry_sha256") or tokenizer_fingerprint(tokenizer) != metadata.get("tokenizer_sha256"):
                raise ValueError("Model family opcode registry/tokenizer fingerprint mismatch")
        model.to(device)
        model.eval()
        bank = AdapterBank(model, metadata["backbone_id"], tuple(metadata["targets"]),
                           opcode_registry=registry, tokenizer=tokenizer, task=task)
        for version, bindings in metadata.get("opcode_bindings", {}).items():
            bank.register_bindings(version, bindings)
        pipeline = None
        if task == "segmentation" and registry is not None:
            pipeline = OpcodeSegmenter(bank, tokenizer)
        elif task == "segmentation":
            pipeline = transformers.pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="none", device=device)
        result = (bank, tokenizer, metadata, pipeline)
        _registry[key] = result
        return result


def load_version(directory: Path, task: str, version: str, device):
    bank, tokenizer, metadata, pipeline = load_family(directory, task, device)
    if version not in metadata["versions"] or version not in metadata["adapters"]:
        raise ValueError(f"No {task} adapter for Python {version}; this family supports {metadata['versions']}")
    with bank.lock:
        if version not in bank.specs:
            relative = Path(metadata["adapters"][version])
            root = Path(directory).resolve()
            adapter_path = (root / relative).resolve()
            if not adapter_path.is_relative_to(root):
                raise ValueError("Adapter path escapes the model family")
            bank.load(version, adapter_path)
    if task == "segmentation":
        return pipeline.for_version(version) if isinstance(pipeline, OpcodeSegmenter) else VersionedPipeline(pipeline, bank, version)
    return bank, tokenizer, metadata
