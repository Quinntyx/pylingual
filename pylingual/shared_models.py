"""Load each immutable local model family once per worker process."""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

import torch
import transformers

from pylingual.lora import AdapterBank, VersionedPipeline

_registry = {}
_registry_lock = RLock()


def load_family(directory: Path, task: str, device):
    directory = Path(directory).resolve()
    key = (str(directory), task, str(device))
    with _registry_lock:
        if key in _registry:
            return _registry[key]
        metadata = json.loads((directory / "family.json").read_text())
        if metadata.get("format") != 1 or not metadata.get("complete") or metadata.get("task") != task:
            raise ValueError(f"Invalid or incomplete {task} model family: {directory}")
        if task == "statement":
            model = transformers.T5ForConditionalGeneration.from_pretrained(directory / "backbone")
            tokenizer = transformers.AutoTokenizer.from_pretrained(directory / "tokenizer")
        elif task == "segmentation":
            model = transformers.AutoModelForTokenClassification.from_pretrained(directory / "backbone")
            tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(directory / "tokenizer")
        else:
            raise ValueError(f"Unknown task {task}")
        model.to(device)
        model.eval()
        bank = AdapterBank(model, metadata["backbone_id"], tuple(metadata["targets"]))
        pipeline = None
        if task == "segmentation":
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
        return VersionedPipeline(pipeline, bank, version)
    return bank, tokenizer, metadata
