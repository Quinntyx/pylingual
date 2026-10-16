"""Balanced shared-backbone training, then frozen-backbone version LoRAs.

No Hub uploads. Sampling is version -> file -> example, so large files cannot
silently dominate the universal objective. PyLingual keeps its two distinct
neural tasks: one universal segmenter and one universal statement translator.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import time
import uuid
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerFast, T5ForConditionalGeneration
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from pylingual.lora import AdapterBank

VERSIONS = tuple(f"3.{minor}" for minor in range(6, 15))
LABELS = {"B": 0, "I": 1, "E": 2}


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def source_prefix(version):
    return f"Python {version}: "


def load_seg_tokenizer(repo):
    path = Path(repo)
    if path.is_dir():
        file = path / "tokenizer.json"
    else:
        file = Path(hf_hub_download(repo, "tokenizer.json"))
    return PreTrainedTokenizerFast(tokenizer_file=str(file), unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def extend_seg_vocabulary(tokenizer, rows):
    # Extend the underlying WordPiece vocabulary, NOT AddedTokens (which change
    # word_ids and break instruction/boundary alignment).
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    if backend["model"]["type"] != "WordPiece":
        raise ValueError("Expected PyLingual WordPiece segmentation tokenizer")
    vocab = backend["model"]["vocab"]
    instructions = {instruction for row in rows for instruction in row["bytecode"].split(" <SEP> ")}
    unknown = sorted(instruction for instruction in instructions if tokenizer.unk_token_id in tokenizer(instruction)["input_ids"])
    for instruction in unknown:
        if instruction not in vocab:
            vocab[instruction] = len(vocab)
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(backend)), unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def encode_row(task, tokenizer, row, max_input=512, max_output=128):
    if task == "statement":
        inputs = tokenizer(source_prefix(row["version"]) + row["bytecode"], truncation=False)
        labels = tokenizer(row["source"], truncation=False)["input_ids"]
        if len(inputs["input_ids"]) > max_input or len(labels) > max_output:
            return None
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels}
    boundaries = row["boundary"]
    if isinstance(boundaries, str):
        boundaries = ast.literal_eval(boundaries)
    encoded = tokenizer(row["bytecode"], truncation=True, max_length=max_input)
    word_ids = encoded.word_ids()
    if any(i is not None and i >= len(boundaries) for i in word_ids):
        raise ValueError("Segmentation tokenizer word IDs do not align to instructions")
    labels = [-100 if i is None else LABELS[boundaries[i]] for i in word_ids]
    if not any(label != -100 for label in labels):
        return None
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels}


def collate(rows, tokenizer, device):
    max_input = max(len(r["input_ids"]) for r in rows)
    max_labels = max(len(r["labels"]) for r in rows)
    return {
        "input_ids": torch.tensor([r["input_ids"] + [tokenizer.pad_token_id] * (max_input - len(r["input_ids"])) for r in rows], device=device),
        "attention_mask": torch.tensor([r["attention_mask"] + [0] * (max_input - len(r["attention_mask"])) for r in rows], device=device),
        "labels": torch.tensor([r["labels"] + [-100] * (max_labels - len(r["labels"])) for r in rows], device=device),
    }


def grouped_examples(task, tokenizer, rows):
    grouped = defaultdict(lambda: defaultdict(list))
    dropped = defaultdict(int)
    for row in rows:
        encoded = encode_row(task, tokenizer, row)
        if encoded is None:
            dropped[row["version"]] += 1
        else:
            grouped[row["version"]][row["file"]].append(encoded)
    if set(grouped) != set(VERSIONS):
        raise ValueError(f"Missing training versions: {set(VERSIONS) - set(grouped)}")
    return {v: list(files.values()) for v, files in grouped.items()}, dict(dropped)


def train_steps(model, tokenizer, grouped, versions, steps, batch_size, accumulation, lr, seed, parameters):
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    model.train()
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    started = time.monotonic()
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for micro in range(accumulation):
            rows = []
            for index in range(batch_size):
                version = versions[((step * accumulation + micro) * batch_size + index) % len(versions)]
                rows.append(rng.choice(rng.choice(grouped[version])))
            batch = collate(rows, tokenizer, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = model(**batch).loss / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        # Warmup then linear decay, with an explicit final 10% floor.
        fraction = min((step + 1) / max(1, steps * 0.05), 1.0) * max(0.1, 1 - step / max(1, steps))
        for group in optimizer.param_groups:
            group["lr"] = lr * fraction
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            record = {"step": step + 1, "steps": steps, "loss": total, "seconds": round(time.monotonic() - started, 2)}
            history.append(record)
            print(json.dumps(record), flush=True)
    model.eval()
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("statement", "segmentation"), required=True)
    parser.add_argument("--statement-init", default="Salesforce/codet5-base")
    parser.add_argument("--segmentation-init", default="syssec-utd/py39-pylingual-v1-segmenter")
    parser.add_argument("--segmentation-tokenizer", default="syssec-utd/py39-pylingual-v1-tokenizer")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--adapter-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("This training entrypoint requires a CUDA GPU")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite model family {args.output}")
    args.output.mkdir(parents=True)
    rows = read_rows(args.data / f"{args.task}-train.jsonl")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if tuple(manifest["versions"]) != VERSIONS or any(r["version"] not in VERSIONS for r in rows):
        raise ValueError("Training versions must be exactly 3.6 through 3.14; 3.15 is excluded")
    quotas = [manifest["counts"][v]["train"] for v in VERSIONS]
    if len(set(quotas)) != 1:
        raise ValueError("Training file counts are not version-balanced")
    if args.task == "statement":
        tokenizer = AutoTokenizer.from_pretrained(args.statement_init)
        if args.statement_init.startswith("Salesforce/codet5"):
            tokenizer.add_tokens([f"<mask_{i}>" for i in range(300)])
        model = T5ForConditionalGeneration.from_pretrained(args.statement_init)
        model.resize_token_embeddings(len(tokenizer))
        # CodeT5 tokenizer and decoder IDs must agree, including newly saved config.
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.decoder_start_token_id = tokenizer.pad_token_id
        targets = ("q", "v")
        initial = args.statement_init
    else:
        tokenizer = extend_seg_vocabulary(load_seg_tokenizer(args.segmentation_tokenizer), rows)
        model = AutoModelForTokenClassification.from_pretrained(args.segmentation_init)
        model.resize_token_embeddings(len(tokenizer))
        # This is a warm-start pilot, not a from-scratch segmentation experiment.
        model.config.id2label = {0: "B", 1: "I", 2: "E"}
        model.config.label2id = LABELS
        targets = ("query", "value")
        initial = args.segmentation_init
    model.to("cuda")
    model.config.use_cache = False
    grouped, dropped = grouped_examples(args.task, tokenizer, rows)
    backbone_id = str(uuid.uuid4())
    family = {"format": 1, "task": args.task, "backbone_id": backbone_id, "versions": VERSIONS, "targets": targets, "source_prefix": "Python {version}: " if args.task == "statement" else "", "initial_model": initial, "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "dropped_long_training_rows": dropped, "encoded_files_per_version": {v: len(files) for v, files in grouped.items()}, "encoded_rows_per_version": {v: sum(map(len, files)) for v, files in grouped.items()}}
    parameters = list(model.parameters())
    family["backbone_parameters"] = sum(p.numel() for p in parameters)
    print(json.dumps({"task": args.task, "parameters": family["backbone_parameters"], "encoded_rows": family["encoded_rows_per_version"]}), flush=True)
    family["backbone_training"] = train_steps(model, tokenizer, grouped, VERSIONS, args.steps, args.batch_size, args.accumulation, 5e-5, args.seed, parameters)
    # Save the unwrapped backbone once. Never merge any version's LoRA into it.
    model.config.use_cache = True
    model.save_pretrained(args.output / "backbone")
    tokenizer.save_pretrained(args.output / "tokenizer")
    del parameters
    bank = AdapterBank(model, backbone_id, targets)
    family["adapters"] = {}
    family["adapter_training"] = {}
    for index, version in enumerate(VERSIONS):
        print(f"Training {args.task} adapter for Python {version}", flush=True)
        bank.add(version, rank=args.rank, alpha=2 * args.rank, dropout=0.05)
        params = bank.trainable(version)
        model.config.use_cache = False
        family["adapter_training"][version] = train_steps(model, tokenizer, grouped, (version,), args.adapter_steps, args.batch_size, args.accumulation, 3e-4, args.seed + index, params)
        directory = args.output / "adapters" / version
        bank.save(version, directory)
        family["adapters"][version] = f"adapters/{version}"
        family["adapter_parameters_per_version"] = sum(p.numel() for p in params)
        # Persist after each version for diagnosis, but runtime only accepts complete families.
        (args.output / "family.json").write_text(json.dumps(family, indent=2) + "\n")
    family["complete"] = True
    family["peak_allocated_cuda_bytes"] = torch.cuda.max_memory_allocated()
    (args.output / "family.json").write_text(json.dumps(family, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output), "peak_cuda_bytes": family["peak_allocated_cuda_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
