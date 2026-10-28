"""Fit shared tokenizers only; never initialize or optimize neural model weights."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from dev_scripts.tokenizer_opcode_registry import RESERVED_SLOT_COUNT, RESERVED_TOKENS, bind_bytecode, instruction_token_indices, known_token


def corpus_iterator(entries, task):
    for provenance in entries:
        with gzip.open(provenance["chunk"], "rt", encoding="utf-8") as stream:
            rows = json.load(stream)[task]
        yield [row[1] for row in rows]


def verify(tokenizer, known, registry):
    tokens = [known_token(name) for name in sorted(known)] + list(RESERVED_TOKENS)
    ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate opcode IDs")
    if any(value in tokenizer.all_special_ids for value in ids):
        raise ValueError("Opcode was incorrectly marked structural/special")
    for token, token_id in zip(tokens, ids):
        if tokenizer(token, add_special_tokens=False)["input_ids"] != [token_id]:
            raise ValueError("Opcode marker is not atomic: "+token)
    reserved = [tokenizer.convert_tokens_to_ids(token) for token in RESERVED_TOKENS]
    if reserved != list(range(reserved[0], reserved[0]+RESERVED_SLOT_COUNT)):
        raise ValueError(f"Reserved IDs are not one contiguous {RESERVED_SLOT_COUNT}-slot range")
    original = "LOAD_CONST , LOAD_FAST <SEP> RETURN_VALUE"
    mapped = bind_bytecode(original, known)
    if mapped != known_token("LOAD_CONST")+" , LOAD_FAST <SEP> "+known_token("RETURN_VALUE"):
        raise ValueError("Binding changed operands")
    future = bind_bytecode("FUTURE_OPCODE_SMOKE , 1 <SEP> RETURN_VALUE", known, {"FUTURE_OPCODE_SMOKE": 0})
    encoded, indices = instruction_token_indices(tokenizer, future)
    if reserved[0] not in encoded["input_ids"] or not {0, 1}.issubset(set(indices)):
        raise ValueError("Future opcode or instruction alignment failed")
    try:
        bind_bytecode("FUTURE_OPCODE_SMOKE", known, {f"NEW_{i}": i for i in range(RESERVED_SLOT_COUNT+1)})
    except ValueError:
        pass
    else:
        raise ValueError("Over-capacity future opcode binding did not fail")
    return {"atomic_known_tokens": len(known), "atomic_reserved_tokens": RESERVED_SLOT_COUNT,
            "reserved_ids": reserved, "operand_preservation": True,
            "future_binding_and_instruction_alignment": True, "overflow_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--compiler-provenance", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=30000)
    args = parser.parse_args()
    if (args.output/"manifest.json").exists():
        raise ValueError("Output already contains a tokenizer family; use a new output")
    data_manifest = json.loads((args.data/"manifest.json").read_text())
    if not data_manifest.get("complete") or "3.15" in data_manifest["versions"]:
        raise ValueError("Incomplete or3.15-contaminated tokenizer corpus")
    entries = [json.loads(line) for line in (args.data/"accepted.jsonl").read_text().splitlines()]
    if len(entries) != data_manifest["files"]:
        raise ValueError("Corpus manifest count mismatch")
    provenance = json.loads(args.compiler_provenance.read_text())
    known = set()
    for version, meta in provenance.items():
        if version not in data_manifest["versions"]:
            raise ValueError("Nontraining version included in compiler vocabulary")
        known.update(name for name in meta["opcodes"] if name and name[0].isalpha() and name.isupper())
    observed = set().union(*(set(entry["observed_opcodes"]) for entry in entries))
    known.update(observed)
    markers = [known_token(name) for name in sorted(known)]
    fitting_vocab = args.vocab_size-len(markers)-len(RESERVED_TOKENS)
    if fitting_vocab < 1000:
        raise ValueError("Vocabulary too small for template and opcode registry")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"complete": False, "format": 1, "artifact_type": "generic-tokenizers-only",
               "model_training_started": False, "data_manifest": data_manifest,
               "data_manifest_sha256": hashlib.sha256((args.data/"manifest.json").read_bytes()).hexdigest(),
               "journal_sha256": hashlib.sha256((args.data/"accepted.jsonl").read_bytes()).hexdigest(),
               "compiler_provenance_sha256": hashlib.sha256(args.compiler_provenance.read_bytes()).hexdigest(),
               "total_vocab_budget": args.vocab_size, "fitted_base_vocab_budget": fitting_vocab,
               "known_opcode_count": len(known), "observed_opcode_count": len(observed),
               "unobserved_known_opcodes": sorted(known-observed), "reserved_slots": RESERVED_SLOT_COUNT,
               "host": platform.node(), "cpu_threads": __import__("os").cpu_count(),
               "packages": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "xdis")},
               "tasks": {}}
    total_start = time.time()
    for task in ("statement", "segmentation"):
        start = time.time()
        print(json.dumps({"fitting": task, "files": len(entries), "base_vocab_budget": fitting_vocab}), flush=True)
        if task == "statement":
            template = AutoTokenizer.from_pretrained(args.template)
            tokenizer = template.train_new_from_iterator(corpus_iterator(entries, task), vocab_size=fitting_vocab)
            tokenizer.model_max_length = 256
        else:
            backend = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            backend.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()])
            backend.pre_tokenizer = pre_tokenizers.Split("<SEP>", "removed")
            trainer = trainers.WordPieceTrainer(vocab_size=fitting_vocab, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
            backend.train_from_iterator(corpus_iterator(entries, task), trainer=trainer)
            backend.decoder = decoders.WordPiece(prefix="##")
            backend.post_processor = processors.TemplateProcessing(single="[CLS]:0 $A:0 [SEP]:0", pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
                special_tokens=[("[CLS]", backend.token_to_id("[CLS]")), ("[SEP]", backend.token_to_id("[SEP]"))])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=1024)
        base_vocab = len(tokenizer)
        tokenizer.add_tokens([AddedToken(token, normalized=False, special=False) for token in markers+list(RESERVED_TOKENS)])
        if len(tokenizer) > args.vocab_size:
            raise ValueError("Vocabulary exceeded declared total budget")
        directory = args.output/task
        directory.mkdir(exist_ok=True)
        tokenizer.save_pretrained(directory)
        registry = {"format": 1, "preprocessing": "opcode-field-placeholders-v1", "version": None,
                    "known_opcodes": {name: {"token": known_token(name), "id": tokenizer.convert_tokens_to_ids(known_token(name))} for name in sorted(known)},
                    "reserved_slots": [{"slot": i, "token": token, "id": tokenizer.convert_tokens_to_ids(token)} for i, token in enumerate(RESERVED_TOKENS)],
                    "source_generation_forbidden_ids": [tokenizer.convert_tokens_to_ids(token) for token in markers+list(RESERVED_TOKENS)]}
        (directory/"opcode_registry.json").write_text(json.dumps(registry, indent=2)+"\n")
        checks = verify(tokenizer, known, registry)
        reloaded = AutoTokenizer.from_pretrained(directory)
        if reloaded.get_vocab() != tokenizer.get_vocab() or verify(reloaded, known, registry) != checks:
            raise ValueError("Tokenizer save/reload changed vocabulary or binding behavior")
        samples = []
        for entry in entries[::max(1, len(entries)//1000)]:
            with gzip.open(entry["chunk"], "rt", encoding="utf-8") as stream:
                rows = json.load(stream)[task]
            row = rows[0]
            mapped = bind_bytecode(row[1], known)
            encoded = tokenizer(mapped, add_special_tokens=False)["input_ids"]
            samples.append({"tokens": len(encoded), "unknown": encoded.count(tokenizer.unk_token_id)})
            if task == "segmentation":
                aligned, indices = instruction_token_indices(tokenizer, mapped)
                bounds = row[2]
                if any(index is not None and index >= len(bounds) for index in indices):
                    raise ValueError("Real segmentation example does not align with boundaries")
        checks["real_sample_unknown_tokens"] = sum(row["unknown"] for row in samples)
        checks["real_sample_tokens"] = sum(row["tokens"] for row in samples)
        checks["real_sample_rows"] = len(samples)
        summary["tasks"][task] = {"base_vocab_size": base_vocab, "total_vocab_size": len(tokenizer),
                                  "fit_and_verification_seconds": round(time.time()-start, 2), "checks": checks,
                                  "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}}
        (args.output/"progress.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps({"task_complete": task, **summary["tasks"][task]}), flush=True)
    summary["complete"] = True
    summary["elapsed_seconds"] = round(time.time()-total_start, 2)
    (args.output/"manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"complete": True, "output": str(args.output), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
