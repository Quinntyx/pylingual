"""Materialize full-coverage Arrow examples from the audited source-file journal."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import time
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
from pylingual.opcodes import OpcodeRegistry, SEPARATOR, instruction_token_indices, tokenizer_fingerprint
from pylingual.segmentation.sliding_window import sliding_window

LABELS = {"B": 0, "I": 1, "E": 2}

def encoded_generator(records, task, tokenizer_dir, ordinary=False, bindings=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    registry = None if ordinary else OpcodeRegistry.load(Path(tokenizer_dir)/"opcode_registry.json", tokenizer)
    bind = (lambda text: text) if ordinary else (lambda text: registry.bind(text, bindings or {}))
    @lru_cache(maxsize=200000)
    def instruction_length(instruction):
        return len(tokenizer(instruction)["input_ids"])
    for record in records:
        with gzip.open(record["chunk"], "rt", encoding="utf-8") as source:
            examples = json.load(source)[task]
        if task == "statement":
            # Same 256/256 policy as normal tokenize_seq2seq; batch for throughput.
            for offset in range(0, len(examples), 512):
                batch = examples[offset:offset+512]
                inputs = tokenizer([bind(row[1]) for row in batch], truncation=True, max_length=256)
                targets = tokenizer([row[0] for row in batch], truncation=True, max_length=256)
                for index in range(len(batch)):
                    yield {"input_ids": inputs["input_ids"][index], "attention_mask": inputs["attention_mask"][index], "labels": targets["input_ids"][index]}
        else:
            for source, bytecode, boundary in examples:
                instructions = bind(bytecode).split(SEPARATOR)
                bounds = ast.literal_eval(boundary) if isinstance(boundary, str) else boundary
                if len(bounds) != len(instructions) or any(label not in LABELS for label in bounds):
                    raise ValueError("Invalid boundary labels: "+record["file"])
                measured = [(instruction, instruction_length(instruction)) for instruction in instructions]
                if any(size > 512 for _, size in measured):
                    raise ValueError("An instruction exceeds the normal segmentation budget")
                for window, originals in sliding_window(measured, 512, 128):
                    if not window:
                        raise ValueError("Empty segmentation window")
                    features, indices = instruction_token_indices(tokenizer, SEPARATOR.join(window))
                    if len(features["input_ids"]) > 512:
                        raise ValueError("Segmentation window exceeds its budget")
                    yield {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"],
                           "labels": [-100 if i is None else LABELS[bounds[originals[i]]] for i in indices]}

def prepare(data, tokenizer, task, output, workers, limit=0, scope="universal", bindings_file=None):
    journal = data/"accepted.jsonl"
    digest = hashlib.sha256(journal.read_bytes()).hexdigest()
    source_manifest = json.loads((data/"manifest.json").read_text())
    records = [json.loads(line) for line in journal.open()]
    if not source_manifest["complete"] or len(records) != source_manifest["files"]:
        raise ValueError("Incomplete source preparation")
    if scope == "universal" and "3.15" in {row["version"] for row in records}:
        raise ValueError("Held-out version leaked into backbone training")
    if len({row["canonical_sha256"] for row in records}) != len(records):
        raise ValueError("Duplicate source canonical")
    if limit:
        records = records[:limit]
    elif scope == "universal":
        from dev_scripts.prepare_tokenizer_corpus import quotas
        cells = Counter((row["version"], row["corpus"]) for row in records)
        if cells != Counter({(v, c): n for v, counts in quotas(len(records)).items() for c, n in counts.items()}):
            raise ValueError("Expected a full version-balanced 90/10 source-file mixture")
    registry_path = tokenizer/"opcode_registry.json"
    bindings = {}
    if scope == "adapter":
        registry = OpcodeRegistry.load(registry_path, AutoTokenizer.from_pretrained(tokenizer))
        observed = {op for row in records for op in row["observed_opcodes"]}
        bindings = registry.validate_bindings(json.loads(bindings_file.read_text())) if bindings_file else registry.bindings_for(observed)
    manifest = {"scope":scope,"bindings":bindings,"split":source_manifest.get("split","train"),"task": task, "journal_sha256": digest, "files": len(records), "smoke_only": bool(limit),
                "encoding_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "tokenizer_json_sha256": hashlib.sha256((tokenizer/"tokenizer.json").read_bytes()).hexdigest(),
                "registry_sha256": None if scope == "control" else hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                "versions": dict(Counter(row["version"] for row in records)),
                "corpora": dict(Counter(row["corpus"] for row in records)),
                "raw_examples": sum(row[task+"_rows"] for row in records),
                "all_converter_rows": True, "limits": {"statement_input":256,"statement_target":256,"segmentation_window":512,"stride":128}}
    if scope == "adapter":
        manifest["registry_fingerprint"] = registry.fingerprint
        manifest["tokenizer_semantic_fingerprint"] = tokenizer_fingerprint(AutoTokenizer.from_pretrained(tokenizer))
    if (output/"manifest.json").exists():
        prior = json.loads((output/"manifest.json").read_text())
        identity = {k:v for k,v in manifest.items() if k != "encoding_implementation_sha256"}
        if not prior.get("complete") or any(prior.get(k, "universal" if k == "scope" else {} if k == "bindings" else "train" if k == "split" else None) != v for k,v in identity.items()):
            raise ValueError("Prepared data configuration changed")
        print(json.dumps({"reused":str(output),"rows":prior["encoded_rows"]}), flush=True)
        return
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    features = Features({"input_ids":Sequence(Value("int32")),"attention_mask":Sequence(Value("int8")),"labels":Sequence(Value("int32"))})
    dataset = Dataset.from_generator(encoded_generator, gen_kwargs={"records":records,"task":task,"tokenizer_dir":str(tokenizer),"ordinary":scope=="control","bindings":bindings},
                                     features=features, num_proc=min(workers,len(records)),cache_dir=str(output/"build-cache"))
    dataset.save_to_disk(str(output/"dataset"), num_shards=max(1,min(32,len(dataset)//100000+1)))
    manifest.update(complete=True, encoded_rows=len(dataset), dataset_fingerprint=dataset._fingerprint, seconds=time.monotonic()-started)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True);parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--task",choices=["statement","segmentation"],required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--smoke-files",type=int,default=0)
    parser.add_argument("--scope",choices=["universal","adapter","control"],default="universal")
    parser.add_argument("--bindings-file",type=Path)
    args=parser.parse_args();prepare(args.data,args.tokenizer,args.task,args.output,args.workers,args.smoke_files,args.scope,args.bindings_file)
if __name__=="__main__": main()
