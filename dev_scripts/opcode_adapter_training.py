"""Production v2 adapter training; never resize vocabularies or save a merged backbone."""
from __future__ import annotations
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path
import torch
from datasets import Dataset
from transformers import Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, DataCollatorForTokenClassification, TrainerCallback
from pylingual.opcodes import SEPARATOR, opcode_names, instruction_token_indices
from pylingual.lora import adapter_key
from dev_scripts import hub_checkpoints
from pylingual.shared_models import load_family
from pylingual.segmentation.sliding_window import sliding_window

LABELS = {"B": 0, "I": 1, "E": 2}


def read_rows(path):
    if path is None:
        return []
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def encode_rows(task, tokenizer, registry, rows, bindings, prefix="", max_input=None, max_output=256):
    if max_input is None:
        max_input = 256 if task == "statement" else 512
    encoded = []
    for row in rows:
        bytecode = registry.bind(row["bytecode"], bindings)
        if task == "statement":
            features = dict(tokenizer(prefix.format(version=row["version"]) + bytecode, truncation=True, max_length=max_input))
            features.pop("token_type_ids", None)
            features["labels"] = tokenizer(row["source"], truncation=True, max_length=max_output)["input_ids"]
            encoded.append(features)
        else:
            bounds = ast.literal_eval(row["boundary"]) if isinstance(row["boundary"], str) else row["boundary"]
            instructions = bytecode.split(SEPARATOR)
            if len(bounds) != len(instructions) or any(label not in LABELS for label in bounds):
                raise ValueError("Segmentation boundaries must cover every instruction with B/I/E")
            measured = [(text, len(tokenizer(text)["input_ids"])) for text in instructions]
            if any(length > max_input for _, length in measured):
                raise ValueError("One instruction exceeds the segmentation window")
            for window, originals in sliding_window(measured, max_input, max_input // 4):
                if not window:
                    raise ValueError("Empty segmentation window")
                features, indices = instruction_token_indices(tokenizer, SEPARATOR.join(window))
                if len(features["input_ids"]) > max_input:
                    raise ValueError("Segmentation window exceeds tokenizer capacity")
                features["labels"] = [-100 if index is None else LABELS[bounds[originals[index]]] for index in indices]
                encoded.append(dict(features))
    if rows and not encoded:
        raise ValueError("No usable adapter examples")
    return encoded


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".adapters." in name or "_pylingual_input_updates" in name:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_encoded_dataset(directory, selection, args, bank, tokenizer, metadata, bindings, split):
    from datasets import load_from_disk
    from pylingual.opcodes import tokenizer_fingerprint
    doc = json.loads((directory/"manifest.json").read_text())
    expected_task = metadata["task"]
    if not doc.get("complete") or doc.get("scope") != "adapter" or doc.get("task") != expected_task or doc.get("split") != split:
        raise ValueError("Wrong encoded adapter data scope/task/partition")
    if doc.get("smoke_only") and args.steps is None:
        raise ValueError("Smoke data cannot be used for full adapter training")
    if doc.get("files") != sum(1 for line in selection.open() if line.strip()) or not doc.get("all_converter_rows"):
        raise ValueError("Encoded source-file budget or converter coverage changed")
    if doc.get("limits") != {"statement_input":256,"statement_target":256,"segmentation_window":512,"stride":128}:
        raise ValueError("Encoded token limits differ from normal training")
    if doc["journal_sha256"] != hashlib.sha256(selection.read_bytes()).hexdigest():
        raise ValueError("Encoded data does not match the selected source journal")
    if doc.get("registry_fingerprint") != bank.registry.fingerprint or doc.get("tokenizer_semantic_fingerprint") != tokenizer_fingerprint(tokenizer):
        raise ValueError("Encoded data tokenizer/registry differs from the frozen family")
    if doc.get("bindings") != bindings or set(doc["versions"]) != {args.version}:
        raise ValueError("Encoded adapter version/bindings changed")
    if metadata.get("source_prefix", ""):
        raise ValueError("Preencoded full-data path does not support source prefixes")
    state = json.loads((directory/"dataset/state.json").read_text())
    if state["_fingerprint"] != doc["dataset_fingerprint"]:
        raise ValueError("Persisted encoded dataset fingerprint changed")
    data = load_from_disk(str(directory/"dataset"))
    if len(data) != doc["encoded_rows"] or not len(data):
        raise ValueError("Encoded adapter row count changed")
    return data


def adapter_card(repo_id, args, provenance, metadata):
    hub = json.loads((args.encoded_data/"hub-source.json").read_text()) if getattr(args, "encoded_data", None) and (args.encoded_data/"hub-source.json").exists() else None
    backbone = metadata.get("hub") or {}
    return f"""---
license: other
---
# {repo_id}

Python {args.version} LoRA for the frozen universal PyLingual {metadata['task']} backbone
(shared-backbone + per-version LoRA study). The backbone and tokenizer are never modified;
`last-checkpoint/` holds the latest resumable adapter checkpoint while training runs and
`adapters/{args.version}/` plus `family.json` hold the final adapter.

- Backbone: {f"[`{backbone['repo_id']}`](https://huggingface.co/{backbone['repo_id']}) at revision `{backbone['revision']}`" if backbone else "local"}, backbone_id `{metadata['backbone_id']}`
- Data: {f"[`{hub['repo_id']}`](https://huggingface.co/datasets/{hub['repo_id']}) `{hub['path']}` at revision `{hub['revision']}`" if hub else "local"}
- Rank {args.rank}, alpha {2*args.rank}, dropout 0.05 on {metadata['targets']}; opcode bindings `{json.dumps(provenance['bindings'])}`
- {args.epochs} epochs, batch {args.batch_size} x {args.accumulation} accumulation, learning rate {args.learning_rate}, {args.precision.upper()}, seed {args.seed}
- Code: https://github.com/quinntyx/pylingual branch `feat-shared-backbone-lora`, commit `{hub_checkpoints.git_commit()}`
"""


def train_opcode_adapter(args, metadata):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        raise RuntimeError("Distributed adapter training requires an initialized process group")
    primary = world_size == 1 or torch.distributed.get_rank() == 0
    if args.steps is None and (args.validation_data is None or args.epochs is None):
        raise ValueError("Full training requires explicit --epochs and --validation-data; --steps is for bounded checks")
    if args.epochs is not None and args.epochs <= 0 or args.steps is not None and args.steps < 1:
        raise ValueError("Training epochs/steps must be positive")
    if min(args.batch_size,args.accumulation,args.rank,args.checkpoint_steps) < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid optimizer/training settings")
    task = metadata["task"]
    bank, tokenizer, _, _ = load_family(args.family, task, torch.device("cuda", local_rank))
    train_rows, valid_rows = read_rows(args.data), read_rows(args.validation_data)
    if not train_rows or any(row.get("version") != args.version or not row.get("file") for row in train_rows+valid_rows):
        raise ValueError("Every adapter row must identify its version and source file")
    if {row["file"] for row in train_rows} & {row["file"] for row in valid_rows}:
        raise ValueError("Adapter training/validation source-file overlap")
    if args.bindings is not None:
        bindings = bank.registry.validate_bindings(json.loads(args.bindings.read_text()))
    elif getattr(args, "encoded_data", None):
        bindings = bank.registry.validate_bindings(json.loads((args.encoded_data/"manifest.json").read_text())["bindings"])
    elif args.version in bank.bindings:
        bindings = bank.bindings[args.version]
    else:
        names = set().union(*(opcode_names(row["bytecode"]) for row in train_rows))
        bindings = bank.registry.bindings_for(names)
    if args.version in metadata["adapters"]:
        bank.load(args.version,args.family/metadata["adapters"][args.version])
        if bank.specs[args.version] != {"rank": args.rank, "alpha": 2 * args.rank, "dropout": .05}:
            raise ValueError("Existing adapter LoRA configuration differs from requested rank/alpha/dropout")
        if bindings != bank.bindings[args.version]:
            raise ValueError("Existing adapter bindings are immutable")
    else:
        bank.add(args.version,rank=args.rank,alpha=2*args.rank,dropout=.05,bindings=bindings)
    parameters = bank.trainable(args.version)
    bank.model.config.use_cache = False
    if getattr(args, "encoded_data", None):
        if valid_rows and getattr(args, "encoded_validation", None) is None:
            raise ValueError("Preencoded training requires matching preencoded validation")
        train_data = load_encoded_dataset(args.encoded_data,args.data,args,bank,tokenizer,metadata,bindings,"train")
        valid_data = load_encoded_dataset(args.encoded_validation,args.validation_data,args,bank,tokenizer,metadata,bindings,"validation") if valid_rows else None
    else:
        train_data = Dataset.from_list(encode_rows(task,tokenizer,bank.registry,train_rows,bindings,metadata.get("source_prefix", "")))
        valid_data = Dataset.from_list(encode_rows(task,tokenizer,bank.registry,valid_rows,bindings,metadata.get("source_prefix", ""))) if valid_rows else None
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    provenance = {"backbone_id":bank.backbone_id,"tokenizer_sha256":bank.tokenizer_id,"opcode_registry_sha256":bank.registry.fingerprint,
                  "version":args.version,"bindings":bindings,"data_sha256":data_hash,"validation_sha256":hashlib.sha256(args.validation_data.read_bytes()).hexdigest() if args.validation_data else None,
                  "rank":args.rank,"alpha":2*args.rank,"dropout":.05,
                  "encoding":"opcode-fields-offset-alignment-v1","input_limit":256 if task=="statement" else 512,"output_limit":256,
                  "segmentation_stride":128 if task=="segmentation" else None,"batch_size":args.batch_size,"accumulation":args.accumulation,"learning_rate":args.learning_rate,"epochs":args.epochs,"steps":args.steps,"precision":args.precision,"seed":args.seed}
    updated = dict(metadata,complete=False)
    if primary:
        shutil.copytree(args.family,args.output,ignore=shutil.ignore_patterns("family.json"))
        (args.output/"family.json").write_text(json.dumps(updated,indent=2)+"\n")
    if world_size > 1:
        torch.distributed.barrier()
    before = frozen_digest(bank.model)
    base_trainer = Seq2SeqTrainer if task == "statement" else Trainer
    class AdapterTrainer(base_trainer):
        def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
            result=super().compute_loss(model,inputs,return_outputs=return_outputs,**kwargs)
            loss=result[0] if return_outputs else result
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite adapter training loss")
            return result
        def _save(self,output_dir=None,state_dict=None,**kwargs):
            if self.is_world_process_zero():
                directory=Path(output_dir or self.args.output_dir)
                bank.save(args.version,directory)
                (directory/"run-provenance.json").write_text(json.dumps(provenance,sort_keys=True,indent=2)+"\n")
        def _load_from_checkpoint(self,resume_from_checkpoint,model=None,**kwargs):
            directory=Path(resume_from_checkpoint)
            if json.loads((directory/"run-provenance.json").read_text()) != provenance:
                raise ValueError("Resume data, tokenizer, binding or optimizer configuration mismatch")
            bank.load(args.version,directory,replace=True)
            bank.trainable(args.version)
    argument_class = Seq2SeqTrainingArguments if task == "statement" else TrainingArguments
    training_args = argument_class(output_dir=str(args.output/"training"),max_steps=args.steps or -1,num_train_epochs=args.epochs or 1,
        per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=args.batch_size,gradient_accumulation_steps=args.accumulation,
        learning_rate=args.learning_rate,weight_decay=.01,fp16=args.precision=="fp16",bf16=False,logging_steps=1 if args.steps else 1000,
        save_strategy="steps",save_steps=args.checkpoint_steps,save_total_limit=2,report_to=[],seed=args.seed,remove_unused_columns=False,
        dataloader_num_workers=0,disable_tqdm=bool(args.steps) or world_size > 1,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        average_tokens_across_devices=world_size > 1,
        logging_nan_inf_filter=world_size == 1)
    if getattr(args, "encoded_data", None):
        provenance["encoded_manifest"] = json.loads((args.encoded_data/"manifest.json").read_text())
    provenance.update(optimizer=training_args.optim.value,weight_decay=training_args.weight_decay,
                      max_grad_norm=training_args.max_grad_norm,lr_scheduler=training_args.lr_scheduler_type.value)
    hub_repo = getattr(args, "hub_repo", None)
    if hub_repo and primary:
        hub_checkpoints.ensure_repo(hub_repo, adapter_card(hub_repo, args, provenance, metadata))
        if args.resume is None:
            # A fresh host continues from the Hub's last checkpoint instead of restarting.
            args.resume = hub_checkpoints.restore(hub_repo, args.output.parent/(args.output.name+"-hub-resume"))
    if world_size > 1:
        torch.distributed.barrier()
    collator = DataCollatorForSeq2Seq(tokenizer,model=bank.model,label_pad_token_id=-100) if task == "statement" else DataCollatorForTokenClassification(tokenizer,label_pad_token_id=-100)
    from dev_scripts.train_universal_models import Telemetry
    audit = Telemetry(args.output)
    if world_size > 1:
        from dev_scripts.loss_normalization import token_normalized_loss
    if world_size > 1:
        provenance.update(world_size=training_args.world_size,
            global_batch_size=training_args.per_device_train_batch_size*args.accumulation*training_args.world_size,
            loss_normalization="global-supervised-token-mean-v1")
    callbacks = [audit]
    if hub_repo:
        callbacks.append(hub_checkpoints.HubCheckpointPush(hub_repo, args.output))
    trainer = AdapterTrainer(model=bank.model,args=training_args,train_dataset=train_data,eval_dataset=valid_data,data_collator=collator,callbacks=callbacks,
        compute_loss_func=token_normalized_loss if world_size > 1 else None)
    audit.trainer = trainer
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    validation = trainer.evaluate() if valid_data is not None else {}
    if audit.updates < 1 or not torch.isfinite(torch.tensor(result.training_loss)) or before != frozen_digest(bank.model):
        raise RuntimeError("No optimizer updates, non-finite loss or frozen-backbone mutation")
    if world_size > 1:
        digests = [None] * world_size
        torch.distributed.all_gather_object(digests, frozen_digest(bank.model))
        if any(digest != before for digest in digests):
            raise RuntimeError("Distributed ranks disagree on frozen backbone identity")
        torch.distributed.barrier()
    if not primary:
        torch.distributed.barrier()
        return
    relative = f"adapters/{args.version}"
    bank.save(args.version,args.output/relative)
    updated["adapters"] = {**metadata["adapters"],args.version:relative}
    updated["versions"] = sorted(set(metadata["versions"])|{args.version},key=lambda value:tuple(map(int,value.split('.'))))
    updated["opcode_bindings"] = {**metadata.get("opcode_bindings",{}),args.version:bindings}
    updated["adapter_updates"] = {**metadata.get("adapter_updates",{}),args.version:{**provenance,"source_family":str(args.family.resolve()),"files":len({row["file"] for row in train_rows}),"encoded_rows":len(train_data),"parameters":sum(p.numel() for p in parameters),"frozen_backbone_sha256":before,"optimizer_updates":audit.updates,"training":result.metrics,"validation":validation}}
    updated["complete"] = True
    temporary=args.output/"family.json.tmp";temporary.write_text(json.dumps(updated,indent=2)+"\n");temporary.replace(args.output/"family.json")
    if hub_repo:
        staging = args.output/"hub-final"
        shutil.copytree(args.output/relative, staging/relative)
        shutil.copy(args.output/"family.json", staging/"family.json")
        revision = hub_checkpoints.push_final(hub_repo, staging, args.output, f"Final {args.version} adapter")
        (args.output/"hub.json").write_text(json.dumps({"repo_id": hub_repo, "revision": revision}, indent=2)+"\n")
    print(json.dumps({"complete":True,"output":str(args.output),"version":args.version,"bindings":bindings,"parameters":sum(p.numel() for p in parameters)}),flush=True)
    if world_size > 1:
        torch.distributed.barrier()
