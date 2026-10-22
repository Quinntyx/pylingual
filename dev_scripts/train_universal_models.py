"""Full two-epoch universal or ordinary per-version training using ordinary PyLingual Trainer semantics."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import torch
from datasets import load_from_disk
import transformers
from transformers import (AutoTokenizer, RobertaConfig, RobertaForMaskedLM,
 RobertaForTokenClassification, T5ForConditionalGeneration, DataCollatorForSeq2Seq,
 DataCollatorForTokenClassification, DataCollatorForLanguageModeling, Seq2SeqTrainer,
 Trainer, TrainingArguments, Seq2SeqTrainingArguments, TrainerCallback, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed
from dev_scripts.resume_layout import checkpoint_world_size, validate_resume_layout
from dev_scripts.loss_normalization import token_normalized_loss
from pylingual.opcodes import OpcodeRegistry

LABELS={"B":0,"I":1,"E":2}

def write(path, obj):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(obj,indent=2,default=str)+"\n");temporary.replace(path)

def fingerprint(module):
    digest=hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode());digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

class NumericalGuard:
    def _load_rng_state(self, checkpoint):
        saved_world = checkpoint_world_size(checkpoint) if checkpoint is not None else None
        if saved_world is not None and saved_world != self.args.world_size:
            if not getattr(self.args, "_pylingual_allow_repartition", False):
                raise ValueError("Checkpoint world size changed without explicit repartition")
            # Old per-rank CUDA RNG files cannot be restored onto a different
            # device/rank topology. Preserve optimizer/scheduler/scaler state,
            # but deterministically reseed each new rank for this transition.
            seed = self.args.seed + self.args.process_index
            set_seed(seed)
            print(json.dumps({"checkpoint_rng_policy":"reseed-after-world-size-change",
                "saved_world_size":saved_world,"world_size":self.args.world_size,
                "rank":self.args.process_index,"seed":seed}),flush=True)
            return
        return super()._load_rng_state(checkpoint)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result=super().compute_loss(model,inputs,return_outputs=return_outputs,**kwargs)
        loss=result[0] if return_outputs else result
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError("Non-finite training loss; do not hide this with log filtering")
        return result
class GuardedTrainer(NumericalGuard,Trainer): pass
class GuardedSeq2SeqTrainer(NumericalGuard,Seq2SeqTrainer): pass

class Telemetry(TrainerCallback):
    def __init__(self, output): self.output=output;self.updates=0;self.started=time.monotonic()
    def on_train_begin(self,args,state,control,optimizer=None,**kwargs):
        raw=getattr(optimizer,"optimizer",optimizer)
        def record(*unused): self.updates+=1
        self.handle=raw.register_step_post_hook(record)
        self.initial_step=state.global_step
        if state.is_world_process_zero:
            steps=[float(value["step"].item() if hasattr(value["step"],"item") else value["step"])
                   for value in raw.state.values() if "step" in value]
            scheduler=kwargs.get("lr_scheduler")
            scheduler_state=scheduler.state_dict() if scheduler is not None else {}
            scaler=getattr(getattr(self,"trainer",None),"accelerator",None)
            scaler=getattr(scaler,"scaler",None)
            restored={"global_step":state.global_step,"world_size":args.world_size,
                "effective_batch_size":args.train_batch_size*args.gradient_accumulation_steps*args.world_size,
                "optimizer_state_entries":len(raw.state),"optimizer_step_min":min(steps) if steps else None,
                "optimizer_step_max":max(steps) if steps else None,
                "scheduler_last_epoch":scheduler_state.get("last_epoch"),
                "scaler":None if scaler is None else scaler.state_dict()}
            if state.global_step>0 and (not steps or max(steps)<=0):
                raise RuntimeError("Checkpoint resumed without optimizer state")
            write(self.output/"resume-verification.json",restored)
    def on_step_end(self,args,state,control,**kwargs):
        if state.is_world_process_zero and (state.global_step<=5 or state.global_step%100==0):
            elapsed=time.monotonic()-self.started
            write(self.output/"progress.json",{"state":"training","global_step":state.global_step,"max_steps":state.max_steps,
             "epoch":state.epoch,"elapsed_seconds":elapsed,"steps_per_second":(state.global_step-self.initial_step)/elapsed,
             "optimizer_updates_this_process":self.updates,"peak_cuda_GiB":torch.cuda.max_memory_allocated()/2**30,
             "world_size":args.world_size,"effective_batch_size":args.train_batch_size*args.gradient_accumulation_steps*args.world_size,
             "recent_metrics":state.log_history[-3:]})
    def on_train_end(self,*args,**kwargs): self.handle.remove()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=["statement","mlm","segmentation"],required=True)
    parser.add_argument("--data",type=Path,required=True);parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--initialization",type=Path)
    parser.add_argument("--resume-original-world-size",type=int,help="Explicitly repartition the original run while keeping its effective batch unchanged")
    parser.add_argument("--resume",action="store_true");parser.add_argument("--smoke-only",action="store_true")
    parser.add_argument("--steps",type=int,default=0);parser.add_argument("--batch-size",type=int)
    parser.add_argument("--accumulation",type=int,default=1);parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--seed",type=int,default=20261020)
    parser.add_argument("--ordinary-version",choices=["3.14","3.15"])
    args=parser.parse_args()
    if args.steps and not args.smoke_only: raise ValueError("Step caps require --smoke-only")
    torch.set_num_threads(4);set_seed(args.seed)
    manifest=json.loads((args.data/"manifest.json").read_text())
    if not manifest.get("complete") or bool(manifest["smoke_only"])!=args.smoke_only: raise ValueError("Wrong data scope")
    expected_task="statement" if args.task=="statement" else "segmentation"
    if manifest["task"]!=expected_task: raise ValueError("Wrong task dataset")
    scope = "control" if args.ordinary_version else "universal"
    if manifest.get("scope","universal") != scope: raise ValueError("Wrong model/data scope")
    if not args.smoke_only:
        if args.ordinary_version:
            if manifest["files"]!=54500 or manifest["versions"]!={args.ordinary_version:54500} or manifest["corpora"]!={"pypi":49050,"csn":5450}:
                raise ValueError("Ordinary controls require exactly54,500single-version90/10trainingfiles")
        elif manifest["files"]!=100000 or "3.15" in manifest["versions"]:
            raise ValueError("Full universal training requires exactly100k files excluding3.15")
    args.output.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer)
    registry=None if args.ordinary_version else OpcodeRegistry.load(args.tokenizer/"opcode_registry.json",tokenizer)
    if hashlib.sha256((args.tokenizer/"tokenizer.json").read_bytes()).hexdigest()!=manifest["tokenizer_json_sha256"]:
        raise ValueError("Tokenizer changed after dataset encoding")
    if registry is not None and hashlib.sha256((args.tokenizer/"opcode_registry.json").read_bytes()).hexdigest()!=manifest["registry_sha256"]:
        raise ValueError("Opcode registry changed after dataset encoding")
    dataset=load_from_disk(str(args.data/"dataset"))
    if len(dataset)!=manifest["encoded_rows"]: raise ValueError("Dataset row count mismatch")
    stored_state=json.loads((args.data/"dataset/state.json").read_text())
    if stored_state["_fingerprint"]!=manifest["dataset_fingerprint"]:
        raise ValueError("Persisted Arrow dataset fingerprint mismatch")
    # HF may change the in-memory fingerprint while restoring formatting on load.
    if args.task=="statement":
        if args.initialization is None: raise ValueError("Generic CodeT5 initialization required")
        model=T5ForConditionalGeneration.from_pretrained(args.initialization)
        original_vocab=model.config.vocab_size
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id=tokenizer.pad_token_id;model.config.eos_token_id=tokenizer.eos_token_id
        model.config.decoder_start_token_id=tokenizer.pad_token_id;model.config.use_cache=False
        model.generation_config.pad_token_id=tokenizer.pad_token_id;model.generation_config.eos_token_id=tokenizer.eos_token_id
        model.generation_config.decoder_start_token_id=tokenizer.pad_token_id
        model.generation_config.bad_words_ids=None if registry is None else [[n] for n in registry.forbidden_ids]
        collator=DataCollatorForSeq2Seq(tokenizer,model=model,label_pad_token_id=-100)
        trainer_cls=GuardedSeq2SeqTrainer
    elif args.task=="mlm":
        cfg=RobertaConfig(vocab_size=len(tokenizer),max_position_embeddings=1026,layer_norm_eps=1e-5,type_vocab_size=1,
                         pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.cls_token_id,eos_token_id=tokenizer.sep_token_id)
        model=RobertaForMaskedLM(cfg);original_vocab=len(tokenizer)
        collator=DataCollatorForLanguageModeling(tokenizer,mlm=True,mlm_probability=.15)
        dataset=dataset.remove_columns("labels");trainer_cls=GuardedTrainer
    else:
        if args.initialization is None: raise ValueError("Newly trained universal MLM initialization required")
        model=RobertaForTokenClassification.from_pretrained(args.initialization,num_labels=3,id2label={v:k for k,v in LABELS.items()},label2id=LABELS)
        original_vocab=model.config.vocab_size
        if model.config.vocab_size!=len(tokenizer): raise ValueError("MLM/tokenizer vocabulary mismatch")
        collator=DataCollatorForTokenClassification(tokenizer,label_pad_token_id=-100);trainer_cls=GuardedTrainer
    initial_hash=fingerprint(model)
    statement=args.task=="statement";mlm=args.task=="mlm"
    argument_class=Seq2SeqTrainingArguments if statement else TrainingArguments
    training_args=argument_class(output_dir=str(args.output/"training"),num_train_epochs=2,max_steps=args.steps or -1,
      per_device_train_batch_size=args.batch_size or (24 if statement else 28),gradient_accumulation_steps=args.accumulation,
      learning_rate=5e-5 if mlm else 2e-5,weight_decay=0 if mlm else .01,fp16=not mlm,bf16=False,
      logging_steps=1000,save_steps=1000 if mlm else 10000,save_total_limit=5 if mlm else 2,
      report_to=[],seed=args.seed,dataloader_num_workers=args.workers,remove_unused_columns=False,
      ddp_find_unused_parameters=False,disable_tqdm=True,logging_nan_inf_filter=False)
    provenance={"scope":scope,"ordinary_version":args.ordinary_version,"task":args.task,"initialization":str(args.initialization),"initial_weights_sha256":initial_hash,
      "data_manifest":manifest,"tokenizer_registry_fingerprint":None if registry is None else registry.fingerprint,"training_arguments":training_args.to_dict(),
      "original_vocab":original_vocab,"vocab":len(tokenizer),"torch":torch.__version__,"transformers":transformers.__version__,
      "python":platform.python_version(),"hostname":platform.node(),"smoke_only":args.smoke_only,
      "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"world_size":training_args.world_size,"loss_normalization":"global-supervised-token-mean-v1","loss_normalization_sha256":hashlib.sha256(Path(token_normalized_loss.__code__.co_filename).read_bytes()).hexdigest()}
    checkpoint=get_last_checkpoint(str(args.output/"training")) if args.resume and (args.output/"training").exists() else None
    training_args._pylingual_allow_repartition = args.resume_original_world_size is not None
    if args.resume_original_world_size is not None and (not args.resume or checkpoint is None):
        raise ValueError("Repartition requires an existing resumable checkpoint")
    layout=None
    if (args.output/"provenance.json").exists():
        previous=json.loads((args.output/"provenance.json").read_text())
        if not args.resume: raise ValueError("Output exists; explicit --resume required")
        layout=validate_resume_layout(previous,provenance,training_args.world_size,args.resume_original_world_size)
    elif args.resume:
        raise ValueError("Resume requires existing run provenance")
    if training_args.process_index==0:
        if not (args.output/"provenance.json").exists():write(args.output/"provenance.json",provenance)
        execution={"provenance":provenance,"checkpoint":checkpoint,"layout":layout,
                   "checkpoint_world_size":checkpoint_world_size(checkpoint) if checkpoint else None,
                   "observed_at":time.time()}
        write(args.output/"latest-execution.json",execution)
        if args.resume:
            with (args.output/"resume-history.jsonl").open("a") as stream:stream.write(json.dumps(execution,default=str)+"\n")
    if torch.distributed.is_initialized(): torch.distributed.barrier()
    audit=Telemetry(args.output);torch.cuda.reset_peak_memory_stats()
    trainer=trainer_cls(model=model,args=training_args,train_dataset=dataset,data_collator=collator,compute_loss_func=token_normalized_loss,callbacks=[audit])
    audit.trainer=trainer
    checkpoint=get_last_checkpoint(str(args.output/"training")) if args.resume and (args.output/"training").exists() else None
    result=trainer.train(resume_from_checkpoint=checkpoint)
    if not torch.isfinite(torch.tensor(result.training_loss)) or audit.updates<1: raise RuntimeError("No valid optimizer updates")
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all(): raise FloatingPointError("Non-finite trained parameters")
    if args.task=="statement":
        model.config.use_cache=True
        model.generation_config.use_cache=True
    trainer.save_model(str(args.output/"checkpoint"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output/"checkpoint")
        report={"scope":scope,"ordinary_version":args.ordinary_version,"complete":True,"smoke_only":args.smoke_only,"task":args.task,"files":manifest["files"],"encoded_rows":len(dataset),
          "epoch":trainer.state.epoch,"global_step":result.global_step,"optimizer_updates_this_process":audit.updates,
          "steps_this_process":result.global_step-audit.initial_step,"training_loss":result.training_loss,"metrics":result.metrics,
          "world_size":training_args.world_size,"global_batch_size":training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps*training_args.world_size,
          "peak_cuda_GiB":torch.cuda.max_memory_allocated()/2**30,"final_weights_sha256":fingerprint(model)}
        if not args.smoke_only and trainer.state.epoch<1.999: raise RuntimeError("Full two epochs did not finish")
        write(args.output/"result.json",report);write(args.output/"progress.json",report)
        print(json.dumps(report),flush=True)
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
if __name__=="__main__": main()
