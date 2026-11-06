"""Bounded real-GPU Trainer smoke; synthetic embedding bindings are harness-only.

This is not the production adapter package/loader implementation or a full run.
"""
from __future__ import annotations
import argparse, ast, gc, gzip, hashlib, json, os, time
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq, DataCollatorForTokenClassification, RobertaConfig,
    RobertaForMaskedLM, RobertaForTokenClassification, Seq2SeqTrainer,
    Seq2SeqTrainingArguments, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback)
from dev_scripts.tokenizer_opcode_registry import bind_bytecode, instruction_token_indices
from pylingual.segmentation.sliding_window import sliding_window
from pylingual.lora import AdapterBank, adapter_key

LABELS = {'B': 0, 'I': 1, 'E': 2}

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')

def fixtures(root, task, tokenizer):
    registry=json.loads((root/'tokenizers'/task/'opcode_registry.json').read_text())
    known=set(registry['known_opcodes']); rows=[]
    with (root/'data-100k/accepted.jsonl').open() as stream:
        for line in stream:
            p=json.loads(line)
            with gzip.open(p['chunk'], 'rt') as f: chunk=json.load(f)[task]
            rows.extend(chunk[:12])
            if len(rows)>=192: break
    encoded=[]
    if task=='statement':
        for row in rows:
            features=tokenizer(bind_bytecode(row[1],known),truncation=True,max_length=256,padding='max_length')
            labels=tokenizer(row[0],truncation=True,max_length=256,padding='max_length')['input_ids']
            features['labels']=[x if x!=tokenizer.pad_token_id else -100 for x in labels]
            encoded.append(dict(features))
    else:
        for row in rows:
            bounds=ast.literal_eval(row[2]) if isinstance(row[2],str) else row[2]
            instructions=bind_bytecode(row[1],known).split(' <SEP> ')
            measured=[(text,len(tokenizer(text)['input_ids'])) for text in instructions]
            for window,original in sliding_window(measured,512,128):
                if not window: continue
                value,indices=instruction_token_indices(tokenizer,' <SEP> '.join(window))
                assert len(value['input_ids'])<=1024
                value['labels']=[-100 if i is None else LABELS[bounds[original[i]]] for i in indices]
                if any(x!=-100 for x in value['labels']): encoded.append(dict(value))
                if len(encoded)>=192: break
            if len(encoded)>=192: break
    assert len(encoded)>=56
    return encoded[:192], registry

def arguments(path, task, steps=4):
    statement=task=='statement'; mlm=task=='mlm'
    cls=Seq2SeqTrainingArguments if statement else TrainingArguments
    return cls(output_dir=str(path),max_steps=steps,per_device_train_batch_size=24 if statement else 28,
        gradient_accumulation_steps=1,learning_rate=5e-5 if mlm else 2e-5,
        weight_decay=0.0 if mlm else 0.01,fp16=not mlm,bf16=False,
        save_strategy='no',logging_steps=1,report_to=[],dataloader_num_workers=0,
        remove_unused_columns=False,ddp_find_unused_parameters=False,seed=20261020,
        disable_tqdm=True)

def train(model, tokenizer, rows, task, out):
    if task=='statement':
        collator=DataCollatorForSeq2Seq(tokenizer,model=model,label_pad_token_id=-100)
        trainer_cls=Seq2SeqTrainer
    elif task=='mlm':
        collator=DataCollatorForLanguageModeling(tokenizer,mlm=True,mlm_probability=0.15)
        rows=[{k:v for k,v in row.items() if k!='labels'} for row in rows]
        trainer_cls=Trainer
    else:
        collator=DataCollatorForTokenClassification(tokenizer,label_pad_token_id=-100)
        trainer_cls=Trainer
    torch.cuda.reset_peak_memory_stats()
    class StepAudit(TrainerCallback):
        completed=0
        def on_train_begin(self,args,state,control,optimizer=None,**kwargs):
            # Trainer callbacks can receive the raw optimizer, not its Accelerate
            # wrapper. Count actual AdamW.step calls: scaler-skipped calls do not run.
            raw=getattr(optimizer,'optimizer',optimizer)
            def updated(optim,args,kwargs): self.completed+=1
            self.handle=raw.register_step_post_hook(updated)
        def on_train_end(self,args,state,control,**kwargs):
            self.handle.remove()
    step_audit=StepAudit()
    trainer=trainer_cls(model=model,args=arguments(out,task),train_dataset=Dataset.from_list(rows),data_collator=collator,callbacks=[step_audit])
    result=trainer.train()
    assert torch.isfinite(torch.tensor(result.training_loss))
    trainer.save_model(out/'checkpoint')
    report={'loss':result.training_loss,'steps':result.global_step,'world_size':trainer.args.world_size,
        'per_device_batch':trainer.args.per_device_train_batch_size,'fp16':trainer.args.fp16,
        'peak_cuda_GiB':round(torch.cuda.max_memory_allocated()/1024**3,3),'metrics':result.metrics,'optimizer_updates':step_audit.completed,'fp16_overflow_skips':result.global_step-step_audit.completed}
    assert step_audit.completed>0,'No optimizer update completed'
    if trainer.is_world_process_zero(): write(out/'result.json',report)
    if torch.distributed.is_initialized(): torch.distributed.barrier()
    model=trainer.accelerator.unwrap_model(trainer.model_wrapped,keep_fp32_wrapper=False)
    if trainer.is_world_process_zero():
        reloaded=type(model).from_pretrained(out/'checkpoint').to(next(model.parameters()).device).eval()
        model.eval();device=next(model.parameters()).device
        sample={k:torch.tensor([v],device=device) for k,v in rows[0].items() if task=='statement' or k!='labels'}
        with torch.no_grad(): torch.testing.assert_close(reloaded(**sample).logits,model(**sample).logits,rtol=0,atol=0)
        report['checkpoint_logits_exact']=True
        write(out/'result.json',report)
        del reloaded;gc.collect();torch.cuda.empty_cache()
    if torch.distributed.is_initialized(): torch.distributed.barrier()
    return model,report

def fingerprint(parameters):
    h=hashlib.sha256()
    for name,p in parameters:
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def adapter_smoke(model, tokenizer, rows, registry, task, out):
    device='cuda';model=model.to(device)
    originals=list(model.named_parameters());before=fingerprint(originals)
    bank=AdapterBank(model,'smoke-only-'+task,('q','v') if task=='statement' else ('query','value'))
    for version in ('smokeA','smokeB'): bank.add(version,rank=8,alpha=16.0)
    model.smoke_embedding_deltas=torch.nn.ParameterDict({adapter_key(v):torch.nn.Parameter(torch.zeros(1,model.config.d_model if task=='statement' else model.config.hidden_size,device=device)) for v in ('smokeA','smokeB')})
    active=[None];slot=registry['reserved_slots'][0]['id']
    def correction(ids):
        return (ids==slot).unsqueeze(-1)*model.smoke_embedding_deltas[adapter_key(active[0])][0]
    if task=='statement':
        def encoder_hook(module,args,kwargs):
            if active[0] is not None:
                ids=kwargs['input_ids'];kwargs['inputs_embeds']=model.get_input_embeddings()(ids)+correction(ids);kwargs['input_ids']=None
            return args,kwargs
        handle=model.get_encoder().register_forward_pre_hook(encoder_hook,with_kwargs=True)
    else:
        def embedding_hook(module,args,output):
            return output if active[0] is None else output+correction(args[0])
        handle=model.get_input_embeddings().register_forward_hook(embedding_hook)
    sample={k:torch.tensor([v],device=device) for k,v in rows[0].items()}
    sample['input_ids'][0,1]=slot
    assert sample['attention_mask'][0,1]==1
    if task!='statement': assert sample['labels'][0,1]!=-100
    reference=None;reports={}
    for version in ('smokeA','smokeB'):
        active[0]=version;parameters=bank.trainable(version)
        delta=model.smoke_embedding_deltas[adapter_key(version)];delta.requires_grad_(True);parameters=parameters+[delta]
        assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in parameters}
        optimizer=torch.optim.AdamW(parameters,lr=2e-5,weight_decay=0.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        model.train();grad_norm=0.0;lora_norm=0.0
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): loss=model(**sample).loss
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            grad_norm=max(grad_norm,float(delta.grad.norm()))
            lora_norm=max(lora_norm,max(float(p.grad.norm()) for p in parameters[:-1] if p.grad is not None))
            scaler.step(optimizer);scaler.update()
        assert grad_norm>0 and lora_norm>0 and bool(delta.detach().ne(0).any())
        model.eval()
        with torch.no_grad(): expected=model(**sample).logits.detach().clone()
        directory=out/version;bank.save(version,directory)
        torch.save({'slot_ids':[slot],'delta':delta.detach().cpu()},directory/'embedding-delta.pt')
        write(directory/'smoke-bindings.json',{'harness_only':True,'synthetic_opcode':'SMOKE_FUTURE_'+version.upper(),'slot':0,'id':slot})
        for layer in bank.layers.values(): del layer.adapters[adapter_key(version)]
        del bank.specs[version]
        delta.data.zero_();bank.load(version,directory)
        restored=torch.load(directory/'embedding-delta.pt',weights_only=True,map_location=device)
        assert restored['slot_ids']==[slot];delta.data.copy_(restored['delta']);bank.select(version)
        with torch.no_grad(): actual=model(**sample).logits.detach()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        if version=='smokeA': reference=expected
        else:
            active[0]='smokeA';bank.select('smokeA')
            with torch.no_grad(): previous=model(**sample).logits.detach()
            torch.testing.assert_close(previous,reference,rtol=0,atol=0)
        reports[version]={'delta_gradient_norm':grad_norm,'lora_gradient_norm':lora_norm,'saved_and_reloaded_exactly':True,'slot':slot,'steps':2}
        del optimizer
    active[0]=None;bank.select(None)
    assert fingerprint(originals)==before,'Frozen backbone mutated'
    handle.remove()
    result={'passed':True,'synthetic_bindings_only':True,'same_slot_reuse_isolated':True,'frozen_backbone_sha256':before,'versions':reports,'production_adapter_format_implemented':False}
    write(out/'result.json',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--task',choices=['statement','segmentation','statement-adapter'],required=True)
    args=parser.parse_args();root=args.root;out=root/'gpu-smoke-20261020-v3';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261020)
    task='statement' if args.task.startswith('statement') else 'segmentation'
    tokenizer=AutoTokenizer.from_pretrained(root/'tokenizers'/task)
    rows,registry=fixtures(root,task,tokenizer)
    started=time.monotonic()
    if args.task=='statement':
        model=T5ForConditionalGeneration.from_pretrained(root/'gpu-smoke-20261020/generic-codet5')
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id=tokenizer.pad_token_id;model.config.eos_token_id=tokenizer.eos_token_id;model.config.decoder_start_token_id=tokenizer.pad_token_id
        model.config.use_cache=False
        model,report=train(model,tokenizer,rows,'statement',out/'statement')
        if int(os.environ.get('RANK','0'))==0:
            model.eval();device=next(model.parameters()).device
            sample={k:torch.tensor([v],device=device) for k,v in rows[0].items() if k!='labels'}
            with torch.no_grad(): generated=model.generate(**sample,max_new_tokens=8,bad_words_ids=[[n] for n in registry['source_generation_forbidden_ids']])
            assert not set(generated[0].tolist()).intersection(registry['source_generation_forbidden_ids'])
            report['generation_passed']=True;report['seconds']=round(time.monotonic()-started,2)
            write(out/'statement/result.json',report)
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
    elif args.task=='statement-adapter':
        model=T5ForConditionalGeneration.from_pretrained(out/'statement/checkpoint')
        adapter_smoke(model,tokenizer,rows,registry,task,out/'statement-adapters')
    else:
        cfg=RobertaConfig(vocab_size=len(tokenizer),max_position_embeddings=1026,layer_norm_eps=1e-5,type_vocab_size=1,pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.cls_token_id,eos_token_id=tokenizer.sep_token_id)
        mlm=RobertaForMaskedLM(cfg)
        mlm,report=train(mlm,tokenizer,rows,'mlm',out/'mlm')
        mlm_encoder_hash=fingerprint(list(mlm.roberta.named_parameters()))
        del mlm;gc.collect();torch.cuda.empty_cache()
        model=RobertaForTokenClassification.from_pretrained(out/'mlm/checkpoint',num_labels=3,id2label={i:k for k,i in LABELS.items()},label2id=LABELS)
        assert fingerprint(list(model.roberta.named_parameters()))==mlm_encoder_hash
        model,report=train(model,tokenizer,rows,'segmentation',out/'segmentation')
        report['mlm_encoder_transfer_exact']=True;write(out/'segmentation/result.json',report)
        saved=RobertaForTokenClassification.from_pretrained(out/'segmentation/checkpoint').to('cuda').eval()
        model.eval();sample={k:torch.tensor([v],device='cuda') for k,v in rows[0].items()}
        with torch.no_grad(): torch.testing.assert_close(saved(**sample).logits,model(**sample).logits,rtol=0,atol=0)
        del saved;gc.collect();torch.cuda.empty_cache()
        adapter_smoke(model,tokenizer,rows,registry,task,out/'segmentation-adapters')
    print(json.dumps({'complete':args.task,'seconds':round(time.monotonic()-started,2)}),flush=True)

if __name__=='__main__': main()
