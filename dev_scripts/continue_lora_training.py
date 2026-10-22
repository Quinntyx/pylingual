"""Continue existing version LoRAs on expanded data; never update the backbone."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from dev_scripts.lora_expansion_data import SQLiteRows

def frozen_digest(model):
    digest=hashlib.sha256()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def file_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def link_once(source,destination):
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        if destination.resolve()!=Path(source).resolve():
            raise ValueError('Refusing to replace existing output '+str(destination))
    else:
        destination.symlink_to(Path(source).resolve(),target_is_directory=Path(source).is_dir())

def main():
    import torch
    from transformers import T5ForConditionalGeneration,RobertaForTokenClassification,PreTrainedTokenizerFast
    from dev_scripts.shared_training import load_statement_tokenizer
    from dev_scripts.full_ablation_training import BalancedSampler,atomic_save,weights,validation_loss,train_stage
    from pylingual.lora import AdapterBank
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=['statement','segmentation'],required=True)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--accumulation',type=int,default=2)
    parser.add_argument('--eval-every',type=int,default=0)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--seed',type=int,default=20261019)
    parser.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if min(args.epochs,args.batch_size,args.accumulation)<1:
        raise ValueError('Positive training counts required')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    data_manifest=json.loads((args.data/'manifest.json').read_text())
    if not data_manifest.get('complete') or data_manifest.get('new_existing_overlap')!=0:
        raise ValueError('Complete disjoint expansion required')
    if not args.smoke_steps and data_manifest.get('new_training_files')!=50000:
        raise ValueError('Production run requires 50k new training files')
    source=args.models/'universal'/args.task
    output=args.output/'universal'/args.task
    output.mkdir(parents=True,exist_ok=True)
    if (output/'family.json').exists():
        previous=json.loads((output/'family.json').read_text())
        lineage=previous.get('continuation',{})
        if lineage.get('data_manifest_sha256')!=file_digest(args.data/'manifest.json') or lineage.get('source_adapter_sha256')!=file_digest(source/'adapters/3.14/adapter.pt'):
            raise ValueError('Output belongs to a different continuation')
        if previous.get('complete'):
            print(json.dumps({'already_complete':str(output)}),flush=True)
            return
    family=json.loads((source/'family.json').read_text())
    if not family.get('complete'):
        raise ValueError('Original family incomplete')
    for name in ('backbone','tokenizer'):
        link_once(source/name,output/name)
    (output/'adapters').mkdir(exist_ok=True)
    link_once(source/'adapters/3.15',output/'adapters/3.15')
    for version in ('314','315'):
        link_once(args.models/('control'+version),args.output/('control'+version))
    loader=T5ForConditionalGeneration if args.task=='statement' else RobertaForTokenClassification
    tokenizer=load_statement_tokenizer(str(source/'tokenizer')) if args.task=='statement' else PreTrainedTokenizerFast.from_pretrained(source/'tokenizer')
    model=loader.from_pretrained(source/'backbone').to('cuda')
    bank=AdapterBank(model,family['backbone_id'],tuple(family['targets']))
    bank.load('3.14',source/'adapters/3.14')
    parameters=bank.trainable('3.14')
    if sum(p.numel() for p in parameters)!=family['adapter_parameters_per_version']:
        raise ValueError('Continuation adapter parameter mismatch')
    before=frozen_digest(model)
    train={'3.14':SQLiteRows(args.data/'examples.sqlite',args.task,'train')}
    valid={'3.14':SQLiteRows(args.data/'examples.sqlite',args.task,'validation')}
    effective=args.batch_size*args.accumulation
    steps=math.ceil(len(train['3.14'])*args.epochs/effective)
    if args.smoke_steps:
        steps=args.smoke_steps
    args.eval_every=args.eval_every or min(4000,max(250,math.ceil(len(train['3.14'])/effective/4)))
    if args.smoke_steps:
        args.eval_every=min(args.eval_every,args.smoke_steps)
    directory=output/'training/continued314'
    directory.mkdir(parents=True,exist_ok=True)
    model.config.use_cache=False
    if not (directory/'last.pt').exists():
        initial_loss=validation_loss(model,tokenizer,valid,args.task,args.batch_size,args.seed+999)
        sampler=BalancedSampler(train,args.seed)
        optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=0.01,fused=True)
        atomic_save(weights(model),directory/'best-parameters.pt')
        atomic_save({'step':0,'steps':steps,'task':args.task,'weights':weights(model),'optimizer':optimizer.state_dict(),'sampler':sampler.state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'best':initial_loss,'history':[{'step':0,'validation_loss':initial_loss,'seconds':0.0}]},directory/'last.pt')
        del optimizer
        print(json.dumps({'continuing_existing_adapter':True,'task':args.task,'train_rows':len(train['3.14']),'validation_rows':len(valid['3.14']),'steps':steps,'initial_validation_loss':initial_loss,'frozen_digest':before}),flush=True)
    family['complete']=False
    family['continuation']={'source_family':str(source),'source_adapter_sha256':file_digest(source/'adapters/3.14/adapter.pt'),'data_manifest_sha256':file_digest(args.data/'manifest.json'),'new_training_files':data_manifest['new_training_files'],'training_files':data_manifest['total_training_files'],'validation_files':data_manifest['validation_files'],'backbone_frozen_before':before,'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},'smoke':bool(args.smoke_steps)}
    (output/'family.json').write_text(json.dumps(family,indent=2)+'\n')
    result=train_stage(model,tokenizer,train,valid,args.task,directory,steps,args.lr,args,args.seed)
    after=frozen_digest(model)
    if before!=after:
        raise RuntimeError('Frozen backbone changed during continuation')
    model.config.use_cache=True
    bank.save('3.14',output/'adapters/3.14')
    family['adapter_training']['3.14']=result
    family['continuation']['backbone_frozen_after']=after
    family['complete']=True
    (output/'family.json').write_text(json.dumps(family,indent=2)+'\n')
    for group in (train,valid):
        group['3.14'].close()
    print(json.dumps({'complete':True,'task':args.task,'unchanged_backbone':before==after,'selected_step':result['selected_step'],'best_validation_loss':result['best_validation_loss'],'output':str(output)}),flush=True)

if __name__=='__main__':
    main()
