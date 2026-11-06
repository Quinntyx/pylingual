"""Full-coverage, validation-selected shared/backbone and version-only ablations.

Statement models always start at generic Salesforce/codet5-base. Segmenters
follow the ordinary fresh WordPiece -> fresh RoBERTa MLM -> token classifier
pipeline. No PyLingual neural checkpoint initializes any study arm.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForMaskedLM, RobertaForTokenClassification, T5ForConditionalGeneration
from dev_scripts.shared_training import LABELS, VERSIONS, collate, encode_row, load_statement_tokenizer, read_rows
from pylingual.lora import AdapterBank


class BalancedSampler:
    def __init__(self, groups, seed):
        self.groups = groups
        self.versions = sorted(groups)
        sizes = {len(group) for group in groups.values()}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError('Each training version must have the same nonzero row count')
        self.size = sizes.pop()
        self.rng = random.Random(seed)
        self.orders = {v:list(range(self.size)) for v in self.versions}
        self.position = 0
        self.epoch = 0
        for order in self.orders.values():
            self.rng.shuffle(order)

    def take(self, count):
        rows = []
        for _ in range(count):
            if self.position == self.size*len(self.versions):
                self.position = 0
                self.epoch += 1
                for order in self.orders.values():
                    self.rng.shuffle(order)
            version = self.versions[self.position % len(self.versions)]
            index = self.position // len(self.versions)
            rows.append(self.groups[version][self.orders[version][index]])
            self.position += 1
        return rows

    def state(self):
        return {'orders':self.orders,'position':self.position,'epoch':self.epoch,'rng':self.rng.getstate()}

    def restore(self,state):
        self.orders,self.position,self.epoch = state['orders'],state['position'],state['epoch']
        self.rng.setstate(state['rng'])


def encode_groups(task, tokenizer, rows):
    groups = defaultdict(list)
    for row in rows:
        encoded = encode_row('segmentation' if task == 'mlm' else task,tokenizer,row)
        if encoded is None:
            raise ValueError('Prepared example unexpectedly failed encoding: '+row['file'])
        groups[row['version']].append(encoded)
    return dict(groups)


def make_batch(rows,tokenizer,device,task):
    batch = collate(rows,tokenizer,device)
    if task != 'mlm':
        return batch
    ids = batch['input_ids']
    eligible = batch['attention_mask'].bool()
    for token in tokenizer.all_special_ids:
        eligible &= ids != token
    mask = (torch.rand(ids.shape,device=device)<0.15)&eligible
    if not mask.any():
        positions = eligible.nonzero()
        if not len(positions):
            raise ValueError('No maskable MLM tokens')
        mask[tuple(positions[0])] = True
    labels = ids.clone()
    labels[~mask] = -100
    replace = torch.rand(ids.shape,device=device)
    random_mask = mask & (replace >= 0.8)&(replace < 0.9)
    ids[mask & (replace < 0.8)] = tokenizer.mask_token_id
    random_ids = torch.randint(len(tokenizer),ids.shape,device=device)
    ids[random_mask] = random_ids[random_mask]
    batch['labels'] = labels
    return batch


def weights(model):
    return {name:parameter.detach().cpu() for name,parameter in model.named_parameters() if parameter.requires_grad}


def restore_weights(model,state):
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name,value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def atomic_save(value,path):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def validation_loss(model,tokenizer,groups,task,batch_size,seed):
    model.eval()
    total,count = 0.0,0
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
        torch.manual_seed(seed)
        for version in sorted(groups):
            rows = groups[version]
            for start in range(0,len(rows),batch_size):
                batch = make_batch(rows[start:start+batch_size],tokenizer,'cuda',task)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss = model(**batch).loss
                n = int((batch['labels']!=-100).sum())
                total += float(loss)*n
                count += n
    model.train()
    return total/count


def train_stage(model,tokenizer,train,validation,task,directory,steps,lr,args,seed):
    directory.mkdir(parents=True,exist_ok=True)
    complete = directory/'complete.json'
    best_path = directory/'best-parameters.pt'
    if complete.exists():
        restore_weights(model,torch.load(best_path,map_location='cpu',weights_only=True))
        return json.loads(complete.read_text())
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=lr,weight_decay=0.01,fused=True)
    sampler = BalancedSampler(train,seed)
    start,best = 0,float('inf')
    checkpoint = directory/'last.pt'
    history = []
    torch.manual_seed(seed)
    if checkpoint.exists():
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['steps'] != steps or state['task'] != task:
            raise ValueError('Resume configuration mismatch')
        restore_weights(model,state['weights'])
        optimizer.load_state_dict(state['optimizer'])
        sampler.restore(state['sampler'])
        start,best,history = state['step'],state['best'],state['history']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    started = time.monotonic()
    model.train()
    for step in range(start,steps):
        optimizer.zero_grad(set_to_none=True)
        training_loss = 0.0
        for micro in range(args.accumulation):
            batch = make_batch(sampler.take(args.batch_size),tokenizer,'cuda',task)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = model(**batch).loss/args.accumulation
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite '+task+' loss at step '+str(step+1))
            loss.backward()
            training_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(parameters,1.0)
        fraction = min((step+1)/max(1,steps*0.06),1.0)*max(0.0,1-step/max(1,steps))
        for group in optimizer.param_groups:
            group['lr'] = lr*fraction
        optimizer.step()
        if (step+1)%25 == 0 or step+1 == steps:
            print(json.dumps({'stage':str(directory),'task':task,'step':step+1,'steps':steps,'loss':training_loss,'epoch':round(sampler.epoch+sampler.position/(sampler.size*len(sampler.versions)),3),'seconds':round(time.monotonic()-started,1),'peak_cuda_bytes':torch.cuda.max_memory_allocated()}),flush=True)
        if (step+1)%args.eval_every == 0 or step+1 == steps:
            value = validation_loss(model,tokenizer,validation,task,args.batch_size,seed+999)
            record = {'step':step+1,'validation_loss':value,'seconds':round(time.monotonic()-started,2)}
            history.append(record)
            if value < best:
                best = value
                atomic_save(weights(model),best_path)
            atomic_save({'step':step+1,'steps':steps,'task':task,'weights':weights(model),'optimizer':optimizer.state_dict(),'sampler':sampler.state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'best':best,'history':history},checkpoint)
            (directory/'progress.json').write_text(json.dumps({'step':step+1,'steps':steps,'best_validation_loss':best,'history':history},indent=2)+'\n')
            print(json.dumps({'stage':str(directory),'validation':record,'best':best}),flush=True)
    restore_weights(model,torch.load(best_path,map_location='cpu',weights_only=True))
    result = {'task':task,'steps':steps,'examples_seen':steps*args.batch_size*args.accumulation,'unique_training_rows':sum(map(len,train.values())),'epochs_equivalent':steps*args.batch_size*args.accumulation/sum(map(len,train.values())),'best_validation_loss':best,'selected_step':min(history,key=lambda item:item['validation_loss'])['step'],'validation':history,'seconds_this_process':time.monotonic()-started}
    complete.write_text(json.dumps(result,indent=2)+'\n')
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def segmentation_tokenizer(rows,path):
    if (path/'tokenizer.json').exists():
        return PreTrainedTokenizerFast.from_pretrained(path)
    backend = Tokenizer(models.WordPiece(unk_token='[UNK]'))
    backend.normalizer = normalizers.Sequence([normalizers.NFD(),normalizers.StripAccents()])
    backend.pre_tokenizer = pre_tokenizers.Split('<SEP>','removed')
    backend.train_from_iterator((row['bytecode'] for row in rows),trainer=trainers.WordPieceTrainer(vocab_size=30000,special_tokens=['[UNK]','[PAD]','[CLS]','[SEP]','[MASK]']))
    backend.decoder = decoders.WordPiece(prefix='##')
    backend.post_processor = processors.TemplateProcessing(single='[CLS]:0 $A:0 [SEP]:0',pair='[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1',special_tokens=[('[CLS]',backend.token_to_id('[CLS]')),('[SEP]',backend.token_to_id('[SEP]'))])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',pad_token='[PAD]',cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]',model_max_length=512)
    tokenizer.save_pretrained(path)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=['statement','segmentation'],required=True)
    parser.add_argument('--role',choices=['universal','control314','control315'],required=True)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--adapter-epochs',type=int,default=8)
    parser.add_argument('--mlm-epochs',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--accumulation',type=int,default=2)
    parser.add_argument('--eval-every',type=int,default=250)
    parser.add_argument('--rank',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261016)
    args = parser.parse_args()
    if min(args.epochs,args.adapter_epochs,args.mlm_epochs,args.batch_size,args.accumulation,args.eval_every,args.rank)<1:
        raise ValueError('All training counts must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    family_path = args.output/'family.json'
    if family_path.exists() and json.loads(family_path.read_text()).get('complete'):
        print(json.dumps({'already_complete':str(args.output)}),flush=True)
        return
    overall = json.loads((args.data/'manifest.json').read_text())
    if not overall.get('complete') or overall['examples'] != 50000:
        raise ValueError('Complete 50k dataset required')
    subset = args.data/('backbone' if args.role == 'universal' else args.role)
    rows = read_rows(subset/(args.task+'-train.jsonl'))
    valid = read_rows(subset/(args.task+'-validation.jsonl'))
    versions = tuple(VERSIONS) if args.role == 'universal' else ('3.14' if args.role == 'control314' else '3.15',)
    if {r['version'] for r in rows} != set(versions):
        raise ValueError('Training-version leak or missing version')
    train_hashes = {r['canonical_sha256'] for r in rows}
    if train_hashes & {r['canonical_sha256'] for r in valid}:
        raise ValueError('Training/validation source leakage')
    effective_batch = args.batch_size*args.accumulation
    universal_steps = math.ceil(40500*args.epochs/effective_batch)
    adapter_steps = math.ceil(4500*args.adapter_epochs/effective_batch)
    mlm_steps = math.ceil(40500*args.mlm_epochs/effective_batch)
    steps = universal_steps if args.role == 'universal' else universal_steps+adapter_steps
    family = {'format':1,'task':args.task,'backbone_id':str(uuid.uuid4()),'versions':['3.14','3.15'] if args.role == 'universal' else list(versions),'backbone_training_versions':list(versions),'adapter_training_versions':['3.14','3.15'] if args.role == 'universal' else [],'source_prefix':'Python {version}: ' if args.task == 'statement' else '', 'initial_model':'Salesforce/codet5-base' if args.task == 'statement' else 'fresh RoBERTa, task-local MLM pretraining','role':args.role,'complete':False,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'data_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),'protocol':{'validation_selected':True,'full_coverage_before_reshuffle':True,'matched_control_optimizer_updates':steps if args.role != 'universal' else universal_steps+adapter_steps,'generic_neural_initialization':True,'training_rows':len(rows),'validation_rows':len(valid)}}
    if family_path.exists():
        previous = json.loads(family_path.read_text())
        if previous['data_manifest_sha256'] != family['data_manifest_sha256'] or previous['role'] != args.role:
            raise ValueError('Existing model family provenance mismatch')
        family['backbone_id'] = previous['backbone_id']
    if args.task == 'statement':
        tokenizer = load_statement_tokenizer('Salesforce/codet5-base')
        tokenizer.add_tokens(['<mask_'+str(i)+'>' for i in range(300)])
        tokenizer.save_pretrained(args.output/'tokenizer')
        model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
        family['initial_model_commit'] = getattr(model.config, '_commit_hash', None)
        model.resize_token_embeddings(len(tokenizer))
        for config in (model.config,model.generation_config):
            config.pad_token_id = tokenizer.pad_token_id
            config.eos_token_id = tokenizer.eos_token_id
            config.decoder_start_token_id = tokenizer.pad_token_id
        targets = ('q','v')
        train = encode_groups(args.task,tokenizer,rows)
        validation = encode_groups(args.task,tokenizer,valid)
    else:
        tokenizer = segmentation_tokenizer(rows,args.output/'tokenizer')
        config = RobertaConfig(vocab_size=len(tokenizer),hidden_size=768,num_hidden_layers=12,num_attention_heads=12,intermediate_size=3072,max_position_embeddings=514,type_vocab_size=1,pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.cls_token_id,eos_token_id=tokenizer.sep_token_id)
        train = encode_groups(args.task,tokenizer,rows)
        validation = encode_groups(args.task,tokenizer,valid)
        mlm = RobertaForMaskedLM(config).to('cuda')
        family['mlm_training'] = train_stage(mlm,tokenizer,train,validation,'mlm',args.output/'training/mlm',mlm_steps,1e-4,args,args.seed)
        config.num_labels = 3
        config.id2label = {0:'B',1:'I',2:'E'}
        config.label2id = LABELS
        torch.manual_seed(args.seed)
        model = RobertaForTokenClassification(config).to('cuda')
        model.roberta.load_state_dict(mlm.roberta.state_dict())
        del mlm
        gc.collect()
        torch.cuda.empty_cache()
        targets = ('query','value')
    model.to('cuda')
    model.config.use_cache = False
    family['targets'] = targets
    family['backbone_parameters'] = sum(p.numel() for p in model.parameters())
    family_path.write_text(json.dumps(family,indent=2)+'\n')
    family['backbone_training'] = train_stage(model,tokenizer,train,validation,args.task,args.output/'training/backbone',steps,5e-5,args,args.seed)
    model.config.use_cache = True
    model.save_pretrained(args.output/'backbone')
    family['adapters'] = {}
    family['adapter_training'] = {}
    if args.role == 'universal':
        bank = AdapterBank(model,family['backbone_id'],targets)
        for index,version in enumerate(('3.14','3.15')):
            adapter_subset = args.data/('backbone' if version == '3.14' else 'adapter315')
            adapter_rows = [r for r in read_rows(adapter_subset/(args.task+'-train.jsonl')) if r['version'] == version]
            adapter_valid = [r for r in read_rows(adapter_subset/(args.task+'-validation.jsonl')) if r['version'] == version]
            adapter_train = encode_groups(args.task,tokenizer,adapter_rows)
            adapter_validation = encode_groups(args.task,tokenizer,adapter_valid)
            bank.add(version,rank=args.rank,alpha=2*args.rank,dropout=0.05)
            parameters = bank.trainable(version)
            model.config.use_cache = False
            family['adapter_training'][version] = train_stage(model,tokenizer,adapter_train,adapter_validation,args.task,args.output/('training/adapter'+version),adapter_steps,3e-4,args,args.seed+index+1)
            bank.save(version,args.output/'adapters'/version)
            family['adapters'][version] = 'adapters/'+version
            family['adapter_parameters_per_version'] = sum(p.numel() for p in parameters)
            family_path.write_text(json.dumps(family,indent=2)+'\n')
    family['complete'] = True
    family['peak_allocated_cuda_bytes'] = torch.cuda.max_memory_allocated()
    family_path.write_text(json.dumps(family,indent=2)+'\n')
    print(json.dumps({'complete':True,'output':str(args.output),'task':args.task,'role':args.role}),flush=True)


if __name__ == '__main__':
    main()
