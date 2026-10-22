"""Read-only three-way segmentation smoke on a fixed v1/3.14 subset."""
from __future__ import annotations
import argparse, contextlib, gc, hashlib, json, random, time
from collections import Counter
from pathlib import Path
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast
from dev_scripts.benchmark_full_retrain import selection
from pylingual.editable_bytecode import PYCFile
from pylingual.masking.model_disasm import create_global_masker
from pylingual.shared_models import load_family, load_version
from pylingual.opcodes import SEPARATOR, instruction_token_indices


def extract(file):
    pyc=PYCFile(str(file)); masker=create_global_masker(pyc); examples=[]; seen=set()
    if tuple(pyc.version[:2]) != (3,14): raise ValueError('Not Python 3.14')
    for bc in pyc.iter_bytecodes():
        if bc.is_comprehension: continue
        groups=bc.get_lno_insts(previously_seen_lines=seen); seen.update(groups)
        instructions=[]; labels=[]
        for insts in groups.values():
            if not insts: continue
            instructions.extend(masker.get_model_view(inst) for inst in insts)
            labels.extend(['B'] if len(insts)==1 else ['B']+['I']*(len(insts)-2)+['E'])
        if instructions: examples.append((instructions,labels))
    return examples


def common_windows(instructions,labels,tokenizers):
    start=0
    while start<len(instructions):
        end=min(start+128,len(instructions))
        while any(len(tokenizer(SEPARATOR.join(instructions[start:end]))['input_ids'])>512 for tokenizer in tokenizers):
            if end-start==1: raise ValueError('One instruction exceeds common 512-token budget')
            end=start+max(1,(end-start)//2)
        yield SEPARATOR.join(instructions[start:end]), labels[start:end]
        start=end


def positions(tokenizer,text,expected):
    feature=tokenizer(text,return_offsets_mapping=True)
    spans=[]; cursor=0
    for instruction in text.split(SEPARATOR):
        spans.append((cursor,cursor+len(instruction)));cursor+=len(instruction)+len(SEPARATOR)
    first=[]
    for lo,hi in spans:
        hits=[i for i,(a,b) in enumerate(feature['offset_mapping']) if b>a and b>lo and a<hi]
        if not hits: raise ValueError('Instruction has no token')
        first.append(hits[0])
    if len(first)!=len(expected): raise ValueError('Instruction alignment mismatch')
    feature.pop('offset_mapping')
    return feature,first


def score(name,model,tokenizer,windows,output,bank=None,version=None,registry=False):
    confusion={truth:{pred:0 for pred in 'BIE'} for truth in 'BIE'}; exact=0; count=0
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    guard=bank.use(version) if bank is not None else contextlib.nullcontext()
    model.eval()
    with guard, torch.inference_mode():
        for offset in range(0,len(windows),8):
            selected=windows[offset:offset+8]; features=[]; firsts=[]
            for text,truth in selected:
                if bank is not None:
                    text=bank.bind_bytecode(text, '3.14')
                feature,first=positions(tokenizer,text,truth)
                if registry:
                    encoded,indices=instruction_token_indices(tokenizer,text)
                    actual={}
                    for position,index in enumerate(indices):
                        if index is not None: actual.setdefault(index,position)
                    if first!=[actual[i] for i in range(len(truth))]: raise ValueError('Runtime alignment differs')
                    if encoded['input_ids']!=feature['input_ids']: raise ValueError('Runtime encoding differs')
                features.append(feature); firsts.append(first)
            batch=tokenizer.pad(features,padding=True,return_tensors='pt').to('cuda:0')
            predictions=model(**batch).logits.argmax(-1).cpu().tolist()
            for (_,truth),first,prediction in zip(selected,firsts,predictions):
                predicted=[model.config.id2label[prediction[index]] for index in first]
                if any(label not in 'BIE' for label in predicted): raise ValueError('Unexpected label mapping')
                exact+=int(predicted==truth);count+=len(truth)
                for actual,pred in zip(truth,predicted): confusion[actual][pred]+=1
    torch.cuda.synchronize(); seconds=time.monotonic()-started
    tp=confusion['B']['B'];fp=sum(confusion[t]['B'] for t in 'IE');fn=sum(confusion['B'][p] for p in 'IE')
    result={'instructions':count,'windows':len(windows),'accuracy':sum(confusion[t][t] for t in 'BIE')/count,
        'boundary_precision':tp/max(1,tp+fp),'boundary_recall':tp/max(1,tp+fn),
        'boundary_f1':2*tp/max(1,2*tp+fp+fn),'exact_windows':exact/len(windows),
        'confusion':confusion,'seconds':seconds,'peak_cuda_MiB':torch.cuda.max_memory_allocated()/2**20}
    (output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'arm':name,**result}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--files',type=int,default=100);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if torch.cuda.device_count()!=1: raise ValueError('Restrict to idle physical GPU2')
    api=HfApi();model_repo='syssec-utd/py314-pylingual-v4-segmenter';tok_repo='syssec-utd/py314-pylingual-v4-tokenizer'
    model_sha=api.model_info(model_repo).sha;tok_sha=api.model_info(tok_repo).sha
    legacy_tokenizer=PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(tok_repo,'tokenizer.json',revision=tok_sha),
        unk_token='[UNK]',pad_token='[PAD]',cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]')
    legacy=AutoModelForTokenClassification.from_pretrained(model_repo,revision=model_sha).to('cuda:0')
    family=a.run/'adapter-runs/segmentation/3.14-attempt1'
    bank,tokenizer,metadata,_=load_family(family,'segmentation',torch.device('cuda:0'))
    load_version(family,'segmentation','3.14',torch.device('cuda:0'))
    manifest=selection(a.run,'3.14','v1'); eligible=[row for row in manifest['files'] if not row['known_training_or_validation_source_overlap']]
    random.Random(20261021).shuffle(eligible); chosen=eligible[:a.files]
    windows=[];failures=[];codeobjects=0;selected=[];dataset=a.run/'independent-benchmark/dataset-v1.1.0'
    for row in chosen:
        try:
            examples=extract(dataset/row['relative']);file_windows=[]
            for instructions,labels in examples:
                file_windows.extend(common_windows(instructions,labels,[legacy_tokenizer,lambda text: tokenizer(bank.bind_bytecode(text,'3.14'))]))
            if not file_windows: raise ValueError('No nonempty code objects')
            windows.extend(file_windows);codeobjects+=len(examples);selected.append(row)
        except Exception as error: failures.append({'file':row['relative'],'error':repr(error)})
    if not windows: raise ValueError('No windows extracted')
    provenance={'dataset':'v1.1.0','version':'3.14','selection_seed':20261021,'requested_files':a.files,'evaluated_files':len(selected),
        'codeobjects':codeobjects,'windows':len(windows),'files':selected,'failures':failures,'dev_model':model_repo,'dev_model_revision':model_sha,
        'dev_tokenizer':tok_repo,'dev_tokenizer_revision':tok_sha,'family':str(family),'adapter_updates':metadata.get('adapter_updates'),
        'policy':'same common nonoverlapping windows, <=512 tokens under each tokenizer; first-token instruction B/I/E labels from pyc line groups; no training; known training/validation SHA overlap excluded; deployed-model training overlap unknown',
        'precision':'FP32','gpu':torch.cuda.get_device_name(0)}
    (a.output/'selection.json').write_text(json.dumps(provenance,indent=2)+'\n')
    results={};results['dev']=score('dev',legacy,legacy_tokenizer,windows,a.output)
    del legacy;gc.collect();torch.cuda.empty_cache()
    results['base']=score('base',bank.model,tokenizer,windows,a.output,bank,None,True)
    results['adapted']=score('adapted',bank.model,tokenizer,windows,a.output,bank,'3.14',True)
    (a.output/'report.json').write_text(json.dumps({'complete':True,'provenance':provenance,'results':results},indent=2)+'\n')
if __name__=='__main__':main()
