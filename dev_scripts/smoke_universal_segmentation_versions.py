"""Frozen universal segmenter, no adapters, matched v1 sources across Python 3.6-3.15."""
from __future__ import annotations
import argparse, hashlib, json, random, shutil, signal, subprocess, time
from collections import Counter
from pathlib import Path
import torch
from dev_scripts.benchmark_full_retrain import training_audit
from pylingual.editable_bytecode import PYCFile
from pylingual.masking.model_disasm import create_global_masker
from pylingual.shared_models import load_family
from pylingual.opcodes import SEPARATOR, instruction_token_indices

VERSIONS=tuple('3.'+str(i) for i in range(6,16))


def extract(file,version):
    pyc=PYCFile(str(file));masker=create_global_masker(pyc);examples=[];seen=set()
    if tuple(pyc.version[:2])!=tuple(map(int,version.split('.'))):raise ValueError('Wrong pyc version')
    for bc in pyc.iter_bytecodes():
        if bc.is_comprehension:continue
        groups=bc.get_lno_insts(previously_seen_lines=seen);seen.update(groups)
        instructions=[];labels=[]
        for insts in groups.values():
            if not insts:continue
            instructions.extend(masker.get_model_view(inst) for inst in insts)
            labels.extend(['B'] if len(insts)==1 else ['B']+['I']*(len(insts)-2)+['E'])
        if instructions:examples.append((instructions,labels))
    return examples


def windows(instructions,labels,bank,tokenizer,version):
    start=0
    while start<len(instructions):
        end=min(start+128,len(instructions))
        while True:
            bound=bank.bind_bytecode(SEPARATOR.join(instructions[start:end]),version)
            encoded,indices=instruction_token_indices(tokenizer,bound)
            if len(encoded['input_ids'])<=512:break
            if end-start==1:raise ValueError('Instruction exceeds 512-token limit')
            end=start+max(1,(end-start)//2)
        first={}
        for position,index in enumerate(indices):
            if index is not None:first.setdefault(index,position)
        if set(first)!=set(range(end-start)):raise ValueError('Runtime instruction alignment failed')
        yield dict(encoded),[first[i] for i in range(end-start)],labels[start:end]
        start=end


def evaluate(bank,tokenizer,requests):
    confusion={t:{p:0 for p in 'BIE'} for t in 'BIE'};exact=0;count=0
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    with bank.use(None),torch.inference_mode():
        for offset in range(0,len(requests),32):
            chosen=requests[offset:offset+32]
            batch=tokenizer.pad([row[0] for row in chosen],padding=True,return_tensors='pt').to('cuda:0')
            predictions=bank.model(**batch).logits.argmax(-1).cpu().tolist()
            for (_,first,truth),prediction in zip(chosen,predictions):
                predicted=[bank.model.config.id2label[prediction[index]] for index in first]
                if any(label not in {'B','I','E'} for label in predicted):raise ValueError('Invalid label mapping')
                exact+=int(predicted==truth);count+=len(truth)
                for actual,pred in zip(truth,predicted):confusion[actual][pred]+=1
    torch.cuda.synchronize();seconds=time.monotonic()-started
    tp=confusion['B']['B'];fp=sum(confusion[t]['B'] for t in 'IE');fn=sum(confusion['B'][p] for p in 'IE')
    return {'instructions':count,'windows':len(requests),'accuracy':sum(confusion[t][t] for t in 'BIE')/count,
        'boundary_precision':tp/max(1,tp+fp),'boundary_recall':tp/max(1,tp+fn),
        'boundary_f1':2*tp/max(1,2*tp+fp+fn),'exact_windows':exact/len(requests),
        'confusion':confusion,'seconds':seconds,'peak_cuda_MiB':torch.cuda.max_memory_allocated()/2**20}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--files',type=int,default=1000);p.add_argument('--versions',nargs='+',choices=VERSIONS,default=list(VERSIONS));p.add_argument('--skip-checkout',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if torch.cuda.device_count()!=1:raise ValueError('Restrict to physical GPU2')
    dataset=a.run/'independent-benchmark/dataset-v1.1.0'
    origin=subprocess.check_output(['git','-C',str(dataset),'remote','get-url','origin'],text=True).strip()
    if 'github.com' not in origin or 'syssec-utd/pylingual_eval_dataset' not in origin:raise ValueError('Unexpected dataset origin')
    commit=subprocess.check_output(['git','-C',str(dataset),'rev-parse','HEAD'],text=True).strip()
    excluded,source_audit=training_audit(a.run);listings={};common=None
    for version in VERSIONS:
        listing=version.replace('.','')+'-pyc-list.txt'
        content=subprocess.check_output(['git','-C',str(dataset),'show','HEAD:'+listing],text=True)
        by_hash={Path(relative).parent.name:relative for relative in content.splitlines() if relative.strip()}
        listings[version]=by_hash;eligible=set(by_hash)-excluded
        common=eligible if common is None else common&eligible
    identities=sorted(common);random.Random(20261021).shuffle(identities)
    if len(identities)<a.files:raise ValueError('Insufficient common nonoverlap sources')
    identities=identities[:a.files]
    # Whole version directories avoid quadratic matching of thousands of sparse patterns.
    # Only the pinned 1000 matched sources/version are evaluated; additional pycs stay in the remote cache.
    patterns=['/README.md']
    for version in VERSIONS:
        patterns.extend(['/'+version.replace('.','')+'-pyc-list.txt','/python-'+version+'/'])
    sparse_path=dataset/'.git/info/sparse-checkout'
    if shutil.which('chezmoi'):
        managed=subprocess.run(['chezmoi','source-path','--',str(sparse_path)],capture_output=True,text=True)
        if managed.returncode==0:raise ValueError('Unexpected chezmoi-managed dataset runtime clone')
    print(json.dumps({'phase':'fetching_selected_pyc_blobs','files_per_version':a.files,'patterns':len(patterns),'configuration_scope':'unmanaged transient dataset clone; preserve tag and existing 3.14/3.15 checkout'}),flush=True)
    if not a.skip_checkout:
        subprocess.run(['git','-C',str(dataset),'sparse-checkout','set','--no-cone','--stdin'],input='\n'.join(patterns)+'\n',text=True,check=True,timeout=900)
    family=a.run/'families-base/segmentation'
    bank,tokenizer,metadata,_=load_family(family,'segmentation',torch.device('cuda:0'));bank.model.eval()
    if bank.specs:raise ValueError('Unexpected adapters loaded')
    baseline_hash=hashlib.sha256((family/'backbone/model.safetensors').read_bytes()).hexdigest()
    provenance={'dataset':'v1.1.0','dataset_commit':commit,'versions':VERSIONS,'evaluation_versions':list(a.versions),'files_per_version':a.files,'selection_seed':20261021,
        'source_identities':identities,'common_eligible_sources':len(common),'known_training_validation_overlap':0,
        'backbone':str(family),'backbone_sha256':baseline_hash,'family_metadata':metadata,'adapters_enabled':False,
        'precision':'FP32','gpu':torch.cuda.get_device_name(0),'source_audit':source_audit,
        'policy':'Same source hashes across versions, all code objects except comprehensions, nonoverlapping <=128 instruction and <=512 token windows, production opcode bindings and first-token alignment; pyc line groups supply labels; not end-to-end decompilation.'}
    (a.output/'selection.json').write_text(json.dumps(provenance,indent=2)+'\n')
    results={}
    selected_versions=tuple(a.versions)
    def alarm_handler(signum,frame):raise TimeoutError('pyc preparation exceeded 10 seconds')
    signal.signal(signal.SIGALRM,alarm_handler)
    for version in selected_versions:
        started=time.monotonic();requests=[];failures=[];files=[];codeobjects=0
        for identity in identities:
            relative=listings[version][identity]
            try:
                signal.alarm(10);examples=extract(dataset/relative,version);file_requests=[]
                for instructions,labels in examples:file_requests.extend(windows(instructions,labels,bank,tokenizer,version))
                signal.alarm(0)
                if not file_requests:raise ValueError('No segmentation examples')
                requests.extend(file_requests);codeobjects+=len(examples);files.append({'source_sha512':identity,'relative':relative})
            except Exception as error:
                signal.alarm(0);failures.append({'source_sha512':identity,'relative':relative,'error':repr(error)})
        if not requests:raise ValueError('No examples for '+version)
        print(json.dumps({'phase':'inference','version':version,'files':len(files),'windows':len(requests)}),flush=True)
        result=evaluate(bank,tokenizer,requests);result.update(version=version,files=len(files),attempted_files=a.files,
            codeobjects=codeobjects,failures=failures,selected_files=files,preparation_and_inference_seconds=time.monotonic()-started)
        results[version]=result
        (a.output/(version+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({key:value for key,value in result.items() if key not in {'confusion','failures','selected_files'}}),flush=True)
    (a.output/'report.json').write_text(json.dumps({'complete':True,'provenance':provenance,'results':results},indent=2)+'\n')
if __name__=='__main__':main()
