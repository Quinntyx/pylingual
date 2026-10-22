"""Sequential, resumable full ablation on one GPU; waits for remote-only data prep."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--accumulation',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--adapter-epochs',type=int,default=8)
    parser.add_argument('--mlm-epochs',type=int,default=5)
    parser.add_argument('--eval-every',type=int,default=250)
    parser.add_argument('--wait-for-data',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = args.data/'manifest.json'
    while not manifest.exists():
        if not args.wait_for_data:
            raise FileNotFoundError(manifest)
        print(json.dumps({'waiting_for_dataset':str(manifest)}),flush=True)
        time.sleep(30)
    dataset = json.loads(manifest.read_text())
    if not dataset.get('complete') or dataset['examples'] != 50000:
        raise RuntimeError('Expected completed 50k dataset')
    repo = Path(__file__).resolve().parents[1]
    scripts = ['dev_scripts/full_ablation_training.py','dev_scripts/shared_training.py','pylingual/lora.py','pylingual/masking/ast_masker.py']
    status = {'format':1,'pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'data_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'code_sha256':{script:hashlib.sha256((repo/script).read_bytes()).hexdigest() for script in scripts},'torch_version':importlib.metadata.version('torch'),'transformers_version':importlib.metadata.version('transformers'),'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'hardware':subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip(),'stages':[],'complete':False}
    status_path = args.output/'study.json'
    for task in ('statement','segmentation'):
        for role in ('universal','control314','control315'):
            output = args.output/role/task
            command = [sys.executable,'-u','-m','dev_scripts.full_ablation_training','--data',str(args.data),'--output',str(output),'--task',task,'--role',role]
            for name in ('batch_size','accumulation','epochs','adapter_epochs','mlm_epochs','eval_every'):
                command.extend(['--'+name.replace('_','-'),str(getattr(args,name))])
            stage = {'task':task,'role':role,'output':str(output),'command':command,'state':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            status['stages'].append(stage)
            status_path.write_text(json.dumps(status,indent=2)+'\n')
            print(json.dumps({'starting_stage':stage}),flush=True)
            started = time.monotonic()
            result = subprocess.run(command,cwd=repo,env={**os.environ,'PYTHONPATH':str(repo),'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'})
            stage.update(exit_code=result.returncode,seconds=time.monotonic()-started,state='complete' if result.returncode==0 else 'failed')
            status_path.write_text(json.dumps(status,indent=2)+'\n')
            if result.returncode:
                raise SystemExit(result.returncode)
            family = json.loads((output/'family.json').read_text())
            if not family.get('complete'):
                raise RuntimeError('Training exited without a complete model family')
    status['complete'] = True
    status['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    status_path.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({'study_training_complete':True,'output':str(args.output)}),flush=True)


if __name__ == '__main__':
    main()
