"""Wait for complete file expansion, then continue the 3.14 LoRAs sequentially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--epochs',type=int,default=3)
args=parser.parse_args()
manifest=args.root/'data-lora314-plus50k/manifest.json'
started=time.monotonic()
while not manifest.exists():
    if time.monotonic()-started>6*60*60:
        raise TimeoutError('Preparation did not finish within six hours')
    print(json.dumps({'waiting_for_complete_50000_new_files':True,'seconds':round(time.monotonic()-started)}),flush=True)
    time.sleep(60)
metadata=json.loads(manifest.read_text())
if not metadata.get('complete') or metadata.get('new_training_files')!=50000 or metadata.get('total_training_files')!=54500:
    raise ValueError('Unexpected expansion dataset')
for task in ('segmentation','statement'):
    command=[sys.executable,'-u','-m','dev_scripts.continue_lora_training','--models',str(args.root/'models-50k'),'--data',str(args.root/'data-lora314-plus50k'),'--output',str(args.root/'models-lora314-plus50k'),'--task',task,'--epochs',str(args.epochs)]
    print(json.dumps({'starting_task':task,'command':command}),flush=True)
    subprocess.run(command,cwd=args.root/'repo',check=True)
print(json.dumps({'continued_314_adapters_complete':True,'models':str(args.root/'models-lora314-plus50k')}),flush=True)
