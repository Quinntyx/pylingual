"""Prepare full per-version adapter/control inputs without using GPU model capacity."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root;run=root/'full-retrain-20261020';state=run/'variant-preparation.json'
 def note(phase,**more):
  temporary=state.with_suffix('.tmp');temporary.write_text(json.dumps({'phase':phase,'pid':os.getpid(),'time':time.time(),**more},indent=2)+'\n');temporary.replace(state)
 def execute(name,argv):
  log=run/'logs'/('prepare-'+name+'.log');log.parent.mkdir(exist_ok=True)
  with log.open('a') as stream:
   subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,check=True)
  return name
 try:
  note('waiting-for-target-sources')
  while not (run/'target-corpora/complete.json').exists(): time.sleep(30)
  complete=json.loads((run/'target-corpora/complete.json').read_text())
  if not complete['complete'] or complete['smoke_only'] or complete['files']!=110000: raise ValueError('Incomplete target source selection')
  # Allocate only genuinely observed new instruction names, not unused compiler pseudo-ops.
  known=set(json.loads((root/'tokenizers/statement/opcode_registry.json').read_text())['known_opcodes'])
  if known!=set(json.loads((root/'tokenizers/segmentation/opcode_registry.json').read_text())['known_opcodes']):
   raise ValueError('Task opcode registries disagree')
  observed_bindings={}
  for version in ('3.14','3.15'):
   manifest=json.loads((run/'target-corpora'/version/'train/manifest.json').read_text())
   unknown=sorted(set(manifest['observed_opcodes'])-known)
   if len(unknown)>16: raise ValueError('Reserved opcode capacity exceeded')
   bindings={name:index for index,name in enumerate(unknown)}
   (run/(version+'-bindings.json')).write_text(json.dumps(bindings,indent=2)+'\n')
   observed_bindings[version]=bindings
  (run/'observed-version-bindings.json').write_text(json.dumps(observed_bindings,indent=2)+'\n')
  note('fitting-ordinary-tokenizers',bindings=observed_bindings)
  for version in ('3.14','3.15'):
   execute('tokenizers-'+version,[sys.executable,'-u','-m','dev_scripts.train_control_tokenizers','--data',str(run/'target-corpora'/version/'train'),'--output',str(run/'control-tokenizers'/version),'--template',str(root/'gpu-smoke-20261020/generic-codet5')])
  requests=[]
  for scope in ('adapter','control'):
   for version in ('3.14','3.15'):
    for task in ('statement','segmentation'):
     for split in ('train','validation'):
      name='-'.join((scope,version,task,split));destination=run/'encoded'/scope/version/task/split
      tokenizer=root/'tokenizers'/task if scope=='adapter' else run/'control-tokenizers'/version/task
      argv=[sys.executable,'-u','-m','dev_scripts.prepare_full_training_data','--scope',scope,'--data',str(run/'target-corpora'/version/split),'--tokenizer',str(tokenizer),'--task',task,'--output',str(destination),'--workers','4']
      if scope=='adapter': argv+=['--bindings-file',str(run/(version+'-bindings.json'))]
      requests.append((name,argv))
  done=[];note('encoding',completed=done,total=len(requests))
  with ThreadPoolExecutor(max_workers=2) as pool:
   futures={pool.submit(execute,name,argv):name for name,argv in requests}
   for future in as_completed(futures):
    done.append(future.result());note('encoding',completed=sorted(done),total=len(requests))
  note('complete',completed=sorted(done),total=len(requests))
 except BaseException as error:
  note('failed',error=repr(error));raise
if __name__=='__main__':main()
