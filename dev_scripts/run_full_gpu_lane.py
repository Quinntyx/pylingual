"""Persistently run a two-GPU task lane, preserving full-run batch semantics."""
import argparse,json,os
from pathlib import Path
import subprocess,sys,time

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--lane',choices=['statement','segmentation'],required=True);p.add_argument('--prepare-pid',type=int,required=True)
 p.add_argument('--gpus');p.add_argument('--original-world-size',type=int)
 a=p.parse_args();run=a.root/'full-retrain-20261020';dataset=run/'data'/a.lane
 gpus=(a.gpus or ('0,1' if a.lane=='statement' else '2,3')).split(',')
 if len(gpus)!=2 or len(set(gpus))!=2 or not all(g.isdigit() for g in gpus):raise ValueError('Two distinct GPU IDs required')
 state=run/(a.lane+'-lane.json')
 def note(phase,**kwargs):
  pending=state.with_suffix('.json.tmp');pending.write_text(json.dumps({'phase':phase,'pid':os.getpid(),'time':time.time(),'gpus':gpus,'world_size':2,**kwargs},indent=2)+'\n');pending.replace(state)
 def command(task,initialization=None):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus))
  batch=36 if task=='statement' else 14
  argv=[str(a.root/'repo/.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc_per_node=2','-m','dev_scripts.train_universal_models']
  argv+=['--task',task,'--data',str(dataset),'--tokenizer',str(a.root/'tokenizers'/a.lane),'--output',str(run/'universal'/task),'--batch-size',str(batch)]
  if initialization:argv+=['--initialization',str(initialization)]
  provenance=run/'universal'/task/'provenance.json'
  if provenance.exists():
   argv+=['--resume'];previous=json.loads(provenance.read_text());old=previous['training_arguments']
   old_world=previous.get('world_size',a.original_world_size)
   if old['per_device_train_batch_size']!=batch or old['gradient_accumulation_steps']!=1 or (old_world is not None and old_world!=2):
    if old_world is None:raise ValueError('Legacy checkpoint needs an explicit original world size')
    argv+=['--resume-original-world-size',str(old_world)]
  note('training-'+task,argv=argv);subprocess.run(argv,env=env,check=True)
 try:
  note('waiting-for-data',prepare_pid=a.prepare_pid)
  while not (dataset/'manifest.json').exists():
   if not Path('/proc/'+str(a.prepare_pid)).exists():raise RuntimeError('Preparation exited without a complete manifest')
   time.sleep(10)
  if a.lane=='statement':command('statement',a.root/'gpu-smoke-20261020/generic-codet5')
  else:
   if not (run/'universal/mlm/result.json').exists():command('mlm')
   if not (run/'universal/segmentation/result.json').exists():command('segmentation',run/'universal/mlm/checkpoint')
  note('complete')
 except BaseException as error:
  note('failed',error=repr(error));raise
if __name__=='__main__':main()
