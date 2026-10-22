"""Autonomous dependency/GPU scheduler for full adapters; ordinary controls require explicit opt-in."""
from __future__ import annotations
import argparse,json,math,os,shutil,subprocess,sys,time
from pathlib import Path

def specs(run, *, include_controls=False):
 jobs=[]
 for version in ('3.14','3.15'):
  for task in ('statement','segmentation'):
   jobs.append({'name':'adapter-'+version+'-'+task,'kind':'adapter','version':version,'task':task,'priority':100 if task=='statement' else 30})
  for task in (('statement','mlm','segmentation') if include_controls else ()):
   jobs.append({'name':'control-'+version+'-'+task,'kind':'control','version':version,'task':task,'priority':90 if task=='statement' else 70 if task=='mlm' else 40})
 return jobs

def reconcile_jobs(state,jobs):
 active={job['name'] for job in jobs};records=state.setdefault('jobs',{})
 for name in list(records):
  if name in active: continue
  record=records[name]
  if record.get('phase')=='running': raise RuntimeError('Cannot silently cancel an active job: '+name)
  state.setdefault('cancelled_jobs',{})[name]={**record,'phase':'cancelled','cancelled_at':time.time(),'reason':'Excluded from authorized study; ordinary controls require explicit opt-in'}
  del records[name]
 state['include_controls']=any(job['kind']=='control' for job in jobs)
 return state

def reserved_backbone_gpus(statement_ready,segmentation_ready):
 return (set() if statement_ready else {0,1}) | (set() if segmentation_ready else {2,3})

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--describe',action='store_true');p.add_argument('--include-controls',action='store_true',help='Only with explicit authorization for six additional full training stages');a=p.parse_args()
 root=a.root;run=root/'full-retrain-20261020';jobs=specs(run,include_controls=a.include_controls);state_path=run/'training-supervisor.json'
 if a.describe: print(json.dumps(jobs,indent=2));return
 state={'phase':'running','pid':os.getpid(),'jobs':{}}
 if state_path.exists(): state=json.loads(state_path.read_text());state.update(pid=os.getpid(),phase='running')
 reconcile_jobs(state,jobs)
 live={};streams={}
 def save():
  state['updated_at']=time.time();temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
 def load(path): return json.loads(path.read_text()) if path.exists() else {}
 def model_output(job): return run/'controls'/job['version']/job['task']
 def finished(job,record):
  if not record.get('output'): return False
  directory=Path(record['output'])
  if job['kind']=='control':
   result=load(directory/'result.json')
   if not result.get('complete'): return False
   if result.get('smoke_only') or result.get('scope')!='control' or result['files']!=54500 or result.get('epoch',0)<1.999:
    raise ValueError('Invalid full control completion:'+job['name'])
   return True
  metadata=load(directory/'family.json')
  if not metadata.get('complete'): return False
  update=metadata.get('adapter_updates',{}).get(job['version'],{})
  if metadata.get('smoke_only') or update.get('files')!=54500 or update.get('epochs')!=2 or update.get('training',{}).get('epoch',0)<1.999:
   raise ValueError('Invalid full adapter completion:'+job['name'])
  return True
 def universal_ready(task):
  lane='statement' if task=='statement' else 'segmentation'
  return load(run/(lane+'-lane.json')).get('phase')=='complete' and load(run/'universal'/task/'result.json').get('complete',False)
 def publish_bases():
  for task in ('statement','segmentation'):
   output=run/'families-base'/task
   if universal_ready(task) and not load(output/'family.json').get('complete'):
    from dev_scripts.publish_shared_backbone import publish
    publish(run/'universal'/task/'checkpoint',root/'tokenizers'/task,task,output,training_versions=['3.6','3.7','3.8','3.9','3.10','3.11','3.12','3.13','3.14'])
 def ready(job):
  task='statement' if job['task']=='statement' else 'segmentation'
  data=run/'encoded'/job['kind']/job['version']/task/'train'
  if not load(data/'manifest.json').get('complete'): return False
  if job['kind']=='adapter':
   return load(run/'families-base'/job['task']/'family.json').get('complete',False) and load(data.parent/'validation/manifest.json').get('complete',False)
  if job['task']=='segmentation': return load(run/'controls'/job['version']/'mlm/result.json').get('complete',False)
  return True
 def argv_for(job,record):
  version,task=job['version'],job['task'];encoded_task='statement' if task=='statement' else 'segmentation'
  data=run/'encoded'/job['kind']/version/encoded_task
  if job['kind']=='control':
   output=model_output(job);record['output']=str(output)
   argv=[sys.executable,'-u','-m','dev_scripts.train_universal_models','--task',task,'--ordinary-version',version,
    '--data',str(data/'train'),'--tokenizer',str(run/'control-tokenizers'/version/encoded_task),'--output',str(output),
    '--accumulation','3' if task=='statement' else '1']
   if task=='statement': argv+=['--initialization',str(root/'gpu-smoke-20261020/generic-codet5')]
   if task=='segmentation': argv+=['--initialization',str(run/'controls'/version/'mlm/checkpoint')]
   if (output/'provenance.json').exists(): argv+=['--resume']
   return argv
  output=run/'adapter-runs'/task/(version+'-attempt'+str(record['attempts']));record['output']=str(output)
  selection=run/'target-corpora'/version
  argv=[sys.executable,'-u','-m','dev_scripts.train_shared_adapter','--family',str(run/'families-base'/task),
   '--data',str(selection/'train/accepted.jsonl'),'--validation-data',str(selection/'validation/accepted.jsonl'),
   '--encoded-data',str(data/'train'),'--encoded-validation',str(data/'validation'),'--version',version,'--output',str(output),
   '--epochs','2','--batch-size','24' if task=='statement' else '28','--accumulation','3' if task=='statement' else '1',
   '--rank','8','--learning-rate','2e-5','--precision','fp16','--seed','20261020']
  previous=record.get('previous_output')
  if previous:
   checkpoints=sorted((Path(previous)/'training').glob('checkpoint-*'),key=lambda path:int(path.name.split('-')[-1]))
   complete=[path for path in checkpoints if (path/'trainer_state.json').exists() and (path/'optimizer.pt').exists()]
   if complete: argv+=['--resume',str(complete[-1])]
  return argv
 def assemble():
  for task in ('statement','segmentation'):
   destination=run/'families'/task
   if load(destination/'family.json').get('complete'): continue
   shutil.copytree(run/'families-base'/task,destination,dirs_exist_ok=True)
   metadata=load(run/'families-base'/task/'family.json');metadata['complete']=False
   (destination/'family.json').write_text(json.dumps(metadata,indent=2)+'\n')
   for version in ('3.14','3.15'):
    record=state['jobs']['adapter-'+version+'-'+task];source=Path(record['output']);part=load(source/'family.json')
    if part['backbone_id']!=metadata['backbone_id'] or part['tokenizer_sha256']!=metadata['tokenizer_sha256']:
     raise ValueError('Adapter family immutable identity differs')
    relative=part['adapters'][version];shutil.copytree(source/relative,destination/relative,dirs_exist_ok=True)
    for key in ('adapters','opcode_bindings','adapter_updates'):
     metadata.setdefault(key,{})[version]=part[key][version]
   metadata['versions']=sorted(set(metadata['versions'])|{'3.15'},key=lambda value:tuple(map(int,value.split('.'))))
   metadata['complete']=True
   temporary=destination/'family.json.tmp';temporary.write_text(json.dumps(metadata,indent=2)+'\n');temporary.replace(destination/'family.json')
 try:
  while True:
   publish_bases();occupied=reserved_backbone_gpus(universal_ready('statement'),universal_ready('segmentation'))
   for job in jobs:
    name=job['name'];record=state['jobs'].setdefault(name,{'phase':'pending','attempts':0})
    if record['phase']=='complete': continue
    if name in live:
     code=live[name].poll()
     if code is None: occupied.add(record['gpu']);continue
     streams.pop(name).close();live.pop(name);record['exit_code']=code
     if code==0 and finished(job,record): record.update(phase='complete',completed_at=time.time());save();continue
     record.update(phase='retry-pending',previous_output=record.get('output'),last_failure=time.time())
    elif record['phase']=='running':
     pid=record.get('pid');proc=Path('/proc/'+str(pid))
     if proc.exists(): occupied.add(record['gpu']);continue
     if finished(job,record): record.update(phase='complete',completed_at=time.time());save();continue
     record.update(phase='retry-pending',previous_output=record.get('output'))
    if record['attempts']>=3: raise RuntimeError('Repeated training failure:'+name)
   if all(record.get('phase')=='complete' for record in state['jobs'].values()) and len(state['jobs'])==len(jobs):
    assemble();state['phase']='training-complete-awaiting-benchmark';save();return
   free=[gpu for gpu in range(4) if gpu not in occupied]
   if free:
    probe=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in probe.stdout.splitlines()}
    free=[gpu for gpu in free if memory[gpu]<1024]
   for job in sorted(jobs,key=lambda item:-item['priority']):
    if not free: break
    record=state['jobs'][job['name']]
    if record['phase'] not in ('pending','retry-pending') or not ready(job): continue
    record['attempts']+=1;record['gpu']=free.pop(0);argv=argv_for(job,record)
    log=run/'logs'/(job['name']+'-attempt'+str(record['attempts'])+'.log');log.parent.mkdir(exist_ok=True)
    stream=log.open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(record['gpu']))
    process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env)
    live[job['name']]=process;streams[job['name']]=stream
    record.update(phase='running',pid=process.pid,argv=argv,log=str(log),started_at=time.time());save()
   save();time.sleep(30)
 except BaseException as error:
  state.update(phase='failed',error=repr(error));save();raise
 finally:
  for stream in streams.values(): stream.close()
if __name__=='__main__':main()
