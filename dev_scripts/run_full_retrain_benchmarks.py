"""Automatically benchmark the shared family on both official evaluation releases; no controls."""
import argparse,csv,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
from dev_scripts.benchmark_full_retrain import ARMS,DATASETS,VERSIONS,pin,selection

def alive(pid):
 try: os.kill(int(pid),0);return True
 except (ProcessLookupError,ValueError,TypeError):return False

def compiler_port(run):
 route=json.loads((run/'benchmark-compiler-route.json').read_text());provenance=run/'benchmark-compiler-rpc.json'
 if route['compiler_provenance_sha256']!=hashlib.sha256(provenance.read_bytes()).hexdigest():raise ValueError('Changed native compiler provenance')
 if route['gateway_port']!=json.loads(provenance.read_text())['port']:raise ValueError('Native compiler route mismatch')
 return int(route['remote_port'])

def compiler_preflight(port):
 for version in VERSIONS:
  with socket.create_connection(('127.0.0.1',port),timeout=10) as connection:
   stream=connection.makefile('rwb');stream.write((json.dumps({'version':version,'source':'x = 1\n','filename':'<benchmark-preflight>'})+'\n').encode());stream.flush();response=json.loads(stream.readline())
   if not response.get('success'): raise RuntimeError('Native verification unavailable: '+str(response))

def free_gpus():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
 used=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines())
 return [int(index.strip()) for index,uuid in csv.reader(rows.splitlines()) if uuid.strip() not in used]

def benchmark_jobs(output):
 jobs=[]
 for arm in ARMS:
  for dataset in DATASETS:
   for version in VERSIONS:
    jobs.append({'name':dataset+'-'+arm+'-'+version,'arm':arm,'dataset':dataset,'version':version,'result':output/dataset/arm/version/'summary.json'})
  jobs.append({'name':arm+'-performance','arm':arm,'dataset':None,'version':None,'result':output/'performance'/arm/'performance.json'})
 return jobs

def dependencies_satisfied(job,state):
 return job['version'] is not None or all(state['jobs'][dataset+'-'+job['arm']+'-'+version]['phase']=='complete' for dataset in DATASETS for version in VERSIONS)

def training_ready(run):
 path=run/'training-supervisor.json'
 state=json.loads(path.read_text()) if path.exists() else {}
 expected={'adapter-'+version+'-'+task for version in VERSIONS for task in ('statement','segmentation')}
 if any(name.startswith('control-') for name in state.get('jobs',{})):raise RuntimeError('Unexpected control-training dependency')
 return state.get('phase') in ('training-complete-awaiting-benchmark','complete') and set(state.get('jobs',{}))==expected and all(record.get('phase')=='complete' for record in state['jobs'].values())

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=None);a=p.parse_args();run=a.run.resolve();output=run/'benchmark'
 expected_port=compiler_port(run)
 if a.port is not None and a.port!=expected_port:raise ValueError('Compiler RPC port differs from pinned benchmark provenance')
 a.port=expected_port
 state_path=run/'benchmark-supervisor.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':{}}
 state.update(pid=os.getpid(),datasets=DATASETS,arms=list(ARMS),comparison_models='cancelled by user; external deployed results are not matched controls');live={};streams={};jobs=benchmark_jobs(output)
 names={job['name'] for job in jobs}
 for name in list(state['jobs']):
  if name in names:continue
  record=state['jobs'][name]
  if record.get('phase')=='running' and alive(record.get('pid')):raise RuntimeError('Unexpected old benchmark still running')
  state.setdefault('cancelled_jobs',{})[name]={**record,'phase':'cancelled','reason':'Superseded by authorized shared-only v1/v2 queue'};del state['jobs'][name]
 for job in jobs:state['jobs'].setdefault(job['name'],{'phase':'pending','attempts':0})
 def save(phase):
  state.update(phase=phase,updated_at=time.time());temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
 def done(job):
  if not job['result'].exists(): return False
  doc=json.loads(job['result'].read_text())
  if job['version']:
   selected=json.loads((job['result'].parent/'selection.json').read_text());rows=[json.loads(line) for line in (job['result'].parent/'file_results.jsonl').open()]
   if len(rows)!=len(selected['files']) or doc[job['version']]['files']!=len(rows):raise ValueError('Incomplete benchmark result')
  elif not doc.get('complete'):return False
  return True
 try:
  while not training_ready(run):save('waiting-for-full-training');time.sleep(30)
  for dataset in DATASETS:
   for version in VERSIONS:pin(output/dataset/('selection-'+version+'.json'),selection(run,version,dataset))
  while True:
   for job in jobs:
    record=state['jobs'][job['name']]
    if job['name'] in live:
     process=live[job['name']]
     if process.poll() is None:continue
     code=process.returncode;streams.pop(job['name']).close();del live[job['name']];record['exit_code']=code
     if code==0 and done(job):record['phase']='complete'
     else:
      record['phase']='retry-pending'
      if record['attempts']>=3:raise RuntimeError('Repeated benchmark failure: '+job['name'])
    elif record.get('phase')=='running' and alive(record.get('pid')):continue
    elif done(job):record['phase']='complete'
    elif record.get('phase')=='running':record['phase']='retry-pending'
   if all(state['jobs'][job['name']]['phase']=='complete' for job in jobs):
    report={'complete':True,'datasets':{},'performance':{},'protocol_sha256':hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest(),'comparison_models':state['comparison_models']}
    for dataset,release in DATASETS.items():
     report['datasets'][dataset]={'release':release,'files_per_version':{v:len(json.loads((output/dataset/('selection-'+v+'.json')).read_text())['files']) for v in VERSIONS},'arms':{arm:{v:json.loads((output/dataset/arm/v/'summary.json').read_text())[v] for v in VERSIONS} for arm in ARMS}}
    for arm in ARMS:report['performance'][arm]=json.loads((output/'performance'/arm/'performance.json').read_text())
    pin(output/'report.json',report);save('complete');print(json.dumps({'full_study_complete':True,'report':str(output/'report.json')}),flush=True);return
   try:compiler_preflight(a.port)
   except Exception as error:state['compiler_error']=repr(error);save('waiting-for-native-compiler-rpc');time.sleep(30);continue
   state.pop('compiler_error',None);free=free_gpus()
   for job in jobs:
    record=state['jobs'][job['name']]
    if not free:break
    if record['phase'] not in ('pending','retry-pending') or not dependencies_satisfied(job,state):continue
    gpu=free.pop(0);record['attempts']+=1
    argv=[sys.executable,'-u','-m','dev_scripts.benchmark_full_retrain','--run',str(run),'--output',str(output),'--arm',job['arm']]
    argv+=['--version',job['version'],'--dataset',job['dataset']] if job['version'] else ['--performance']
    log=run/'logs'/('benchmark-'+job['name']+'-attempt'+str(record['attempts'])+'.log');log.parent.mkdir(exist_ok=True);stream=log.open('a')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYLINGUAL_COMPILE_GATEWAY_PORT=str(a.port))
    process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env);live[job['name']]=process;streams[job['name']]=stream
    record.update(phase='running',pid=process.pid,gpu=gpu,argv=argv,log=str(log),started_at=time.time());save('running-benchmarks')
   save('running-benchmarks');time.sleep(15)
 except BaseException as error:
  state['error']=repr(error);save('failed');raise
 finally:
  for stream in streams.values():stream.close()
if __name__=='__main__':main()
