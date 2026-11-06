"""Pause only this experiment's owned lanes for the authorized 2+2 transition."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path

def write(path,value):
 pending=path.with_suffix('.json.tmp');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)
def command(pid):
 try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return ''
def identity(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
 except FileNotFoundError:return None
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--release-supervisor',action='store_true');a=p.parse_args();run=a.root/'full-retrain-20261020';path=run/'repartition-state.json'
 if a.release_supervisor:
  state=json.loads(path.read_text());pid=state['supervisor_pid']
  if identity(pid)!=state['supervisor_identity']:raise RuntimeError('Supervisor PID identity changed')
  os.kill(pid,signal.SIGCONT);state.update(phase='running-2+2',supervisor_suspended=False,released_at=time.time());write(path,state);print(json.dumps(state));return
 if path.exists():raise RuntimeError('Existing repartition record requires inspection, not an automatic repeated stop')
 lanes={lane:json.loads((run/(lane+'-lane.json')).read_text()) for lane in ('statement','segmentation')}
 supervisor=json.loads((run/'training-supervisor.json').read_text());spid=supervisor['pid']
 if str(a.root) not in command(spid) or 'dev_scripts.' not in command(spid):raise RuntimeError('Unexpected supervisor ownership')
 if any(v.get('phase')=='running' for v in supervisor['jobs'].values()):raise RuntimeError('Additional training jobs are active')
 roots={v['pid'] for v in lanes.values()}
 for lane,metadata in lanes.items():
  cmd=command(metadata['pid'])
  if 'dev_scripts.run_full_gpu_lane' not in cmd or str(a.root) not in cmd or '--lane '+lane not in cmd:raise RuntimeError('Unexpected lane ownership')
 mlm=run/'universal/mlm/training';checkpoints=[]
 for checkpoint in mlm.glob('checkpoint-*'):
  required=('model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth')
  if all((checkpoint/name).is_file() and (checkpoint/name).stat().st_size>0 for name in required):checkpoints.append(checkpoint)
 if not checkpoints:raise RuntimeError('No complete single-GPU MLM checkpoint')
 checkpoint=max(checkpoints,key=lambda p:int(p.name.split('-')[1]));saved=json.loads((checkpoint/'trainer_state.json').read_text())
 if saved['global_step']<9000 or saved['max_steps']!=216712:raise RuntimeError('Unexpected MLM training scope')
 progress={task:json.loads((run/'universal'/task/'progress.json').read_text()) for task in ('statement','mlm')}
 if progress['statement']['world_size']!=3 or progress['mlm']['world_size']!=1:raise RuntimeError('Not the expected original layout')
 archive=run/'universal/statement-before-ddp-loss-fix-20261020'
 if archive.exists():raise RuntimeError('Statement quarantine already exists')
 pairs=[tuple(map(int,line.split())) for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,ppid='],text=True).splitlines() if line.strip()]
 owned=set(roots)
 while True:
  more={pid for pid,parent in pairs if parent in owned}-owned
  if not more:break
  owned.update(more)
 identities={pid:identity(pid) for pid in owned}
 state={'phase':'stopping-3+1','started_at':time.time(),'previous_progress':progress,'previous_lanes':lanes,'supervisor_pid':spid,'supervisor_identity':identity(spid),'supervisor_suspended':False,'owned_pids':sorted(owned),'mlm_checkpoint':str(checkpoint),'mlm_global_step':saved['global_step'],'statement_quarantine':str(archive),'reason':'Transformers multiplied native mean loss by DDP world size; preserve valid single-GPU MLM, restart incorrectly scaled statement'}
 write(path,state)
 try:
  os.kill(spid,signal.SIGSTOP);state['supervisor_suspended']=True;write(path,state)
  for pid in sorted(owned,key=lambda p:(p not in roots,p)):
   if identities[pid] is not None and identity(pid)==identities[pid]:
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
  deadline=time.monotonic()+45
  while time.monotonic()<deadline:
   remaining=[pid for pid in owned if command(pid) and identity(pid)==identities[pid]]
   if not remaining:break
   time.sleep(1)
  for pid in remaining:
   if identity(pid)==identities[pid]:
    try:os.kill(pid,signal.SIGKILL)
    except ProcessLookupError:pass
  incomplete=[]
  for candidate in mlm.glob('checkpoint-*'):
   required=('model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth')
   if not all((candidate/name).is_file() and (candidate/name).stat().st_size>0 for name in required):
    quarantine=mlm.parent/('incomplete-'+candidate.name+'-before-2+2');candidate.rename(quarantine);incomplete.append(str(quarantine))
  remaining_checkpoints=list(mlm.glob('checkpoint-*'))
  actual_checkpoint=max(remaining_checkpoints,key=lambda p:int(p.name.split('-')[1]))
  actual_saved=json.loads((actual_checkpoint/'trainer_state.json').read_text())
  state.update(mlm_checkpoint=str(actual_checkpoint),mlm_global_step=actual_saved['global_step'],incomplete_checkpoint_quarantine=incomplete)
  (run/'universal/statement').rename(archive)
  state.update(phase='stopped-awaiting-2+2',stopped_at=time.time());write(path,state);print(json.dumps(state),flush=True)
 except BaseException:
  if state['supervisor_suspended']:os.kill(spid,signal.SIGCONT)
  raise
if __name__=='__main__':main()
