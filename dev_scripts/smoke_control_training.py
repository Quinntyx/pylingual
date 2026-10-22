import json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);run=root/'full-retrain-20261020';base=run/'control-path-smoke';base.mkdir(exist_ok=True)
for task in ('statement','segmentation'):
 subprocess.run([sys.executable,'-m','dev_scripts.prepare_full_training_data','--data',str(run/'target-corpora-smoke/3.15/train'),'--tokenizer',str(run/'control-tokenizer-smoke/3.15'/task),'--task',task,'--output',str(base/'data'/task),'--workers','2','--scope','control','--smoke-files','4'],check=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
for task in ('mlm','segmentation','statement'):
 argv=[sys.executable,'-m','dev_scripts.train_universal_models','--task',task,'--data',str(base/'data'/('statement' if task=='statement' else 'segmentation')),'--tokenizer',str(run/'control-tokenizer-smoke/3.15'/('statement' if task=='statement' else 'segmentation')),'--output',str(base/task),'--smoke-only','--steps','4','--batch-size','2','--ordinary-version','3.15']
 if task=='statement': argv+=['--initialization',str(root/'gpu-smoke-20261020/generic-codet5')]
 if task=='segmentation': argv+=['--initialization',str(base/'mlm/checkpoint')]
 subprocess.run(argv,env=env,check=True)
print(json.dumps({'complete':True,'smoke_only':True,'ordinary_version':'3.15','tasks':['mlm','segmentation','statement']}),flush=True)
