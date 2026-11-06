from __future__ import annotations
import json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);run=root/'full-retrain-20261020';out=run/'arrow-adapter-smoke';out.mkdir(exist_ok=True)
(root/'full-retrain-20261020/315-bindings-smoke.json').write_text(json.dumps({'TRACE_RECORD':0}))
for task in ('statement','segmentation'):
 for split in ('train','validation'):
  argv=[sys.executable,'-m','dev_scripts.prepare_full_training_data','--scope','adapter','--data',str(run/'target-corpora-smoke/3.15'/split),'--tokenizer',str(root/'tokenizers'/task),'--task',task,'--output',str(out/'data'/task/split),'--workers','2','--smoke-files','4','--bindings-file',str(run/'315-bindings-smoke.json')]
  subprocess.run(argv,check=True)
 from dev_scripts.publish_shared_backbone import publish
 family=publish(root/'gpu-smoke-20261020-v3'/task/'checkpoint',root/'tokenizers'/task,task,out/'base'/task,training_versions=['3.6','3.7','3.8','3.9','3.10','3.11','3.12','3.13','3.14'],smoke_only=True)
 argv=[sys.executable,'-m','dev_scripts.train_shared_adapter','--family',str(family),'--data',str(run/'target-corpora-smoke/3.15/train/accepted.jsonl'),'--validation-data',str(run/'target-corpora-smoke/3.15/validation/accepted.jsonl'),'--encoded-data',str(out/'data'/task/'train'),'--encoded-validation',str(out/'data'/task/'validation'),'--version','3.15','--output',str(out/'trained'/task),'--steps','4','--batch-size','2','--accumulation','1','--checkpoint-steps','2']
 subprocess.run(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),check=True)
 metadata=json.loads((out/'trained'/task/'family.json').read_text())
 update=metadata['adapter_updates']['3.15']
 if update['files']!=4 or update['optimizer_updates']<1 or not metadata['complete']: raise RuntimeError('Arrow adapter smoke did not complete')
print(json.dumps({'complete':True,'smoke_only':True,'memory_mapped_adapter_cli':True,'tasks':['statement','segmentation']}),flush=True)
