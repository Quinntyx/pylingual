"""Full, resumable benchmark of the corrected 100k/54,500-file retrain.

Independent full-file bytecode equivalence is the primary quality metric.
Teacher-forced validation scores are diagnostics, not decompilation accuracy.
"""
from __future__ import annotations
import argparse,contextlib,fcntl,gc,gzip,hashlib,json,math,os,statistics,time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ARMS=('adapted',)
AVAILABLE_ARMS=('no_adapter','statement_adapter','adapted')
DATASETS={'v1':'v1.1.0','v2':'v2.0.1'}
VERSIONS=('3.14','3.15')

def sha(path):
 with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def pin(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and json.loads(path.read_text())!=value: raise ValueError('Changed benchmark inputs: '+str(path))
 temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def family_directory(run,arm):
 # Base families are immutable and available while version adapters are still training.
 base=run/'families-base'
 return base if arm=='no_adapter' and base.is_dir() else run/'families'

def configuration(run,arm,version):
 ordinary={'STATEMENT_MODEL':{'REPO':str(run/'controls'/version/'statement/checkpoint'),'REVISION':None,'TOKENIZER':str(run/'control-tokenizers'/version/'statement')},
  'SEGMENTATION_MODEL':{'REPO':str(run/'controls'/version/'segmentation/checkpoint'),'REVISION':None,'TOKENIZER':str(run/'control-tokenizers'/version/'segmentation')}}
 if arm=='control': return {'v'+version:ordinary}
 return {'SHARED_BACKBONES':{'STATEMENT_MODEL':str(family_directory(run,arm)/'statement'),'SEGMENTATION_MODEL':str(family_directory(run,arm)/'segmentation'),
  'USE_ADAPTERS':arm!='no_adapter','USE_STATEMENT_ADAPTERS':arm in ('statement_adapter','adapted'),'USE_SEGMENTATION_ADAPTERS':arm=='adapted'}}

def training_audit(run):
 root=run.parent;journals=[root/'data-100k/accepted.jsonl']
 journals.extend(run/'target-corpora'/v/s/'accepted.jsonl' for v in VERSIONS for s in ('train','validation'))
 identities=set();summary={};canonicals={}
 for path in journals:
  rows=[json.loads(line) for line in path.open() if line.strip()]
  counts=Counter(row['corpus'] for row in rows);versions=Counter(row['version'] for row in rows)
  universal=path==journals[0];expected=100000 if universal else 54500 if path.parent.name=='train' else 500
  if len(rows)!=expected or counts!={'pypi':expected*9//10,'csn':expected//10}: raise ValueError('Incorrect full source budget: '+str(path))
  if universal and (set(versions)!={'3.'+str(i) for i in range(6,15)} or max(versions.values())-min(versions.values())>1): raise ValueError('Backbone versions unbalanced or contaminated')
  for key in ('source_sha512','source_id','payload_sha512'):
   identities.update(row[key] for row in rows if row.get(key))
  canonical={row['canonical_sha256'] for row in rows};canonicals[str(path)]=canonical
  if len(canonical)!=expected: raise ValueError('Duplicate normalized source in partition')
  summary[str(path)]={'files':len(rows),'corpora':dict(counts),'versions':dict(versions),'journal_sha256':sha(path)}
 for version in VERSIONS:
  train=canonicals[str(run/'target-corpora'/version/'train/accepted.jsonl')]
  valid=canonicals[str(run/'target-corpora'/version/'validation/accepted.jsonl')]
  if train&valid or (train|valid)&canonicals[str(journals[0])]: raise ValueError('Train/validation/backbone source leakage')
 return identities,summary

def dataset_directory(run,dataset):
 if dataset not in DATASETS: raise ValueError('Unknown evaluation dataset: '+dataset)
 return run/'independent-benchmark'/('dataset-'+DATASETS[dataset])

def selection(run,version,dataset='v2'):
 dataset_name=dataset;dataset=dataset_directory(run,dataset_name);listing=dataset/(version.replace('.','')+'-pyc-list.txt')
 excluded,audit=training_audit(run);files=[]
 paths=[line.strip() for line in listing.read_text().splitlines() if line.strip()]
 if len(set(paths))!=len(paths): raise ValueError('Duplicate benchmark entries')
 for relative in paths:
  file=(dataset/relative).resolve()
  if not file.is_relative_to(dataset.resolve()) or not file.is_file(): raise ValueError('Missing/unsafe benchmark input')
  overlap=file.parent.name in excluded
  files.append({'version':version,'relative':relative,'source_sha512':file.parent.name,'pyc_sha256':sha(file),'known_training_or_validation_source_overlap':overlap})
 if not files: raise ValueError('Empty independent benchmark')
 return {'dataset':dataset_name,'release':DATASETS[dataset_name],'commit':__import__('subprocess').check_output(['git','-C',str(dataset),'rev-parse','HEAD'],text=True).strip(),'versions':[version],'files_per_version':len(files),'full_listing':str(listing),'listing_sha256':sha(listing),'source_hash_overlap':sum(row['known_training_or_validation_source_overlap'] for row in files),'scoring_policy':'all official files; report non-overlap stratum separately, not all results held-out',
  'source_audit':audit,'canonical_benchmark_overlap':'not rederived: benchmark contains bytecode, not raw sources; raw SHA-512 identity checked', 'files':files}

def require_full_models(run,arm,versions):
 if arm=='control':
  roots=[(run/'controls'/version,54500,'control') for version in versions]
 else: roots=[(run/'universal',100000,'universal')]
 for root,budget,scope in roots:
  for task in ('statement','mlm','segmentation'):
   result=json.loads((root/task/'result.json').read_text())
   if not result.get('complete') or result.get('smoke_only') or result.get('scope')!=scope or result.get('files')!=budget or result.get('epoch',0)<1.999:
    raise ValueError('Not a completed full two-epoch model: '+str(root/task))
 if arm!='control':
  for task in ('statement','segmentation'):
   metadata=json.loads((family_directory(run,arm)/task/'family.json').read_text())
   if not metadata.get('complete') or metadata.get('smoke_only'): raise ValueError('Incomplete/smoke family')
   enabled=arm in ('adapted','statement_adapter') if task=='statement' else arm=='adapted'
   if enabled:
    for version in versions:
     update=metadata.get('adapter_updates',{}).get(version,{})
     if update.get('files')!=54500 or update.get('epochs')!=2 or update.get('training',{}).get('epoch',0)<1.999:
      raise ValueError('Not a completed full version adapter: '+task+' '+version)

def fingerprints(run,arm,versions):
 repo=Path(__file__).resolve().parents[1];paths=set(repo.joinpath('pylingual').rglob('*.py'))
 paths.add(Path(__file__));paths.add(repo/'dev_scripts/evaluate_shared.py');paths.add(repo/'dev_scripts/benchmark_stage_timing.py')
 for version in versions:
  families=[run/'controls'/version/'statement/checkpoint',run/'controls'/version/'segmentation/checkpoint',run/'control-tokenizers'/version] if arm=='control' else [family_directory(run,arm)/'statement',family_directory(run,arm)/'segmentation']
  for family in families:
   paths.update(p for p in family.rglob('*') if p.is_file())
 return {str(path):sha(path) for path in sorted(paths)}

def task_objects(segmenter,translator,task):
 wrapper=translator if task=='statement' else segmenter
 return wrapper,wrapper.model,wrapper.tokenizer

def context(wrapper):
 return wrapper.bank.use(wrapper.version) if hasattr(wrapper,'bank') else contextlib.nullcontext()

def validation(args,pair,output):
 import torch
 from datasets import load_from_disk
 from dev_scripts.shared_training import collate
 scope='control' if args.arm=='control' else 'adapter';report={}
 for task in ('statement','segmentation'):
  path=args.run/'encoded'/scope/args.version/task/'validation'
  data=load_from_disk(str(path/'dataset'));wrapper,model,tokenizer=task_objects(*pair,task)
  model.eval();tokens=correct=0;weighted_loss=0.;started=time.monotonic()
  with context(wrapper),torch.inference_mode():
   for offset in range(0,len(data),16):
    batch=collate([data[i] for i in range(offset,min(offset+16,len(data)))],tokenizer,'cuda')
    answer=model(**batch);labels=batch['labels'];mask=labels!=-100;count=int(mask.sum())
    loss=float(answer.loss)
    if not math.isfinite(loss): raise RuntimeError('Non-finite validation loss')
    tokens+=count;correct+=int(((answer.logits.argmax(-1)==labels)&mask).sum());weighted_loss+=loss*count
  report[task]={'rows':len(data),'supervised_tokens':tokens,'teacher_forced_token_accuracy':correct/tokens,'token_weighted_loss':weighted_loss/tokens,'seconds':time.monotonic()-started,
   'interpretation':'diagnostic; tokenization/windows differ between ordinary and shared models, not full-file accuracy','manifest_sha256':sha(path/'manifest.json')}
 pin(output/'validation.json',report)

def stratified_summary(rows,version):
 def summarize(subset):
  count=len(subset);passed=sum(bool(row.get('success')) for row in subset)
  return {'files':count,'equal_files':passed,'file_accuracy':passed/count if count else None,
   'code_objects':sum(row.get('code_objects',0) for row in subset),'equal_code_objects':sum(row.get('equal_code_objects',0) for row in subset),
   'categories':dict(Counter(row.get('category','unknown') for row in subset)),'seconds':round(sum(row.get('seconds',0) for row in subset),3)}
 subset=[row for row in rows if row['version']==version]
 result=summarize(subset)
 result['known_training_or_validation_source_overlap']=sum(bool(row.get('known_training_or_validation_source_overlap')) for row in subset)
 result['non_overlapping_source_files']=summarize([row for row in subset if not row.get('known_training_or_validation_source_overlap')])
 result['known_overlapping_source_files']=summarize([row for row in subset if row.get('known_training_or_validation_source_overlap')])
 result['coverage_note']='Raw source SHA-512 overlap checked; canonical source equivalence not rederived from bytecode.'
 return {version:result}

def instrumentation_provenance(run,repo):
 path=run/'benchmark-instrumentation-manifest.json'
 if not path.exists(): return None
 document=json.loads(path.read_text());original=run/'source-manifest.json'
 if sha(original)!=document['original_source_manifest_sha256']: raise ValueError('Training source manifest changed after instrumentation')
 expected=dict(json.loads(original.read_text())['files']);overrides=document['overrides']
 allowed={'dev_scripts/benchmark_full_retrain.py','dev_scripts/evaluate_shared.py','dev_scripts/benchmark_stage_timing.py'}
 if set(overrides)!=allowed: raise ValueError('Instrumentation attempted to change production/training source')
 for name,change in overrides.items():
  if expected.get(name)!=change['original_sha256']: raise ValueError('Incorrect original instrumentation identity: '+name)
  expected[name]=change['instrumented_sha256']
 mismatched=[name for name,digest in expected.items() if not (repo/name).is_file() or sha(repo/name)!=digest]
 if mismatched: raise ValueError('Benchmark source differs from frozen training source plus logging-only overlay: '+repr(mismatched))
 return {'manifest_sha256':sha(path),'original_source_manifest_sha256':sha(original),'overrides':overrides}

def unit(args):
 import torch
 from pylingual.models import load_models
 from pylingual.utils.version import PythonVersion
 from dev_scripts.evaluate_shared import file_metrics
 torch.set_num_threads(4);require_full_models(args.run,args.arm,[args.version]);output=args.output/args.dataset/args.arm/args.version;output.mkdir(parents=True,exist_ok=True)
 selected=selection(args.run,args.version,args.dataset);pin(output/'selection.json',selected)
 config=output/'runtime.json';pin(config,configuration(args.run,args.arm,args.version))
 pin(output/'provenance.json',{'arm':args.arm,'version':args.version,'file_timeout_seconds':args.timeout,'precision':'FP32 inference',
  'stage_timing':bool(getattr(args,'stage_timing',False)),
 'instrumentation':instrumentation_provenance(args.run,Path(__file__).resolve().parents[1]) if getattr(args,'stage_timing',False) else None,
 'dataset':args.dataset,'release':DATASETS[args.dataset],'protocol_sha256':sha(args.run/'protocol.json'),'fingerprints':fingerprints(args.run,args.arm,[args.version]),'compiler_rpc':json.loads((args.run/'benchmark-compiler-rpc.json').read_text()),'compiler_route':json.loads((args.run/'benchmark-compiler-route.json').read_text())})
 from dev_scripts.benchmark_stage_timing import TIMING_PROTOCOL
 if getattr(args,'stage_timing',False): pin(output/'timing-protocol.json',TIMING_PROTOCOL)
 started=time.perf_counter()
 pair=load_models(config_file=config,version=PythonVersion(args.version))
 if getattr(args,'stage_timing',False) and torch.cuda.is_initialized(): torch.cuda.synchronize()
 model_load_elapsed=time.perf_counter()-started
 if getattr(args,'stage_timing',False):
  import platform
  pin(output/'timing-environment.json',{'host':platform.node(),'python':platform.python_version(),'torch':str(torch.__version__),'torch_threads':torch.get_num_threads(),'cpu_affinity_count':len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count(),'rayon_num_threads':os.environ.get('RAYON_NUM_THREADS'),'tokenizers_parallelism':os.environ.get('TOKENIZERS_PARALLELISM'),'gpu':torch.cuda.get_device_name() if torch.cuda.is_initialized() else None,'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_initialized() else None,'scope':'This benchmark host, not necessarily the deployed four-core/2080 workers.'})
  # A fresh worker startup is separate from the warm per-file denominator.
  with (output/'model-startup.jsonl').open('a') as stream: stream.write(json.dumps({'model_load_seconds':model_load_elapsed})+'\n')
 diagnostic=args.output/'validation'/args.arm/args.version;diagnostic.mkdir(parents=True,exist_ok=True)
 with (diagnostic/'validation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not (diagnostic/'validation.json').exists(): validation(args,pair,diagnostic)
 settings=SimpleNamespace(output=output,config=config,versions=[args.version],dataset=dataset_directory(args.run,args.dataset),timeout=args.timeout,resume=True,stage_timing=bool(getattr(args,'stage_timing',False)))
 if not (output/'summary.json').exists(): file_metrics(settings,selected)
 rows=[json.loads(line) for line in (output/'file_results.jsonl').open()]
 if len(rows)!=len(selected['files']) or len({r['relative'] for r in rows})!=len(rows): raise RuntimeError('Incomplete full-file benchmark')
 if any(row.get('phase')=='model_load' for row in rows): raise RuntimeError('Infrastructure model load failure invalidates benchmark')
 summary=stratified_summary(rows,args.version)
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'complete':True,'arm':args.arm,'version':args.version,'dataset':args.dataset,'summary':summary}),flush=True)

def performance(args):
 import torch
 from datasets import load_from_disk
 from pylingual.models import load_models
 from pylingual.utils.version import PythonVersion
 from dev_scripts.shared_training import collate
 torch.set_num_threads(4);require_full_models(args.run,args.arm,VERSIONS);output=args.output/'performance'/args.arm;output.mkdir(parents=True,exist_ok=True)
 pin(output/'performance-provenance.json',{'fingerprints':fingerprints(args.run,args.arm,VERSIONS),'protocol_sha256':sha(args.run/'protocol.json')})
 if (output/'performance.json').exists(): return
 torch.cuda.init();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();pairs={};loads={}
 for version in VERSIONS:
  config=output/('performance-'+version+'.json');pin(config,configuration(args.run,args.arm,version))
  torch.cuda.synchronize();started=time.perf_counter();pairs[version]=load_models(config_file=config,version=PythonVersion(version));torch.cuda.synchronize();loads[version]=time.perf_counter()-started
 storages={}
 for pair in pairs.values():
  for model in (pair[0].model,pair[1].model):
   for value in list(model.parameters())+list(model.buffers()):
    storage=value.untyped_storage();storages[storage.data_ptr()]=storage.nbytes()
 gc.collect();torch.cuda.empty_cache()
 resident={'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_reserved_bytes':torch.cuda.memory_reserved(),'unique_storage_bytes':sum(storages.values()),'loaded_versions':list(VERSIONS),
  'shared_statement_object':pairs['3.14'][1].model is pairs['3.15'][1].model,'shared_segmentation_object':pairs['3.14'][0].model is pairs['3.15'][0].model}
 results={};scope='control' if args.arm=='control' else 'adapter'
 for version,pair in pairs.items():
  results[version]={}
  for task in ('statement','segmentation'):
   data=load_from_disk(str(args.run/'encoded'/scope/version/task/'validation/dataset'))
   wrapper,model,tokenizer=task_objects(*pair,task);batch=collate([data[i] for i in range(min(8,len(data)))],tokenizer,'cuda');batch.pop('labels')
   elapsed=[];output_tokens=[];torch.cuda.reset_peak_memory_stats()
   with context(wrapper),torch.inference_mode():
    for iteration in range(12):
     torch.cuda.synchronize();started=time.perf_counter()
     if task=='statement':
      answer=model.generate(**batch,max_length=512,**getattr(wrapper,'generation_kwargs',{}));generated=int((answer!=tokenizer.pad_token_id).sum())
     else: answer=model(**batch);generated=0
     torch.cuda.synchronize();seconds=time.perf_counter()-started
     if iteration>=2: elapsed.append(seconds);output_tokens.append(generated)
   results[version][task]={'examples':batch['input_ids'].shape[0],'padded_input_shape':list(batch['input_ids'].shape),'seconds_median':statistics.median(elapsed),
    'seconds_min':min(elapsed),'seconds_max':max(elapsed),'samples_per_second':batch['input_ids'].shape[0]/statistics.median(elapsed),
    'generated_tokens_per_second':sum(output_tokens)/sum(elapsed) if task=='statement' else None,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
    'limits':'tokenized inputs differ by tokenizer; model kernels only, no translator output cache; two warmups, ten repeats'}
 pin(output/'performance.json',{'complete':True,'precision':'FP32','resident_two_versions':resident,'cold_load_seconds':loads,'timing':results})
 print(json.dumps({'performance_complete':True,'arm':args.arm,'resident':resident}),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--arm',choices=AVAILABLE_ARMS,required=True);p.add_argument('--dataset',choices=DATASETS,default='v2');p.add_argument('--version',choices=VERSIONS);p.add_argument('--performance',action='store_true');p.add_argument('--timeout',type=int,default=300)
 p.add_argument('--stage-timing',action=argparse.BooleanOptionalAction,default=None,help='Log exclusive per-file/aggregate stage wall times; optional run/benchmark-timing.json enables queued benchmarks')
 a=p.parse_args();a.run=a.run.resolve();a.output=a.output.resolve()
 marker=a.run/'benchmark-timing.json'
 if a.stage_timing is None: a.stage_timing=marker.exists() and json.loads(marker.read_text()).get('enabled',False)
 if bool(a.version)==bool(a.performance): p.error('Choose version or performance')
 performance(a) if a.performance else unit(a)
if __name__=='__main__':main()
