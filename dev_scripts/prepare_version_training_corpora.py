"""Recover retained selections and add only new NAS-sampled3.15 source files."""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json,multiprocessing,os,queue,signal,socket,subprocess,threading,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from dev_scripts.prepare_tokenizer_corpus import extract
from dev_scripts.run_remote_ablation import Compiler,Gateway,Handler
from dev_scripts.run_gpu4_tokenizer_preparation import compiler_command,close_compilers,verify_runtime

class Alive:
 def poll(self): return None
class Native315Proxy:
 process=Alive()
 def request(self,record):
  with socket.create_connection(('127.0.0.1',45615),timeout=45) as connection:
   stream=connection.makefile('rwb');stream.write((json.dumps(record)+'\n').encode());stream.flush()
   return json.loads(stream.readline(16000000))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--workers',type=int,default=8);p.add_argument('--smoke',action='store_true')
 p.add_argument('--output',type=Path);p.add_argument('--repo',type=Path)
 p.add_argument('--versions',nargs='+',choices=['3.14','3.15'],default=['3.14','3.15'])
 p.add_argument('--compiler-workers',type=int,default=4);a=p.parse_args()
 root=a.root;run=root/'full-retrain-20261020';repo=a.repo or root/'repo';versions=tuple(dict.fromkeys(a.versions))
 output=a.output or run/('target-corpora-smoke' if a.smoke else 'target-corpora')
 if a.workers<1 or a.compiler_workers<1: raise ValueError('Worker counts must be positive')
 output.mkdir(parents=True,exist_ok=True)
 # Source selection is immutable; transported normalized source bytes are independently checksummed.
 checksums=json.loads((run/'retained-sources/checksums.json').read_text())
 if not checksums['complete'] or checksums['files']!=60000: raise ValueError('Incomplete retained-source transfer')
 legacy=json.load(gzip.open(run/'legacy-selection-identities.json.gz','rt'))
 if len(legacy)!=60000: raise ValueError('Wrong retained metadata count')
 backbone=[json.loads(line) for line in (root/'data-100k/accepted.jsonl').open()]
 base_raw={row['source_sha512'] for row in backbone};base_canonical={row['canonical_sha256'] for row in backbone}
 if any(row['source_sha512'] in base_raw or row['canonical_sha256'] in base_canonical for row in legacy):
  raise ValueError('Retained/backbone source overlap')
 canonical_exclusions=json.loads((root/'excluded-source-identities.json').read_text())
 forbidden_raw=set(canonical_exclusions['source_sha512'])|base_raw
 forbidden_canonical=set(canonical_exclusions['canonical_sha256'])|base_canonical
 reserved=set(canonical_exclusions['nas_identities'])
 reserved.update(row['corpus']+'-'+row['source_id'] for row in backbone+legacy)
 reserved.update(forbidden_raw)
 reserved_path=output/'source-reserved.json';reserved_path.write_text(json.dumps(sorted(reserved)))
 journal=output/'accepted.jsonl';accepted=[json.loads(line) for line in journal.open()] if journal.exists() else []
 by_identity={(row['version'],row['split'],row['source_sha512']):row for row in accepted}
 seen_raw={row['source_sha512'] for row in accepted};seen_canonical={row['canonical_sha256'] for row in accepted}
 failures=Counter();start=time.monotonic();compilers=[];sampler=None
 def add(result,split,origin,expected=None):
  if 'failure' in result: return False
  row=result['provenance'];row.update(split=split,sample_origin=origin)
  if expected is not None:
   if row['canonical_sha256']!=expected['canonical_sha256']: raise ValueError('Retained canonical changed: '+json.dumps({'expected':expected,'actual':row}))
   row['legacy_scope']=expected['scope']
  if row['source_sha512'] in base_raw or row['canonical_sha256'] in base_canonical: raise ValueError('Backbone leakage')
  if row['source_sha512'] in seen_raw or row['canonical_sha256'] in seen_canonical:
   if expected is not None: raise ValueError('Retained duplicate')
   return False
  seen_raw.add(row['source_sha512']);seen_canonical.add(row['canonical_sha256']);accepted.append(row)
  by_identity[(row['version'],row['split'],row['source_sha512'])]=row
  with journal.open('a') as stream: stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush()
  return True
 def progress(phase):
  counts=Counter((r['version'],r['split'],r['corpus']) for r in accepted)
  report={'phase':phase,'files':len(accepted),'counts':[[*key,value] for key,value in sorted(counts.items())],
          'failures':dict(failures),'elapsed_seconds':time.monotonic()-start,'smoke_only':a.smoke}
  temporary=output/'progress.tmp';temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output/'progress.json')
  print(json.dumps(report),flush=True)
 try:
  baseline=json.loads((root/'compiler-provenance.json').read_text())
  image='python:'+baseline['3.14']['version'].split()[0]+'-slim'
  descriptor=json.loads(subprocess.run(['docker','image','inspect',image],capture_output=True,text=True,check=True).stdout)[0]
  owner='target-full-retrain-20261020-'+str(os.getpid());wrapper=output/'python314'
  import shlex
  wrapper.write_text('#!/bin/sh\nexec '+shlex.join(compiler_command(repo,descriptor['Id'],owner,os.getuid(),os.getgid()))+' "$@"\n');wrapper.chmod(0o700)
  pool314=queue.Queue();pool315=queue.Queue()
  for _ in range(a.compiler_workers):
   compiler=Compiler(str(wrapper),str(repo/'dev_scripts/compile_source_server.py'));compilers.append(compiler)
   verify_runtime('3.14',baseline['3.14'],compiler.metadata);pool314.put(compiler)
   if '3.15' in versions: pool315.put(Native315Proxy())
  with Gateway(('127.0.0.1',0),Handler) as gateway:
   gateway.compiler_pools={'3.14':pool314,'3.15':pool315};threading.Thread(target=gateway.serve_forever,daemon=True).start()
   compiler_provenance={'3.14':{**compilers[0].metadata,'image_id':descriptor['Id'],'execution_host':'syssec-gpu4'}}
   if '3.15' in versions:
    runtime315=json.loads((run/'native315-compile-rpc.json').read_text())
    compiler_provenance['3.15']={**runtime315,'execution_host':'syssec-canis','source_disk_staging':False}
   (output/'compiler-provenance.json').write_text(json.dumps(compiler_provenance,indent=2)+'\n')
   # Preserve all54,5003.14training and4,5003.15training identities, plus separate500validation/version.
   selected=[row for row in legacy if row["version"] in versions]
   if a.smoke:
    selected=[]
    for version in versions:
     for split in ('train','validation'):
      for corpus in ('pypi','csn'):
       selected.extend([row for row in legacy if row['version']==version and row['split']==split and row['corpus']==corpus][:2])
   pending=[r for r in selected if (r['version'],r['split'],r['source_sha512']) not in by_identity]
   with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as workers:
    for offset in range(0,len(pending),a.workers*4):
     batch=pending[offset:offset+a.workers*4];jobs=[]
     for row in batch:
      name=row['version']+'/'+row['split']+'/'+row['canonical_sha256']+'.py';source=run/'retained-sources'/name
      raw=source.read_bytes()
      if hashlib.sha256(raw).hexdigest()!=checksums['sha256'][name]: raise ValueError('Retained source checksum mismatch')
      record={key:row[key] for key in ('source_id','source_path','source_sha512','corpus')}
      record.update(raw_b64=base64.b64encode(raw).decode(),payload_sha512=hashlib.sha512(raw).hexdigest(),nas_source_version=None,nas_identity=row['corpus']+'-'+row['source_id'],retained_normalized=True,retained_canonical_sha256=row['canonical_sha256'])
      jobs.append((record,row['version'],str(output),gateway.server_address[1],row['split']))
     for row,job,result in zip(batch,jobs,workers.map(extract,jobs)):
      if 'failure' in result:
       # Retry isolated compiler transport/timeouts; never silently replace a retained file.
       for _ in range(2):
        result=extract(job)
        if 'failure' not in result: break
       if 'failure' in result: raise RuntimeError('Retained conversion failed: '+json.dumps({'metadata':row,'failure':result}))
      add(result,row['split'],'retained',row)
     progress('recovering-retained')
    if not a.smoke and '3.15' in versions:
     sampler=subprocess.Popen([sys.executable,'-u','-m','dev_scripts.sample_nas_sources','--root','/syssec-nas1/pyc-research/datasets',
      '--catalog',str(root/'nas-catalog.json.gz'),'--reserved',str(reserved_path),'--seed','20261021','--resume-offset',str(len(accepted)),'--read-workers','8'],
      stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
     for corpus,target in (('pypi',49050),('csn',5450)):
      while sum(r['version']=='3.15' and r['split']=='train' and r['corpus']==corpus for r in accepted)<target:
       remaining=target-sum(r['version']=='3.15' and r['split']=='train' and r['corpus']==corpus for r in accepted)
       requested=min(remaining,a.workers*4)
       sampler.stdin.write(json.dumps({'corpus':corpus,'count':requested})+'\n');sampler.stdin.flush();records=[]
       while True:
        line=sampler.stdout.readline()
        if not line: raise ConnectionError('NAS sampler exited')
        row=json.loads(line)
        if row.get('batch_complete'):
         if not row['emitted']: raise RuntimeError('NAS source pool exhausted')
         break
        records.append(row)
       # On resume, reject already selected raw identities BEFORE writing their chunk paths.
       fresh=[]
       for record in records:
        if record['source_sha512'] in seen_raw or record['source_sha512'] in forbidden_raw:
         failures['reserved-or-duplicate-raw-before-write']+=1
        else: fresh.append(record)
       jobs=[(record,'3.15',str(output),gateway.server_address[1],'train') for record in fresh]
       for result in workers.map(extract,jobs):
        if 'failure' in result: failures[result['failure']]+=1;continue
        row=result['provenance']
        if row['source_sha512'] in forbidden_raw or row['canonical_sha256'] in forbidden_canonical:
         failures['reserved-or-prior-canonical']+=1;continue
        if not add(result,'train','nas-new'): failures['duplicate']+=1
       progress('sampling-new315')
   for version in versions:
    for split in ('train','validation'):
     chosen=[r for r in accepted if r['version']==version and r['split']==split]
     counts=dict(Counter(r['corpus'] for r in chosen));wanted={'pypi':49050,'csn':5450} if split=='train' else {'pypi':450,'csn':50}
     if not a.smoke and counts!=wanted: raise ValueError('Wrong final target quotas')
     directory=output/version/split;directory.mkdir(parents=True,exist_ok=True)
     (directory/'accepted.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in chosen))
     manifest={'complete':True,'files':len(chosen),'version':version,'split':split,'corpora':counts,'smoke_only':a.smoke,
       'source_file_disjoint_from_backbone':True,'all_normal_converter_rows':True,
       'statement_rows':sum(r['statement_rows'] for r in chosen),'segmentation_rows':sum(r['segmentation_rows'] for r in chosen),
       'observed_opcodes':sorted({op for r in chosen for op in r['observed_opcodes']}),'origins':dict(Counter(r['sample_origin'] for r in chosen))}
     (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
   progress('complete');(output/'complete.json').write_text(json.dumps({'complete':True,'files':len(accepted),'smoke_only':a.smoke})+'\n')
 finally:
  if sampler is not None:
   sampler.stdin.close()
   try: sampler.wait(timeout=15)
   except subprocess.TimeoutExpired: sampler.terminate();sampler.wait(timeout=5)
  close_compilers(compilers)
if __name__=='__main__':main()
