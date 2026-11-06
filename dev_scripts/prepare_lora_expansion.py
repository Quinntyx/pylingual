"""Add 50k new Python3.14 training files without staging source on the controller."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import time
import tokenize
import warnings
from dev_scripts.lora_expansion_data import TASKS, open_database, insert_file, inventory

TOKENIZERS={}
EXCLUDED=set()

def initialize(statement_tokenizer, segmentation_tokenizer, excluded):
    global TOKENIZERS, EXCLUDED
    import torch
    from transformers import PreTrainedTokenizerFast
    from dev_scripts.shared_training import load_statement_tokenizer
    torch.set_num_threads(1)
    TOKENIZERS={'statement':load_statement_tokenizer(statement_tokenizer),'segmentation':PreTrainedTokenizerFast.from_pretrained(segmentation_tokenizer)}
    EXCLUDED=excluded
    warnings.filterwarnings('ignore',category=SyntaxWarning)

def extract(job):
    from dev_scripts.dataset_generation.bytecode2csv import bytecode2csv_exception_wrapper
    from dev_scripts.dataset_generation.normalize_source import normalize_source
    from dev_scripts.shared_training import encode_row
    record, output, port, original = job
    try:
        if original:
            provenance=record.copy()
            py=Path(record['file'])
            pyc=py.with_suffix('.pyc')
        else:
            raw=base64.b64decode(record['raw_b64'],validate=True)
            if hashlib.sha512(raw).hexdigest()!=record['source_sha512']:
                raise ValueError('Transport checksum mismatch')
            encoding,_=tokenize.detect_encoding(io.BytesIO(raw).readline)
            normalized=normalize_source(raw.decode(encoding),version=(3,14),replace_docstrings=True)
            canonical=hashlib.sha256(normalized.encode()).hexdigest()
            if canonical in EXCLUDED:
                return {'failure':'existing-canonical'}
            if int(canonical[:8],16)%10==0:
                return {'failure':'reserved-validation-partition'}
            directory=Path(output)/'sources'/canonical[:2]/canonical
            directory.mkdir(parents=True,exist_ok=True)
            py,pyc=directory/'source.py',directory/'source.pyc'
            py.write_text(normalized)
            with socket.create_connection(('127.0.0.1',port),timeout=35) as connection:
                stream=connection.makefile('rwb')
                stream.write((json.dumps({'version':'3.14','source':normalized,'filename':str(py)})+'\n').encode())
                stream.flush()
                response=json.loads(stream.readline(4000000))
            if not response.get('success'):
                return {'failure':'compile','detail':response.get('error','')[:180]}
            pyc.write_bytes(base64.b64decode(response['pyc_b64'],validate=True))
            provenance={key:record[key] for key in ('corpus','source_id','source_path','source_sha512')}
            provenance.update(version='3.14',canonical_sha256=canonical,file=str(py))
        extracted=bytecode2csv_exception_wrapper((py,pyc))
        if isinstance(extracted,Exception):
            return {'failure':'extract','detail':str(extracted)[:180]}
        segment_rows,statement_rows=extracted
        result={task:[] for task in TASKS}
        for source,bytecode,_ in statement_rows:
            encoded=encode_row('statement',TOKENIZERS['statement'],{'version':'3.14','source':source,'bytecode':bytecode})
            if encoded is not None:
                result['statement'].append(encoded)
        for source,bytecode,boundary,_ in segment_rows:
            encoded=encode_row('segmentation',TOKENIZERS['segmentation'],{'version':'3.14','source':source,'bytecode':bytecode,'boundary':boundary})
            if encoded is not None:
                result['segmentation'].append(encoded)
        if any(not result[task] for task in TASKS):
            return {'failure':'no-eligible-rows'}
        return {'provenance':provenance,'rows':result}
    except Exception as exc:
        return {'failure':type(exc).__name__,'detail':str(exc)[:180]}
    finally:
        signal.alarm(0)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--existing-data',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--source-script',required=True)
    parser.add_argument('--source-reserved',required=True)
    parser.add_argument('--source-known-hosts',required=True)
    parser.add_argument('--gateway-port',type=int,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--new-files',type=int,default=50000)
    parser.add_argument('--smoke-original-files',type=int,default=0)
    parser.add_argument('--seed',type=int,default=20261019)
    args=parser.parse_args()
    if args.new_files<10 or args.new_files%10:
        raise ValueError('File budget must be positive and divisible by ten')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.json').exists() and json.loads((args.output/'manifest.json').read_text()).get('complete'):
        print(json.dumps({'already_complete':True}),flush=True)
        return
    db=open_database(args.output/'examples.sqlite')
    originals=[]
    excluded=set()
    raw_excluded=set()
    with (args.existing_data/'accepted.jsonl').open() as stream:
        for line in stream:
            entry=json.loads(line)
            p=entry['provenance']
            excluded.add(p['canonical_sha256'])
            raw_excluded.add(p['source_sha512'])
            if p['version']=='3.14':
                originals.append((p,entry['split']))
    if len(originals)!=5000:
        raise ValueError('Expected original 4500 training + 500 validation files')
    if args.smoke_original_files:
        originals=[item for split in ('train','validation') for item in [pair for pair in originals if pair[1]==split][:args.smoke_original_files]]
    initialize(str(args.models/'universal/statement/tokenizer'),str(args.models/'universal/segmentation/tokenizer'),excluded)
    failures=Counter()
    known={row[0] for row in db.execute('SELECT canonical_sha256 FROM files')}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        pending=[(p,split) for p,split in originals if p['canonical_sha256'] not in known]
        for start in range(0,len(pending),64):
            batch=pending[start:start+64]
            for (provenance,split),result in zip(batch,pool.map(extract,[(p,str(args.output),args.gateway_port,True) for p,_ in batch],chunksize=1)):
                if 'failure' in result:
                    raise RuntimeError('Cannot reconstruct original file: '+str(result))
                insert_file(db,result['provenance'],split,'original',result['rows'])
            print(json.dumps({'original_files_ready':min(start+64,len(pending)),'original_files_total':len(pending)}),flush=True)
        ssh=['ssh','-p','2227','-l','henryh@syssec.utdallas.edu','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+args.source_known_hosts,'syssec-docker0.utdallas.edu','python3','-u',args.source_script,'--reserved',args.source_reserved,'--seed',str(args.seed)]
        process=None
        def connect():
            nonlocal process
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            count=db.execute("SELECT COUNT(*) FROM files WHERE scope='new'").fetchone()[0]
            process=subprocess.Popen(ssh+['--resume-offset',str(count)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            return False
        primed=connect()
        targets={'pypi':args.new_files*9//10,'csn':args.new_files//10}
        try:
            while True:
                counts=Counter(dict(db.execute("SELECT corpus,COUNT(*) FROM files WHERE scope='new' GROUP BY corpus")))
                if all(counts[corpus]==target for corpus,target in targets.items()):
                    break
                for corpus,target in targets.items():
                    remaining=target-counts[corpus]
                    if remaining<=0:
                        continue
                    for attempt in range(6):
                        try:
                            request={'corpus':corpus,'count':min(128,max(8,remaining))}
                            if not primed:
                                request['exclude_sha512']=list(raw_excluded|{r[0] for r in db.execute('SELECT source_sha512 FROM files')})
                            process.stdin.write(json.dumps(request)+'\n')
                            process.stdin.flush()
                            primed=True
                            records=[]
                            while True:
                                line=process.stdout.readline()
                                if not line:
                                    raise ConnectionError('Source sampler disconnected')
                                record=json.loads(line)
                                if record.get('batch_complete'):
                                    if not record['emitted']:
                                        raise RuntimeError('Corpus exhausted: '+corpus)
                                    break
                                records.append(record)
                            break
                        except (OSError,json.JSONDecodeError) as exc:
                            if attempt==5:
                                raise
                            print(json.dumps({'source_reconnect':attempt+1,'error':str(exc)}),flush=True)
                            primed=connect()
                            time.sleep(min(2**attempt,15))
                    for result in pool.map(extract,[(r,str(args.output),args.gateway_port,False) for r in records],chunksize=1):
                        if 'failure' in result:
                            failures[result['failure']]+=1
                            if failures[result['failure']]<=2:
                                print(json.dumps({'rejection':result}),flush=True)
                            continue
                        if counts[corpus]>=target:
                            break
                        p=result['provenance']
                        if p['canonical_sha256'] in excluded:
                            raise RuntimeError('Existing-file leakage')
                        try:
                            insert_file(db,p,'train','new',result['rows'])
                            counts[corpus]+=1
                        except sqlite3.IntegrityError:
                            failures['new-duplicate']+=1
                    progress={**inventory(db),'new_files':sum(counts.values()),'target_new_files':args.new_files,'failures':dict(failures)}
                    (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                    print(json.dumps(progress),flush=True)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    overlap=db.execute('SELECT canonical_sha256 FROM files WHERE scope="new"').fetchall()
    if any(row[0] in excluded for row in overlap):
        raise RuntimeError('Final canonical overlap')
    counts=inventory(db)
    manifest={'complete':True,'version':'3.14','new_training_files':args.new_files,'original_training_files':sum(p[1]=='train' for p in originals),'validation_files':sum(p[1]=='validation' for p in originals),'total_training_files':sum(p[1]=='train' for p in originals)+args.new_files,'mix_unit':'source files','new_training_mix':targets,'sampling':'all eligible rows per accepted file, not one row per file','max_input_tokens':512,'max_output_tokens':128,'backbone_unchanged':True,'canonical_exclusions':len(excluded),'raw_exclusions':len(raw_excluded),'new_existing_overlap':0,**counts,'failures':dict(failures)}
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':
    main()
