"""Build 50k file-diverse examples on the GPU host from remote source samples.

Sources travel directly from the corpus host to this host. Compile requests
use an SSH-forwarded in-memory compiler gateway; no source is staged there.
"""
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
import random
import signal
import socket
import subprocess
import time
import tokenize
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from dev_scripts.dataset_generation.bytecode2csv import bytecode2csv_exception_wrapper
from dev_scripts.dataset_generation.normalize_source import normalize_source
from dev_scripts.shared_training import load_statement_tokenizer, source_prefix
from pylingual.masking.ast_masker import add_dummy_decorators

TOKENIZER = None


def build_example(arguments):
    record, version, root, gateway_port, seed = arguments
    try:
        raw = base64.b64decode(record['raw_b64'], validate=True)
        if hashlib.sha512(raw).hexdigest() != record['source_sha512']:
            raise ValueError('Source transport checksum mismatch')
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding)
        canonical = normalize_source(text, version=(3, 14), replace_docstrings=True)
        canonical_hash = hashlib.sha256(canonical.encode()).hexdigest()
        minor = int(version.split('.')[1])
        normalized = normalize_source(text, version=(3, minor), replace_docstrings=True)
        if minor <= 7:
            normalized = add_dummy_decorators(normalized)
        directory = Path(root)/'candidates'/version/record['source_sha512'][:2]/record['source_sha512']
        directory.mkdir(parents=True, exist_ok=True)
        py, pyc = directory/'source.py', directory/'source.pyc'
        py.write_text(normalized)
        with socket.create_connection(('127.0.0.1', gateway_port), timeout=35) as connection:
            stream = connection.makefile('rwb')
            stream.write((json.dumps({'version': version, 'source': normalized, 'filename': str(py)})+'\n').encode())
            stream.flush()
            response = json.loads(stream.readline(4000000))
        if not response['success']:
            return {'failure': 'compile-'+version, 'detail': response.get('error', '')[:180]}
        pyc.write_bytes(base64.b64decode(response['pyc_b64'], validate=True))
        rows = bytecode2csv_exception_wrapper((py, pyc))
        if isinstance(rows, Exception):
            return {'failure': 'extract-'+version, 'detail': str(rows)[:180]}
        segments, statements = rows
        rng = random.Random(seed ^ int(canonical_hash[:16], 16))
        rng.shuffle(statements)
        rng.shuffle(segments)
        chosen = None
        for source, bytecode, _ in statements:
            if len(TOKENIZER(source)['input_ids']) > 128:
                continue
            if len(TOKENIZER(source_prefix(version)+bytecode)['input_ids']) > 512:
                continue
            chosen = {'source': source, 'bytecode': bytecode}
            break
        if chosen is None or not segments:
            return {'failure': 'no-eligible-rows-'+version}
        segment = next((row for row in segments if len(row[1].split(' <SEP> ')) <= 256), min(segments, key=lambda row: len(row[1])))
        provenance = {k: record[k] for k in ('corpus','source_id','source_path','source_sha512')}
        provenance.update(version=version, canonical_sha256=canonical_hash, file=str(py))
        return {'provenance': provenance, 'statement': chosen, 'segmentation': {'source': segment[0], 'bytecode': segment[1], 'boundary': segment[2]}}
    except Exception as exc:
        return {'failure': type(exc).__name__+'-'+version, 'detail': str(exc)[:180]}


def main():
    global TOKENIZER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-script', required=True)
    parser.add_argument('--source-reserved', required=True)
    parser.add_argument('--source-known-hosts', required=True)
    parser.add_argument('--gateway-port', type=int, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261016)
    args = parser.parse_args()
    versions = tuple('3.'+str(v) for v in range(6,16))
    wanted = {'train': {'pypi':4050, 'csn':450}, 'validation': {'pypi':450, 'csn':50}}
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.output/'accepted.jsonl'
    accepted = []
    if journal.exists():
        for line in journal.read_text().splitlines():
            accepted.append(json.loads(line))
    counts = {v: {split: Counter() for split in wanted} for v in versions}
    seen = set()
    for entry in accepted:
        p = entry['provenance']
        counts[p['version']][entry['split']][p['corpus']] += 1
        seen.add(p['canonical_sha256'])
    failures = Counter()
    candidates = 0
    TOKENIZER = load_statement_tokenizer('Salesforce/codet5-base')
    TOKENIZER.add_tokens(['<mask_'+str(i)+'>' for i in range(300)])
    import torch
    torch.set_num_threads(1)
    ssh = ['ssh', '-p', '2227', '-l', 'henryh@syssec.utdallas.edu', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+args.source_known_hosts, 'syssec-docker0.utdallas.edu', 'python3', '-u', args.source_script, '--reserved', args.source_reserved, '--seed', str(args.seed)]
    # Resume with a new deterministic traversal and exclude accepted raw hashes
    # on the source host, so reconnects do not retransmit accepted source files.
    ssh.extend(['--resume-offset', str(len(accepted))])
    source = subprocess.Popen(ssh, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    sampler_primed = False
    reconnects = 0

    def sample_batch(corpus, count):
        nonlocal source, sampler_primed, reconnects
        for attempt in range(6):
            try:
                request = {'corpus': corpus, 'count': count}
                if not sampler_primed:
                    request['exclude_sha512'] = [entry['provenance']['source_sha512'] for entry in accepted]
                source.stdin.write(json.dumps(request)+'\n')
                source.stdin.flush()
                sampler_primed = True
                records = []
                while True:
                    line = source.stdout.readline()
                    if not line:
                        raise ConnectionError('Remote source sampler connection closed')
                    record = json.loads(line)
                    if record.get('batch_complete'):
                        if not record['emitted']:
                            raise RuntimeError('Exhausted '+corpus+' corpus before quotas were met')
                        return records
                    records.append(record)
            except (OSError, json.JSONDecodeError) as exc:
                if attempt == 5:
                    raise
                reconnects += 1
                print(json.dumps({'source_reconnect': reconnects, 'error': str(exc), 'accepted': len(accepted)}), flush=True)
                if source.poll() is None:
                    source.terminate()
                try:
                    source.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    source.kill()
                    source.wait()
                source = subprocess.Popen(ssh, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
                sampler_primed = False
                time.sleep(min(2**attempt, 15))
    cursor = Counter()
    try:
        with journal.open('a') as journal_stream, ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
            while True:
                needed = {corpus: [v for v in versions if any(counts[v][split][corpus] < wanted[split][corpus] for split in wanted)] for corpus in ('pypi','csn')}
                if not any(needed.values()):
                    break
                for corpus in ('pypi','csn'):
                    if not needed[corpus]:
                        continue
                    remaining = sum(max(0,wanted[split][corpus]-counts[v][split][corpus]) for v in versions for split in wanted)
                    request_count = min(450 if corpus == 'pypi' else 100, max(20, remaining))
                    batch = []
                    for record in sample_batch(corpus, request_count):
                        version = needed[corpus][cursor[corpus] % len(needed[corpus])]
                        cursor[corpus] += 1
                        batch.append((record,version,str(args.output),args.gateway_port,args.seed))
                    for result in pool.map(build_example, batch, chunksize=1):
                        candidates += 1
                        if 'failure' in result:
                            failures[result['failure']] += 1
                            if failures[result['failure']] <= 2:
                                print(json.dumps({'rejection':result}),flush=True)
                            continue
                        p = result['provenance']
                        split = 'validation' if int(p['canonical_sha256'][:8],16)%10 == 0 else 'train'
                        if p['canonical_sha256'] in seen:
                            failures['canonical_duplicate'] += 1
                            continue
                        if counts[p['version']][split][p['corpus']] >= wanted[split][p['corpus']]:
                            failures['filled_quota'] += 1
                            continue
                        result['split'] = split
                        seen.add(p['canonical_sha256'])
                        counts[p['version']][split][p['corpus']] += 1
                        accepted.append(result)
                        journal_stream.write(json.dumps(result)+'\n')
                    journal_stream.flush()
                    progress = {'accepted':len(accepted),'target':50000,'candidates':candidates,'versions':{v:sum(sum(c.values()) for c in counts[v].values()) for v in versions},'failures':dict(failures)}
                    (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                    print(json.dumps(progress),flush=True)
    finally:
        if source.stdin:
            source.stdin.close()
        try:
            source.wait(timeout=10)
        except subprocess.TimeoutExpired:
            source.terminate()
            source.wait(timeout=10)
    if len(accepted) != 50000:
        raise RuntimeError('Wrong final example count')
    subsets = {'backbone':versions[:-1], 'adapter315':('3.15',), 'control314':('3.14',), 'control315':('3.15',)}
    manifest = {'format':2,'complete':True,'seed':args.seed,'examples':50000,'statement_examples':50000,'segmentation_examples':50000,'sampling':'one eligible example per task per unique canonical source file','source_host':'syssec-adwaith1','source_roots':['/data2/data2_pypi_py_list_2023-06-03.txt','/data2/data2_csn_py_list_2023-06-03.txt'],'blend':{'pypi':0.9,'csn':0.1},'versions':versions,'counts':{v:{split:sum(counts[v][split].values()) for split in wanted} for v in versions},'corpus_counts':counts,'failures':failures,'candidates':candidates}
    compiler_provenance = args.output/'compiler-provenance.json'
    if compiler_provenance.exists():
        manifest['compiler_provenance_sha256'] = hashlib.sha256(compiler_provenance.read_bytes()).hexdigest()
    for name, subset_versions in subsets.items():
        directory = args.output/name
        directory.mkdir(exist_ok=True)
        selected = [r for r in accepted if r['provenance']['version'] in subset_versions]
        for task in ('statement','segmentation'):
            for split in wanted:
                with (directory/(task+'-'+split+'.jsonl')).open('w') as stream:
                    for entry in selected:
                        if entry['split'] == split:
                            stream.write(json.dumps({**entry['provenance'],**entry[task]})+'\n')
        subset_manifest = {**manifest,'examples':len(selected),'statement_examples':len(selected),'segmentation_examples':len(selected),'versions':subset_versions,'counts':{v:manifest['counts'][v] for v in subset_versions},'corpus_counts':{v:manifest['corpus_counts'][v] for v in subset_versions},'parent_manifest':'../manifest.json','files':[{'source_sha512':r['provenance']['source_sha512'],'source_id':r['provenance']['source_id'],'canonical_sha256':r['provenance']['canonical_sha256'],'version':r['provenance']['version'],'split':r['split']} for r in selected]}
        (directory/'manifest.json').write_text(json.dumps(subset_manifest,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'complete':True,'examples':len(accepted),'output':str(args.output)}),flush=True)


if __name__ == '__main__':
    main()
