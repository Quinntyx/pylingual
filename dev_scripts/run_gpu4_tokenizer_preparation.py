"""GPU4-local Docker compilers, mounted-NAS conversion, and tokenizer-only fitting.

No per-file SSH traffic. Images must already be cached; containers never pull.
Resume the existing data-100k journal, or use --smoke for an isolated 90-file test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import shlex
import signal
import subprocess
import threading
import uuid

from dev_scripts.run_remote_ablation import Compiler, Gateway, Handler


OWNER_LABEL = 'pi.pylingual.compile-owner'


def compiler_command(repo: Path, image: str, owner: str, uid: int, gid: int):
    return ['docker', 'run', '--rm', '--pull=never', '-i', '--network=none',
            '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--pids-limit=64', '--memory=512m', '--user', f'{uid}:{gid}',
            '--label', OWNER_LABEL+'='+owner,
            '--mount', f'type=bind,source={repo},target={repo},readonly',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', image, 'python']


def verify_runtime(version, expected, actual):
    if actual['version'].split()[0] != expected['version'].split()[0]:
        raise ValueError('Compiler patch version differs for '+version)
    if actual['magic'] != expected['magic']:
        raise ValueError('Compiler bytecode magic differs for '+version)
    if sorted(actual['opcodes']) != sorted(expected['opcodes']):
        raise ValueError('Compiler opcode table differs for '+version)


def close_compilers(compilers):
    # EOF ends the JSON-line servers normally, including Python running as PID 1.
    # Close every input first so the containers exit concurrently.
    for compiler in compilers:
        try:
            compiler.process.stdin.close()
        except OSError:
            pass
    for compiler in compilers:
        try:
            compiler.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            compiler.close()


def stop_owned_containers(owner):
    listed = subprocess.run(['docker', 'ps', '-q', '--filter', 'label='+OWNER_LABEL+'='+owner],
                            capture_output=True, text=True, timeout=10)
    ids = listed.stdout.split() if listed.returncode == 0 else []
    if ids:
        subprocess.run(['docker', 'stop', '--time', '2', *ids], timeout=20, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--nas-root', default='/syssec-nas1/pyc-research/datasets')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--compile-copies', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.compile_copies < 1:
        parser.error('Worker and compiler counts must be positive')
    root = args.root.resolve()
    repo = root/'repo'
    provenance = root/'compiler-provenance.json'
    previous = root/'compiler-provenance.before-gpu4-docker.json'
    original_bytes = (previous if previous.exists() else provenance).read_bytes()
    baseline = json.loads(original_bytes)
    owner = uuid.uuid4().hex
    wrappers = root/'compiler-wrappers'/owner
    wrappers.mkdir(parents=True)
    compilers, pools, metadata = [], {}, {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Requested stop: '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)

    try:
        for minor in range(6, 15):
            version = f'3.{minor}'
            image = 'python:'+baseline[version]['version'].split()[0]+'-slim'
            descriptor = json.loads(subprocess.run(['docker', 'image', 'inspect', image],
                                    capture_output=True, text=True, check=True, timeout=10).stdout)[0]
            command = compiler_command(repo, descriptor['Id'], owner, os.getuid(), os.getgid())
            wrapper = wrappers/('python'+version)
            wrapper.write_text('#!/bin/sh\nexec '+shlex.join(command)+' "$@"\n')
            wrapper.chmod(0o700)
            pools[version] = queue.Queue()
            for _ in range(args.compile_copies):
                compiler = Compiler(str(wrapper), str(repo/'dev_scripts/compile_source_server.py'))
                compilers.append(compiler)
                verify_runtime(version, baseline[version], compiler.metadata)
                if not compiler.request({'source': 'x = 1\n', 'filename': '<gpu4-preflight>'}).get('success'):
                    raise RuntimeError('Compile smoke failed: '+version)
                pools[version].put(compiler)
            metadata[version] = {**compiler.metadata, 'executable': str(wrapper),
                                 'servers': args.compile_copies, 'execution_host': platform.node(),
                                 'transport': 'GPU4 loopback + Docker stdin/stdout',
                                 'image': image, 'image_id': descriptor['Id'],
                                 'repo_digests': descriptor.get('RepoDigests', [])}
        if not previous.exists():
            previous.write_bytes(original_bytes)
        provenance.write_text(json.dumps(metadata, indent=2)+'\n')
        journal = root/'data-100k/accepted.jsonl'
        prefix = journal.read_bytes() if journal.exists() else b''
        transition = {'host': platform.node(), 'compiler_transport': 'GPU4-local; no SSH tunnel',
                      'original_provenance_sha256': hashlib.sha256(original_bytes).hexdigest(),
                      'current_provenance_sha256': hashlib.sha256(provenance.read_bytes()).hexdigest(),
                      'preserved_files': prefix.count(b'\n'), 'prefix_sha256': hashlib.sha256(prefix).hexdigest(),
                      'exact_python_patch_versions_magic_and_opcodes_match': True,
                      'model_training_started': False}
        (root/('compiler-transition-smoke.json' if args.smoke else 'compiler-transition.json')).write_text(json.dumps(transition, indent=2)+'\n')
        print(json.dumps({'compiler_preflight': True, 'servers': len(compilers), **transition}), flush=True)
        if args.preflight_only:
            return
        data = root/('data-docker-smoke90' if args.smoke else 'data-100k')
        output = root/('tokenizers-docker-smoke90-'+owner[:8] if args.smoke else 'tokenizers')
        python = repo/'.venv/bin/python'
        env = {**os.environ, 'PYTHONPATH': str(repo), 'OMP_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'RAYON_NUM_THREADS': '24', 'TOKENIZERS_PARALLELISM': 'true'}
        with Gateway(('127.0.0.1', 0), Handler) as gateway:
            gateway.compiler_pools = pools
            threading.Thread(target=gateway.serve_forever, daemon=True).start()
            try:
                prep = [str(python), '-u', '-m', 'dev_scripts.prepare_tokenizer_corpus',
                        '--output', str(data), '--exclusions', str(root/'excluded-source-identities.json'),
                        '--source-script', str(repo/'dev_scripts/sample_nas_sources.py'),
                        '--source-reserved', str(root/'reserved-benchmark-sources.json'),
                        '--known-hosts', str(root/'source-host.known_hosts'),
                        '--gateway-port', str(gateway.server_address[1]), '--workers', str(args.workers),
                        '--files', '90' if args.smoke else '100000', '--nas-root', args.nas_root,
                        '--nas-catalog', str(root/'nas-catalog.json.gz')]
                subprocess.run(prep, cwd=repo, env=env, check=True)
            finally:
                gateway.shutdown()
        subprocess.run([str(python), '-u', '-m', 'dev_scripts.train_generic_tokenizers',
                        '--data', str(data), '--output', str(output),
                        '--template', str(repo/'dev_scripts/statement/tokenizer'),
                        '--compiler-provenance', str(provenance),
                        '--vocab-size', '2048' if args.smoke else '30000'], cwd=repo, env=env, check=True)
        print(json.dumps({'complete': True, 'tokenizers': str(output), 'model_training_started': False}), flush=True)
    finally:
        close_compilers(compilers)
        stop_owned_containers(owner)


if __name__ == '__main__':
    main()
