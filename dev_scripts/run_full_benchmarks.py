"""Wait for full training, then run every independent benchmark arm over SSH."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time

from dev_scripts.run_remote_ablation import Compiler, Gateway, Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--remote-python', required=True)
    parser.add_argument('--training-host', default='pylingual-train')
    args = parser.parse_args()
    study_path = args.remote_root + '/models-50k/study.json'
    while True:
        command = 'if test -f ' + shlex.quote(study_path) + '; then cat ' + shlex.quote(study_path) + '; else printf "{}"; fi'
        probe = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', args.training_host, command], capture_output=True, text=True)
        if probe.returncode:
            print(json.dumps({'waiting_for_training_connection': probe.stderr[-250:]}), flush=True)
            time.sleep(30)
            continue
        study = json.loads(probe.stdout)
        if any(stage['state'] == 'failed' for stage in study.get('stages', [])):
            raise RuntimeError('Training failed; refusing to benchmark incomplete models')
        if study.get('complete'):
            break
        print(json.dumps({'waiting_for_complete_training': True, 'stages': [{'task': stage['task'], 'role': stage['role'], 'state': stage['state']} for stage in study.get('stages', [])]}), flush=True)
        time.sleep(60)
    compilers = []
    try:
        pools, metadata = {}, {}
        for version in ('3.14', '3.15'):
            executable = shutil.which('python' + version)
            if not executable:
                raise RuntimeError('Existing compiler unavailable: ' + version)
            compiler = Compiler(executable, str(args.repo / 'dev_scripts/compile_source_server.py'))
            compilers.append(compiler)
            if not compiler.metadata['version'].startswith(version + '.'):
                raise RuntimeError('Compiler version mismatch')
            pools[version] = queue.Queue()
            pools[version].put(compiler)
            metadata[version] = {'executable': executable, **compiler.metadata}
        args.workspace.mkdir(parents=True, exist_ok=True)
        (args.workspace / 'benchmark-compiler-provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
        subprocess.run(['scp', str(args.workspace / 'benchmark-compiler-provenance.json'), args.training_host + ':' + args.remote_root + '/'], check=True)
        with Gateway(('127.0.0.1', 0), Handler) as gateway:
            gateway.compiler_pools = pools
            thread = threading.Thread(target=gateway.serve_forever, daemon=True)
            thread.start()
            port = gateway.server_address[1]
            try:
                for arm in ('control', 'no_adapter', 'statement_adapter', 'adapted'):
                    for version in ('3.14', '3.15', None):
                        command = ['env', 'PYTHONPATH=.', 'PYLINGUAL_COMPILE_GATEWAY_PORT=48662', 'TOKENIZERS_PARALLELISM=false', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=1', args.remote_python, '-u', '-m', 'dev_scripts.benchmark_full_ablation', '--models', args.remote_root + '/models-50k', '--data', args.remote_root + '/data-50k', '--dataset', args.remote_root + '/benchmark-dataset', '--output', args.remote_root + '/benchmark-50k', '--arm', arm]
                        command.extend(['--version', version] if version else ['--performance'])
                        remote = 'cd ' + shlex.quote(args.remote_root + '/repo') + ' && ' + shlex.join(command)
                        print(json.dumps({'starting_benchmark': arm, 'version': version or 'resident-performance'}), flush=True)
                        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-o', 'ExitOnForwardFailure=yes', '-R', '127.0.0.1:48662:127.0.0.1:' + str(port), args.training_host, remote], check=True)
            finally:
                gateway.shutdown()
        print(json.dumps({'all_benchmarks_complete': True, 'remote_output': args.remote_root + '/benchmark-50k'}), flush=True)
    finally:
        for compiler in compilers:
            compiler.close()


if __name__ == '__main__':
    main()
