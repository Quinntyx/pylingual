"""Controller for direct source-to-training-host Python3.14 LoRA data expansion."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import threading
from dev_scripts.run_remote_ablation import Compiler, Gateway, Handler, run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--remote-python',required=True)
    parser.add_argument('--output-name',default='data-lora314-plus50k')
    parser.add_argument('--source-host',default='syssec-adwaith1')
    parser.add_argument('--training-host',default='pylingual-train')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--compile-copies',type=int,default=8)
    parser.add_argument('--new-files',type=int,default=50000)
    parser.add_argument('--smoke-original-files',type=int,default=0)
    args=parser.parse_args()
    if not os.environ.get('SSH_AUTH_SOCK'):
        raise RuntimeError('Launch under ssh-agent')
    run(['ssh-add',Path.home()/'.ssh/id_ed25519'],stdin=subprocess.DEVNULL)
    known_hosts=args.workspace/'source-host.known_hosts'
    reserved=args.workspace/'reserved-benchmark-sources.json'
    if not known_hosts.is_file() or not reserved.is_file():
        raise RuntimeError('Existing pinned SSH trust and benchmark exclusions required')
    source_home=subprocess.run(['ssh','-o','BatchMode=yes',args.source_host,'printf %s "$HOME"'],capture_output=True,text=True,check=True).stdout.strip()
    if not source_home.startswith('/home/') or '\n' in source_home:
        raise RuntimeError('Unexpected source home')
    source_root=source_home+'/pylingual-lora314-50k-20261019'
    output=args.remote_root+'/'+args.output_name
    run(['ssh',args.source_host,'mkdir','-p',source_root])
    run(['scp',args.repo/'dev_scripts/sample_remote_sources.py',reserved,args.source_host+':'+source_root+'/'])
    run(['ssh',args.training_host,'mkdir','-p',output])
    scripts=[args.repo/'dev_scripts'/name for name in ('prepare_lora_expansion.py','lora_expansion_data.py')]
    run(['scp',*scripts,args.training_host+':'+args.remote_root+'/repo/dev_scripts/'])
    compilers=[]
    try:
        executable=shutil.which('python3.14')
        if not executable:
            raise RuntimeError('Existing Python3.14 compiler missing')
        pool=queue.Queue()
        for _ in range(args.compile_copies):
            compiler=Compiler(executable,str(args.repo/'dev_scripts/compile_source_server.py'))
            compilers.append(compiler)
            if not compiler.metadata['version'].startswith('3.14.'):
                raise RuntimeError('Compiler version mismatch')
            pool.put(compiler)
        provenance={'3.14':{'executable':executable,**compilers[0].metadata,'servers':args.compile_copies}}
        provenance_file=args.workspace/('compiler-'+args.output_name+'.json')
        provenance_file.write_text(json.dumps(provenance,indent=2)+'\n')
        run(['scp',provenance_file,args.training_host+':'+output+'/compiler-provenance.json'])
        with Gateway(('127.0.0.1',0),Handler) as gateway:
            gateway.compiler_pools={'3.14':pool}
            threading.Thread(target=gateway.serve_forever,daemon=True).start()
            command=['env','PYTHONPATH=.','TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',args.remote_python,'-u','-m','dev_scripts.prepare_lora_expansion','--output',output,'--existing-data',args.remote_root+'/data-50k','--models',args.remote_root+'/models-50k','--source-script',source_root+'/sample_remote_sources.py','--source-reserved',source_root+'/reserved-benchmark-sources.json','--source-known-hosts',args.remote_root+'/source-host.known_hosts','--gateway-port','48663','--workers',str(args.workers),'--new-files',str(args.new_files),'--smoke-original-files',str(args.smoke_original_files)]
            remote='cd '+shlex.quote(args.remote_root+'/repo')+' && '+shlex.join(command)
            try:
                run(['ssh','-A','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-R','127.0.0.1:48663:127.0.0.1:'+str(gateway.server_address[1]),args.training_host,remote])
            finally:
                gateway.shutdown()
        print(json.dumps({'preparation_complete':True,'remote_output':output}),flush=True)
    finally:
        for compiler in compilers:
            compiler.close()

if __name__=='__main__':
    main()
