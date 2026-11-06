"""Remote-only corpus transfer with persistent per-version compile servers.

Run under ssh-agent; the existing private key is loaded into that temporary
agent, never copied. Sampled source files never touch local disk.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import queue
import select
import shlex
import shutil
import socketserver
import subprocess
import threading
import time


class Compiler:
    def __init__(self, executable, script):
        self.executable = executable
        self.script = script
        self.process = subprocess.Popen([executable, '-u', script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.metadata = self.readline()

    def readline(self):
        ready, _, _ = select.select([self.process.stdout], [], [], 25)
        if not ready:
            self.process.terminate()
            raise TimeoutError('Compiler response timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Compiler exited: '+self.executable)
        return json.loads(line)

    def request(self, record):
        if self.process.poll() is not None:
            raise RuntimeError('Compiler is no longer running')
        self.process.stdin.write(json.dumps(record)+'\n')
        self.process.stdin.flush()
        return self.readline()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class Gateway(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        compiler = None
        pool = None
        try:
            request = json.loads(self.rfile.readline(16000000))
            pool = self.server.compiler_pools[request['version']]
            compiler = pool.get(timeout=30)
            response = compiler.request(request)
        except Exception as exc:
            response = {'success':False,'error':type(exc).__name__+': '+str(exc)}
        finally:
            if compiler is not None:
                if compiler.process.poll() is not None:
                    compiler.close()
                    compiler = Compiler(compiler.executable,compiler.script)
                pool.put(compiler)
        self.wfile.write((json.dumps(response)+'\n').encode())


def run(argv, **kwargs):
    print(json.dumps({'command':shlex.join(str(v) for v in argv)}),flush=True)
    return subprocess.run([str(v) for v in argv],check=True,**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--remote-python',required=True)
    parser.add_argument('--source-host',default='syssec-adwaith1')
    parser.add_argument('--training-host',default='pylingual-train')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--compile-copies',type=int,default=2)
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    args.workspace.mkdir(parents=True,exist_ok=True)
    if not os.environ.get('SSH_AUTH_SOCK'):
        raise RuntimeError('Run this coordinator under ssh-agent')
    run(['ssh-add',Path.home()/'.ssh/id_ed25519'],stdin=subprocess.DEVNULL)
    keys = subprocess.run(['ssh-keygen','-F','[syssec-docker0.utdallas.edu]:2227'],check=True,capture_output=True,text=True).stdout
    pinned = '\n'.join(line for line in keys.splitlines() if line and not line.startswith('#'))+'\n'
    if len(pinned.strip()) < 20:
        raise RuntimeError('No trusted source-host key available')
    known_hosts = args.workspace/'source-host.known_hosts'
    known_hosts.write_text(pinned)
    source_home = subprocess.run(['ssh','-o','BatchMode=yes',args.source_host,'printf %s "$HOME"'],check=True,capture_output=True,text=True).stdout.strip()
    if not source_home.startswith('/home/') or '\n' in source_home:
        raise RuntimeError('Unexpected source home path')
    source_root = source_home+'/pylingual-full50k'
    run(['ssh',args.source_host,'mkdir','-p',source_root])
    run(['scp',args.repo/'dev_scripts/sample_remote_sources.py',args.workspace/'reserved-benchmark-sources.json',args.source_host+':'+source_root+'/'])
    run(['ssh',args.training_host,'mkdir','-p',args.remote_root+'/repo/dev_scripts',args.remote_root+'/data-50k'])
    run(['scp',known_hosts,args.training_host+':'+args.remote_root+'/source-host.known_hosts'])
    tracked = subprocess.run(['git','-C',str(args.repo),'ls-files','-z'],check=True,capture_output=True).stdout.decode().split('\0')
    files = [p for p in tracked if p and (args.repo/p).is_file()]
    for script in ('prepare_remote_ablation.py','shared_training.py'):
        name = 'dev_scripts/'+script
        if name not in files:
            files.append(name)
    run(['rsync','-a','--from0','--files-from=-',str(args.repo)+'/',args.training_host+':'+args.remote_root+'/repo/'],input=('\0'.join(files)+'\0').encode())
    nested = ['ssh','-p','2227','-l','henryh@syssec.utdallas.edu','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+args.remote_root+'/source-host.known_hosts','syssec-docker0.utdallas.edu','hostname']
    run(['ssh','-A','-o','BatchMode=yes',args.training_host,shlex.join(nested)])
    compilers = []
    try:
        pools = {}
        metadata = {}
        for minor in range(6,16):
            version = '3.'+str(minor)
            executable = shutil.which('python'+version)
            if minor in (6,7):
                executable = str(Path.home()/'.pyenv/versions'/('3.6.15' if minor == 6 else '3.7.17')/'bin/python')
            if not executable or not Path(executable).is_file():
                raise RuntimeError('Existing compiler missing: '+version)
            pools[version] = queue.Queue()
            for _ in range(args.compile_copies):
                compiler = Compiler(executable,str(args.repo/'dev_scripts/compile_source_server.py'))
                compilers.append(compiler)
                if not compiler.metadata['version'].startswith(version+'.'):
                    raise RuntimeError('Compiler version mismatch')
                smoke = compiler.request({'source':'def add(a, b):\n    return a + b\n','filename':'<compile-smoke>'})
                if not smoke.get('success'):
                    raise RuntimeError('Compiler smoke test failed: '+str(smoke))
                pools[version].put(compiler)
            metadata[version] = {'executable':executable,**compiler.metadata,'servers':args.compile_copies}
        (args.workspace/'compiler-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps({'compiler_preflight':True,'versions':list(metadata),'servers':len(compilers),'remote_to_remote_source_access':True}),flush=True)
        if args.preflight:
            return
        with Gateway(('127.0.0.1',0),Handler) as gateway:
            gateway.compiler_pools = pools
            thread = threading.Thread(target=gateway.serve_forever,daemon=True)
            thread.start()
            local_port = gateway.server_address[1]
            remote_port = 48661
            command = 'cd '+shlex.quote(args.remote_root+'/repo')+' && '+shlex.join(['env','PYTHONPATH=.','TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',args.remote_python,'-u','-m','dev_scripts.prepare_remote_ablation','--output',args.remote_root+'/data-50k','--source-script',source_root+'/sample_remote_sources.py','--source-reserved',source_root+'/reserved-benchmark-sources.json','--source-known-hosts',args.remote_root+'/source-host.known_hosts','--gateway-port',str(remote_port),'--workers',str(args.workers)])
            try:
                run(['ssh','-A','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-R','127.0.0.1:'+str(remote_port)+':127.0.0.1:'+str(local_port),args.training_host,command])
            finally:
                gateway.shutdown()
            print(json.dumps({'data_preparation_complete':True,'remote_output':args.remote_root+'/data-50k'}),flush=True)
    finally:
        for compiler in compilers:
            compiler.close()


if __name__ == '__main__':
    main()
