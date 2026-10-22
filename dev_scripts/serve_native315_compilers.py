"""RAM-only native compiler RPC; sampled packages are compiled, never executed."""
import argparse,json,queue,signal
from pathlib import Path
from dev_scripts.run_remote_ablation import Compiler,Gateway,Handler

def main():
 p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--metadata',type=Path,required=True)
 p.add_argument('--copies',type=int,default=4);p.add_argument('--port',type=int,default=0);a=p.parse_args()
 compilers=[]
 try:
  pool=queue.Queue()
  for _ in range(a.copies):
   compiler=Compiler(a.python,str(Path(__file__).with_name('compile_source_server.py')));compilers.append(compiler);pool.put(compiler)
  with Gateway(('127.0.0.1',a.port),Handler) as gateway:
   gateway.compiler_pools={'3.15':pool}
   metadata={**compilers[0].metadata,'port':gateway.server_address[1],'copies':a.copies,'source_disk_staging':False}
   a.metadata.parent.mkdir(parents=True,exist_ok=True);a.metadata.write_text(json.dumps(metadata,indent=2)+'\n')
   print(json.dumps(metadata),flush=True);gateway.serve_forever()
 finally:
  for compiler in compilers: compiler.close()
if __name__=='__main__':main()
