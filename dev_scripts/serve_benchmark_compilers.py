"""RAM-only native 3.14/3.15 compilers for reproducible file verification."""
import argparse,json,queue
from pathlib import Path
from dev_scripts.run_remote_ablation import Compiler,Gateway,Handler

def main():
 p=argparse.ArgumentParser();p.add_argument('--python314',required=True);p.add_argument('--python315',required=True)
 p.add_argument('--metadata',type=Path,required=True);p.add_argument('--copies',type=int,default=4);a=p.parse_args()
 workers=[];pools={};metadata={}
 try:
  for version,python in [('3.14',a.python314),('3.15',a.python315)]:
   pools[version]=queue.Queue()
   for _ in range(a.copies):
    worker=Compiler(python,str(Path(__file__).with_name('compile_source_server.py')));workers.append(worker)
    if not worker.metadata['version'].startswith(version+'.'): raise ValueError('Compiler version mismatch')
    pools[version].put(worker)
   metadata[version]={'executable':python,**worker.metadata}
  with Gateway(('127.0.0.1',0),Handler) as gateway:
   gateway.compiler_pools=pools
   record={'port':gateway.server_address[1],'copies_per_version':a.copies,'source_disk_staging':False,'compilers':metadata}
   a.metadata.parent.mkdir(parents=True,exist_ok=True);a.metadata.write_text(json.dumps(record,indent=2)+'\n')
   print(json.dumps(record),flush=True);gateway.serve_forever()
 finally:
  for worker in workers: worker.close()
if __name__=='__main__':main()
