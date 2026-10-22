"""Audit all requested official release files without running any model."""
import argparse,json,time
from collections import Counter
from pathlib import Path
from dev_scripts.benchmark_full_retrain import DATASETS,VERSIONS,selection,pin

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();report={'complete':False,'datasets':{},'versions':list(VERSIONS)}
 for dataset in DATASETS:
  report['datasets'][dataset]={}
  for version in VERSIONS:
   chosen=selection(a.run,version,dataset)
   magics=Counter()
   for row in chosen['files']:
    path=a.run/'independent-benchmark'/('dataset-'+DATASETS[dataset])/row['relative']
    with path.open('rb') as stream:magics[stream.read(4).hex()]+=1
   report['datasets'][dataset][version]={'files':len(chosen['files']),'release':chosen['release'],'commit':chosen['commit'],'magic_counts':dict(magics),'source_hash_overlap':chosen['source_hash_overlap'],'non_overlapping_files':len(chosen['files'])-chosen['source_hash_overlap']}
 report['complete']=True;report['observed_at']=time.time();pin(a.run/'official-evaluation-release-audit.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
