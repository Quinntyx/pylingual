from pathlib import Path
import json
from dev_scripts.run_full_retrain_benchmarks import compiler_preflight,benchmark_jobs,compiler_port
run=Path('/home/henryh/pylingual-tokenizer-100k-20261019/full-retrain-20261020')
compiler_preflight(compiler_port(run))
audit=json.loads((run/'official-evaluation-release-audit.json').read_text())
assert audit['complete'] and set(audit['versions'])=={'3.14','3.15'}
print(json.dumps({'compiler_preflight':'passed','official_files':{dataset:{version:record['files'] for version,record in rows.items()} for dataset,rows in audit['datasets'].items()},'queued_benchmark_jobs':len(benchmark_jobs(run/'benchmark'))}),flush=True)
