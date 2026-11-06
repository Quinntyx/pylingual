import json,os,socket,tempfile
from pathlib import Path
from dev_scripts.benchmark_full_retrain import configuration
from pylingual.models import load_models
from pylingual.utils.version import PythonVersion
root=Path('/home/henryh/pylingual-tokenizer-100k-20261019/full-retrain-20261020')
for version in ('3.14','3.15'):
 with socket.create_connection(('127.0.0.1',45616),timeout=10) as connection:
  stream=connection.makefile('rwb');stream.write((json.dumps({'version':version,'source':'x = 1\n','filename':'<benchmark-preflight>'})+'\n').encode());stream.flush();response=json.loads(stream.readline())
  assert response['success'],response
 print(json.dumps({'compiler_verified':version}),flush=True)
# The ordinary local-artifact loader is the actual path used by the control arm.
config={'v3.15':{'STATEMENT_MODEL':{'REPO':str(root/'control-path-smoke/statement/checkpoint'),'REVISION':None,'TOKENIZER':str(root/'control-tokenizer-smoke/3.15/statement')},'SEGMENTATION_MODEL':{'REPO':str(root/'control-path-smoke/segmentation/checkpoint'),'REVISION':None,'TOKENIZER':str(root/'control-tokenizer-smoke/3.15/segmentation')}}}
# Locate the existing bounded tokenizer smoke export deterministically.
for task in ('statement','segmentation'):
 path=Path(config['v3.15'][task.upper()+'_MODEL']['TOKENIZER'])
 if not path.exists(): raise FileNotFoundError(path)
with tempfile.TemporaryDirectory() as temporary:
 path=Path(temporary)/'runtime.json';path.write_text(json.dumps(config))
 segmenter,translator=load_models(config_file=path,version=PythonVersion('3.15'))
 translator.generation_kwargs={'max_new_tokens':4}
 assert len(translator(['LOAD_CONST 1']))==1
 assert segmenter(['LOAD_CONST 1 <SEP> RETURN_VALUE'],batch_size=1)
 print(json.dumps({'ordinary_local_model_load_and_inference':True,'device':str(translator.device)}),flush=True)
