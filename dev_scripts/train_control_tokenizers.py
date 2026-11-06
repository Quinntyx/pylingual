"""Fit ordinary per-version PyLingual tokenizers, without shared opcode placeholders."""
from __future__ import annotations
import argparse,gzip,hashlib,json,time
from pathlib import Path
from tokenizers import Tokenizer,decoders,models,normalizers,pre_tokenizers,processors,trainers
from transformers import AutoTokenizer,PreTrainedTokenizerFast

def iterator(records,task):
 for record in records:
  with gzip.open(record['chunk'],'rt') as stream: rows=json.load(stream)[task]
  yield [row[1] for row in rows]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--template',type=Path,required=True)
 p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
 source=json.loads((a.data/'manifest.json').read_text());records=[json.loads(line) for line in (a.data/'accepted.jsonl').open()]
 if not source['complete'] or len(records)!=source['files']: raise ValueError('Incomplete source corpus')
 if bool(source.get('smoke_only',False))!=a.smoke_only: raise ValueError('Wrong tokenizer fitting scope')
 if not a.smoke_only and (source['files']!=54500 or source['split']!='train' or source['corpora']!={'pypi':49050,'csn':5450}):
  raise ValueError('Ordinary tokenizer requires54,500trainingfiles,90/10mixture')
 identity={'journal_sha256':hashlib.sha256((a.data/'accepted.jsonl').read_bytes()).hexdigest(),'files':len(records),
  'version':source['version'],'vocab_budget':30000,'ordinary_version_specific':True,'smoke_only':a.smoke_only}
 a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'manifest.json').exists():
  old=json.loads((a.output/'manifest.json').read_text())
  if not old['complete'] or any(old.get(k)!=v for k,v in identity.items()): raise ValueError('Tokenizer output identity changed')
  for task,record in old['tasks'].items():
   if hashlib.sha256((a.output/task/'tokenizer.json').read_bytes()).hexdigest()!=record['tokenizer_json_sha256']:
    raise ValueError('Saved tokenizer changed')
  print(json.dumps({'reused':str(a.output)}),flush=True);return
 start=time.monotonic();summary={**identity,'complete':False,'tasks':{}}
 for task in ('statement','segmentation'):
  task_start=time.monotonic();print(json.dumps({'fitting':task,'files':len(records),'version':source['version']}),flush=True)
  if task=='statement':
   tokenizer=AutoTokenizer.from_pretrained(a.template).train_new_from_iterator(iterator(records,task),vocab_size=30000)
   tokenizer.model_max_length=256
  else:
   backend=Tokenizer(models.WordPiece(unk_token='[UNK]'))
   backend.normalizer=normalizers.Sequence([normalizers.NFD(),normalizers.StripAccents()])
   backend.pre_tokenizer=pre_tokenizers.Split('<SEP>','removed')
   backend.train_from_iterator(iterator(records,task),trainer=trainers.WordPieceTrainer(vocab_size=30000,special_tokens=['[UNK]','[PAD]','[CLS]','[SEP]','[MASK]']))
   backend.decoder=decoders.WordPiece(prefix='##')
   backend.post_processor=processors.TemplateProcessing(single='[CLS]:0 $A:0 [SEP]:0',pair='[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1',
    special_tokens=[('[CLS]',backend.token_to_id('[CLS]')),('[SEP]',backend.token_to_id('[SEP]'))])
   tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',pad_token='[PAD]',cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]',model_max_length=1024)
  path=a.output/task;path.mkdir(exist_ok=True);tokenizer.save_pretrained(path)
  if AutoTokenizer.from_pretrained(path).get_vocab()!=tokenizer.get_vocab(): raise ValueError('Tokenizer reload changed vocabulary')
  summary['tasks'][task]={'vocab_size':len(tokenizer),'seconds':time.monotonic()-task_start,
   'tokenizer_json_sha256':hashlib.sha256((path/'tokenizer.json').read_bytes()).hexdigest()}
  (a.output/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
 summary.update(complete=True,seconds=time.monotonic()-start)
 (a.output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
