"""Universal-only statement generation on disjoint 3.14 validation files.

This is a teacher-forcing-free statement diagnostic, NOT full-file equivalence.
No model weights or training processes are modified.
"""
from __future__ import annotations
import argparse,ast,gzip,hashlib,io,json,os,re,time,tokenize
from collections import Counter
from pathlib import Path

MASK=re.compile(r"<mask_(\d+)>")
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def ast_key(text):
    try:return ast.dump(ast.parse(MASK.sub(r'MASK_\1',text.strip())),include_attributes=False)
    except (SyntaxError,ValueError,RecursionError):return None
def token_key(text):
    try:
        ignored={tokenize.ENCODING,tokenize.NL,tokenize.NEWLINE,tokenize.ENDMARKER,tokenize.INDENT,tokenize.DEDENT,tokenize.COMMENT}
        return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(MASK.sub(r'MASK_\1',text.strip())).readline) if t.type not in ignored]
    except (tokenize.TokenError,IndentationError,SyntaxError):return None

def prepare(args):
    from transformers import AutoTokenizer
    from pylingual.opcodes import OpcodeRegistry
    source=args.run/'target-corpora/3.14/validation/accepted.jsonl'
    records=[json.loads(line) for line in source.open() if line.strip()]
    base=[json.loads(line) for line in (args.run.parent/'data-100k/accepted.jsonl').open() if line.strip()]
    target_train=[json.loads(line) for line in (args.run/'target-corpora/3.14/train/accepted.jsonl').open() if line.strip()]
    assert len(records)==500 and len(base)==100000 and len(target_train)==54500
    valid_ids={r['canonical_sha256'] for r in records}
    assert len(valid_ids)==500
    assert not valid_ids.intersection(r['canonical_sha256'] for r in base+target_train)
    family=args.run/'families-base/statement';metadata=json.loads((family/'family.json').read_text())
    assert metadata['complete'] and not metadata['adapters'] and metadata['held_out_version']=='3.15'
    tokenizer=AutoTokenizer.from_pretrained(family/'tokenizer')
    registry=OpcodeRegistry.load(family/'tokenizer/opcode_registry.json',tokenizer)
    rows=[];inputs={};targets={}
    for record in records:
        with gzip.open(record['chunk'],'rt') as stream:examples=json.load(stream)['statement']
        assert len(examples)==record['statement_rows']
        for target,bytecode,*_ in examples:
            key=hashlib.sha256(bytecode.encode()).hexdigest()
            if key not in inputs:
                inputs[key]={'key':key,'bytecode':bytecode,'input_tokens':len(tokenizer(registry.bind(bytecode))['input_ids'])}
                targets[key]=set()
            else:assert inputs[key]['bytecode']==bytecode
            targets[key].add(target)
            rows.append({'source_id':record['source_sha512'],'key':key,'target':target,'target_tokens':len(tokenizer(target)['input_ids'])})
    expected=json.loads((args.run/'encoded/adapter/3.14/statement/validation/manifest.json').read_text())['raw_examples']
    assert len(rows)==expected
    manifest={'version':'3.14','compiler':'3.14.6','adapters_enabled':False,'files':len(records),'rows':len(rows),'unique_inputs':len(inputs),
      'ambiguous_inputs':sum(len(v)>1 for v in targets.values()),'source_overlap_with_backbone_or_target_train':0,
      'validation_journal_sha256':digest(source),'family':str(family),'family_metadata':metadata,
      'backbone_sha256':digest(family/'backbone/model.safetensors'),'precision':'FP32','max_generation_length':512,
      'source_boundaries':'ground-truth converter boundaries; segmentation model not involved',
      'selection':'all statement examples from all 500 target-validation source files; no length-based filtering',
      'normal_inference':'VersionedTranslator(use_adapter=False), greedy normal generation, max_length=512',
      'inputs_exceeding_training256':sum(i['input_tokens']>256 for i in inputs.values()),'targets_exceeding_training256':sum(r['target_tokens']>256 for r in rows)}
    args.output.mkdir(parents=True,exist_ok=True)
    save(args.output/'plan.json',{'manifest':manifest,'inputs':list(inputs.values()),'rows':rows})
    save(args.output/'manifest.json',manifest)
    print(json.dumps(manifest),flush=True)

def worker(args):
    import torch
    from pylingual.shared_models import load_family
    from pylingual.models import VersionedTranslator
    torch.set_num_threads(2)
    assert torch.cuda.device_count()==1,'Assign exactly one physical GPU per worker'
    plan=json.loads((args.output/'plan.json').read_text());manifest=plan['manifest'];family=Path(manifest['family'])
    bank,tokenizer,metadata,_=load_family(family,'statement',torch.device('cuda'))
    assert not bank.specs and metadata['backbone_id']==manifest['family_metadata']['backbone_id']
    assert digest(family/'backbone/model.safetensors')==manifest['backbone_sha256']
    model=bank.model;model.eval();assert next(model.parameters()).dtype==torch.float32
    translator=VersionedTranslator(bank,tokenizer,torch.device('cuda'),'3.14',metadata.get('source_prefix',''),use_adapter=False)
    inputs=[row for index,row in enumerate(sorted(plan['inputs'],key=lambda row:row['key'])) if index%args.shards==args.shard]
    inputs.sort(key=lambda row:(row['input_tokens'],row['key']))
    predictions={};started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    prediction_file=args.output/('predictions-'+str(args.shard)+'.jsonl')
    if prediction_file.exists():
        predictions={row['key']:row for line in prediction_file.open() if (row:=json.loads(line))}
    inputs=[row for row in inputs if row['key'] not in predictions]
    with prediction_file.open('a') as stream,torch.inference_mode():
        for offset in range(0,len(inputs),32):
            batch=inputs[offset:offset+32];tick=time.monotonic()
            try:
                outputs=translator([row['bytecode'] for row in batch]);assert len(outputs)==len(batch)
                results=[{'key':row['key'],'prediction':text,'input_tokens':row['input_tokens'],'error':None} for row,text in zip(batch,outputs)]
            except torch.cuda.OutOfMemoryError:raise
            except Exception as error:
                results=[]
                for row in batch:
                    try:results.append({'key':row['key'],'prediction':translator([row['bytecode']])[0],'input_tokens':row['input_tokens'],'error':None})
                    except Exception as inner:results.append({'key':row['key'],'prediction':'','input_tokens':row['input_tokens'],'error':repr(inner)})
            for result in results:
                predictions[result['key']]=result;stream.write(json.dumps(result)+'\n')
            stream.flush()
            save(args.output/('worker-'+str(args.shard)+'.json'),{'phase':'running','shard':args.shard,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'processed_unique_inputs':len(predictions),'seconds':time.monotonic()-started,'last_batch_seconds':time.monotonic()-tick,'backbone_sha256':manifest['backbone_sha256']})
            if offset==0 or offset%512==0:print(json.dumps({'shard':args.shard,'processed_unique_inputs':len(predictions),'seconds':time.monotonic()-started,'sample_predictions':[r['prediction'] for r in results[:3]]}),flush=True)
    save(args.output/('worker-'+str(args.shard)+'.json'),{'phase':'complete','shard':args.shard,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
       'unique_inputs':len(predictions),'seconds':time.monotonic()-started,'peak_cuda_MiB':torch.cuda.max_memory_allocated()/2**20,
       'backbone_sha256':manifest['backbone_sha256'],'adapters_enabled':False})
    print(json.dumps({'complete':True,'shard':args.shard,'unique_inputs':len(predictions)}),flush=True)

def combine(args):
    plan=json.loads((args.output/'plan.json').read_text());manifest=json.loads((args.output/'manifest.json').read_text());predictions={};workers=[]
    for shard in range(args.shards):
        report=json.loads((args.output/('worker-'+str(shard)+'.json')).read_text());assert report['phase']=='complete' and not report['adapters_enabled']
        assert report['backbone_sha256']==manifest['backbone_sha256'];workers.append(report)
        for line in (args.output/('predictions-'+str(shard)+'.jsonl')).open():
            row=json.loads(line);assert row['key'] not in predictions;predictions[row['key']]=row
    assert set(predictions)=={row['key'] for row in plan['inputs']}
    metrics=Counter();examples=[];by_length={'within_training256':Counter(),'outside_training256':Counter()};targets={}
    for row in plan['rows']:targets.setdefault(row['key'],set()).add(row['target'])
    unique_scores={}
    for row in plan['rows']:
        answer=predictions[row['key']];prediction=answer['prediction'];target=row['target'];expected_ast=ast_key(target);predicted_ast=ast_key(prediction)
        exact=prediction.strip()==target.strip();expected_tokens=token_key(target);actual_tokens=token_key(prediction)
        matched_tokens=expected_tokens is not None and actual_tokens==expected_tokens
        comparable=expected_ast is not None;matched_ast=comparable and predicted_ast==expected_ast
        counts={'rows':1,'exact_text':int(exact),'python_token_match':int(matched_tokens),'ast_comparable':int(comparable),'ast_match':int(matched_ast),
                'generation_errors':int(bool(answer['error'])),'runtime_fallback_predictions':int('Decompiler error: line too long' in prediction),'empty_predictions':int(not prediction.strip()),'any_reference_text':int(prediction.strip() in {v.strip() for v in targets[row['key']]})}
        metrics.update(counts);group='within_training256' if answer['input_tokens']<=256 and row['target_tokens']<=256 else 'outside_training256';by_length[group].update(counts)
        unique_scores.setdefault(row['key'],{'exact_text':False,'python_token_match':False,'ast_match':False})
        for k,score in [('exact_text',exact),('python_token_match',matched_tokens),('ast_match',matched_ast)]:unique_scores[row['key']][k]|=score
        if not matched_tokens and len(examples)<30:examples.append({'bytecode':next(i['bytecode'] for i in plan['inputs'] if i['key']==row['key']),'expected':target,'prediction':prediction,'error':answer['error'],'input_tokens':answer['input_tokens']})
    def rates(counts):
        d=dict(counts);n=d['rows']
        for field in ['exact_text','python_token_match','any_reference_text']:d[field+'_rate']=d[field]/n if n else None
        d['ast_match_rate']=d.get('ast_match',0)/d['ast_comparable'] if d.get('ast_comparable') else None
        return d
    result={'complete':True,'analysis_script_sha256':digest(__file__),'manifest':manifest,'workers':workers,'metrics':rates(metrics),'by_training_length':{k:rates(v) for k,v in by_length.items() if v},
       'unique_input_any_reference_rates':{k:sum(v[k] for v in unique_scores.values())/len(unique_scores) for k in ['exact_text','python_token_match','ast_match']},
       'mismatch_examples':examples,'limitations':['Statement generation with gold boundaries, not full-file decompilation or bytecode equivalence.',
       'AST comparison covers only references that independently parse; incomplete control-flow headers are excluded from that metric.',
       'Text/token mismatches can still be semantically equivalent; any-reference scores use alternative targets observed for identical masked bytecode.']}
    save(args.output/'report.json',result);print(json.dumps(result),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','worker','combine']);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=2);args=parser.parse_args()
    {'prepare':prepare,'worker':worker,'combine':combine}[args.phase](args)
if __name__=='__main__':main()
