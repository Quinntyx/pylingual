import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from datasets import Dataset
from dev_scripts.opcode_adapter_training import load_encoded_dataset

class EncodedAdapterDataTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.selection=self.root/'selected.jsonl';self.selection.write_text(json.dumps({'version':'3.15','file':'fixture.py'})+'\n')
  data=Dataset.from_list([{'input_ids':[2,5,3],'attention_mask':[1,1,1],'labels':[5,6,3]}]);data.save_to_disk(self.root/'dataset')
  self.doc={'scope':'adapter','task':'statement','split':'train','complete':True,'files':1,'all_converter_rows':True,
   'limits':{'statement_input':256,'statement_target':256,'segmentation_window':512,'stride':128},
   'smoke_only':True,'journal_sha256':hashlib.sha256(self.selection.read_bytes()).hexdigest(),'registry_fingerprint':'registry',
   'tokenizer_semantic_fingerprint':'tokenizer','bindings':{},'versions':{'3.15':1},'encoded_rows':1,
   'dataset_fingerprint':json.loads((self.root/'dataset/state.json').read_text())['_fingerprint']}
  self.args=SimpleNamespace(version='3.15',steps=4);self.bank=SimpleNamespace(registry=SimpleNamespace(fingerprint='registry'))
  self.mock=patch('pylingual.opcodes.tokenizer_fingerprint',return_value='tokenizer');self.mock.start();self.addCleanup(self.mock.stop)
 def load(self,**updates):
  doc={**self.doc,**updates};(self.root/'manifest.json').write_text(json.dumps(doc))
  return load_encoded_dataset(self.root,self.selection,self.args,self.bank,object(),{'task':'statement'}, {},'train')
 def test_loads_persisted_arrow_and_handles_restored_fingerprint(self): self.assertEqual(len(self.load()),1)
 def test_rejects_changed_source_identity(self):
  with self.assertRaises(ValueError): self.load(journal_sha256='changed')
 def test_rejects_changed_tokenizer_or_registry(self):
  for field in ('tokenizer_semantic_fingerprint','registry_fingerprint'):
   with self.subTest(field=field),self.assertRaises(ValueError): self.load(**{field:'changed'})
 def test_rejects_version_binding_partition_or_task_change(self):
  for update in ({'versions':{'3.14':1}},{'bindings':{'NEW':0}},{'split':'validation'},{'task':'segmentation'},{'scope':'control'}):
   with self.subTest(update=update),self.assertRaises(ValueError): self.load(**update)
 def test_rejects_smoke_data_for_full_training(self):
  self.args.steps=None
  with self.assertRaises(ValueError): self.load()
 def test_rejects_row_count_or_stored_fingerprint_change(self):
  for update in ({'encoded_rows':2},{'dataset_fingerprint':'changed'}):
   with self.subTest(update=update),self.assertRaises(ValueError): self.load(**update)
 def test_rejects_changed_budget_limits_or_converter_coverage(self):
  for update in ({'files':2},{'limits':{}},{'all_converter_rows':False}):
   with self.subTest(update=update),self.assertRaises(ValueError): self.load(**update)
if __name__=='__main__': unittest.main()
