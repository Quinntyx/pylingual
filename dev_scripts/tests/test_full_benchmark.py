import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dev_scripts.benchmark_full_retrain import configuration,pin,selection,require_full_models,dataset_directory

class FullBenchmarkTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
 def test_ordinary_control_uses_its_own_tokenizers_and_checkpoints(self):
  config=configuration(self.root,'control','3.15');self.assertNotIn('SHARED_BACKBONES',config)
  self.assertIn('/controls/3.15/statement/checkpoint',config['v3.15']['STATEMENT_MODEL']['REPO'])
 def test_shared_arms_change_only_adapter_enablement(self):
  configs=[configuration(self.root,arm,'3.15')['SHARED_BACKBONES'] for arm in ('no_adapter','statement_adapter','adapted')]
  self.assertEqual(len({c['STATEMENT_MODEL'] for c in configs}),1);self.assertEqual(len({c['SEGMENTATION_MODEL'] for c in configs}),1)
  self.assertEqual([(c['USE_STATEMENT_ADAPTERS'],c['USE_SEGMENTATION_ADAPTERS']) for c in configs],[(False,False),(True,False),(True,True)])
 def test_pinned_identity_rejects_changed_inputs(self):
  path=self.root/'pin.json';pin(path,{'x':1});pin(path,{'x':1})
  with self.assertRaises(ValueError):pin(path,{'x':2})
 def test_rejects_smoke_or_partial_controls_even_with_full_data(self):
  root=self.root/'controls/3.15'
  for task in ('statement','mlm','segmentation'):
   (root/task).mkdir(parents=True);(root/task/'result.json').write_text(json.dumps({'complete':True,'scope':'control','files':54500,'smoke_only':False,'epoch':2.0}))
  require_full_models(self.root,'control',['3.15'])
  for update in ({'smoke_only':True},{'epoch':0.5},{'files':20},{'scope':'universal'}):
   original={'complete':True,'scope':'control','files':54500,'smoke_only':False,'epoch':2.0}
   (root/'statement/result.json').write_text(json.dumps({**original,**update}))
   with self.subTest(update=update),self.assertRaises(ValueError):require_full_models(self.root,'control',['3.15'])
 def fixture(self,listing):
  dataset=dataset_directory(self.root,'v2');(dataset/'pylingualv2').mkdir(parents=True)
  path=dataset/'pylingualv2/python-3.15/identity/case.pyc';path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
  (dataset/'315-pyc-list.txt').write_text(listing)
  return 'pylingualv2/python-3.15/identity/case.pyc'
 def test_selection_uses_every_entry_and_hashes_bytes(self):
  relative=self.fixture('pylingualv2/python-3.15/identity/case.pyc\n')
  with patch('dev_scripts.benchmark_full_retrain.training_audit',return_value=(set(),{})),patch('subprocess.check_output',return_value='test-commit\n'):
   result=selection(self.root,'3.15');self.assertEqual(len(result['files']),1);self.assertEqual(result['files'][0]['relative'],relative)
 def test_selection_records_overlap_without_dropping_official_files(self):
  self.fixture('pylingualv2/python-3.15/identity/case.pyc\n')
  with patch('dev_scripts.benchmark_full_retrain.training_audit',return_value=({'identity'},{})),patch('subprocess.check_output',return_value='test-commit\n'):
   result=selection(self.root,'3.15')
  self.assertEqual(len(result['files']),1)
  self.assertEqual(result['source_hash_overlap'],1)
  self.assertTrue(result['files'][0]['known_training_or_validation_source_overlap'])
 def test_selection_rejects_duplicates_and_traversal(self):
  relative=self.fixture('pylingualv2/python-3.15/identity/case.pyc\n');listing=dataset_directory(self.root,'v2')/'315-pyc-list.txt'
  with patch('dev_scripts.benchmark_full_retrain.training_audit',return_value=(set(),{})),patch('subprocess.check_output',return_value='test-commit\n'):
   for contents in (relative+'\n'+relative+'\n','../../outside.pyc\n'):
    listing.write_text(contents)
    with self.assertRaises(ValueError):selection(self.root,'3.15')
 def test_base_only_uses_completed_base_families_while_adapters_train(self):
  base=self.root/'families-base';base.mkdir()
  config=configuration(self.root,'no_adapter','3.14')['SHARED_BACKBONES']
  self.assertEqual(config['STATEMENT_MODEL'],str(base/'statement'))
  self.assertEqual(config['SEGMENTATION_MODEL'],str(base/'segmentation'))
  self.assertFalse(config['USE_ADAPTERS'])
  self.assertFalse(config['USE_STATEMENT_ADAPTERS'])
  self.assertFalse(config['USE_SEGMENTATION_ADAPTERS'])
  for task in ('statement','mlm','segmentation'):
   result=self.root/'universal'/task/'result.json';result.parent.mkdir(parents=True)
   result.write_text(json.dumps({'complete':True,'scope':'universal','files':100000,'smoke_only':False,'epoch':2.0}))
  for task in ('statement','segmentation'):
   metadata=base/task/'family.json';metadata.parent.mkdir()
   metadata.write_text(json.dumps({'complete':True,'smoke_only':False,'adapters':{}}))
  require_full_models(self.root,'no_adapter',['3.14'])
  adapted=configuration(self.root,'adapted','3.14')['SHARED_BACKBONES']
  self.assertEqual(adapted['STATEMENT_MODEL'],str(self.root/'families/statement'))
  self.assertEqual(adapted['SEGMENTATION_MODEL'],str(self.root/'families/segmentation'))
if __name__=='__main__':unittest.main()
