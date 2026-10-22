import copy,tempfile,unittest
from pathlib import Path
from dev_scripts.resume_layout import checkpoint_world_size,validate_resume_layout

class ResumeLayoutTests(unittest.TestCase):
 def provenance(self,batch,world=None):
  value={'task':'statement','initialization':'generic','initial_weights_sha256':'same','data_manifest':{'same':True},'tokenizer_registry_fingerprint':'same',
   'training_arguments':{'per_device_train_batch_size':batch,'gradient_accumulation_steps':1,'learning_rate':2e-5,'num_train_epochs':2,'fp16':True,'bf16':False,'seed':1337}}
  if world is not None:value['world_size']=world
  return value
 def test_statement_three_to_two_keeps_72(self):
  self.assertEqual(validate_resume_layout(self.provenance(24),self.provenance(36,2),2,3)['effective_batch_size'],72)
 def test_mlm_one_to_two_keeps_28(self):
  self.assertEqual(validate_resume_layout(self.provenance(28),self.provenance(14,2),2,1)['effective_batch_size'],28)
 def test_rejects_changed_effective_batch(self):
  with self.assertRaisesRegex(ValueError,'effective batch'):validate_resume_layout(self.provenance(24),self.provenance(24,2),2,3)
 def test_rejects_implicit_batch_change(self):
  with self.assertRaisesRegex(ValueError,'explicit original'):validate_resume_layout(self.provenance(24),self.provenance(36,2),2)
 def test_rejects_known_world_change_without_permission(self):
  with self.assertRaises(ValueError):validate_resume_layout(self.provenance(24,3),self.provenance(24,2),2)
 def test_rejects_wrong_original_world(self):
  with self.assertRaisesRegex(ValueError,'contradicts'):validate_resume_layout(self.provenance(24,3),self.provenance(36,2),2,1)
 def test_rejects_optimizer_or_identity_drift(self):
  for key in ('learning_rate','seed','num_train_epochs'):
   current=self.provenance(36,2);current['training_arguments'][key]=999
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'optimizer mismatch'):validate_resume_layout(self.provenance(24),current,2,3)
  current=self.provenance(36,2);current['initial_weights_sha256']='different'
  with self.assertRaisesRegex(ValueError,'identity mismatch'):validate_resume_layout(self.provenance(24),current,2,3)
 def test_subsequent_same_layout_resume_is_unchanged(self):
  result=validate_resume_layout(self.provenance(14,2),self.provenance(14,2),2)
  self.assertFalse(result['changed_from_original_layout'])
 def test_old_single_rank_mlm_can_resume_but_old_distributed_loss_cannot(self):
  current=self.provenance(14,2);current['loss_normalization']='global-supervised-token-mean-v1'
  validate_resume_layout(self.provenance(28),current,2,1)
  current=self.provenance(36,2);current['loss_normalization']='global-supervised-token-mean-v1'
  with self.assertRaisesRegex(ValueError,'incompatible loss'):validate_resume_layout(self.provenance(24),current,2,3)
 def test_checkpoint_world_size(self):
  with tempfile.TemporaryDirectory() as temporary:
   path=Path(temporary);self.assertIsNone(checkpoint_world_size(path))
   (path/'rng_state.pth').touch();self.assertEqual(checkpoint_world_size(path),1)
   for rank in range(2):(path/f'rng_state_{rank}.pth').touch()
   self.assertEqual(checkpoint_world_size(path),2)
if __name__=='__main__':unittest.main()
