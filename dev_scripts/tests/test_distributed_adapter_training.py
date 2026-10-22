import os,unittest
from types import SimpleNamespace
from unittest.mock import patch
from dev_scripts.opcode_adapter_training import train_opcode_adapter

class DistributedAdapterGuardTests(unittest.TestCase):
 def test_distributed_requires_initialized_group_before_rank_query(self):
  with patch.dict(os.environ,{'WORLD_SIZE':'4','LOCAL_RANK':'2'}),patch('torch.distributed.is_initialized',return_value=False),patch('torch.distributed.get_rank') as rank:
   with self.assertRaisesRegex(RuntimeError,'initialized process group'):train_opcode_adapter(SimpleNamespace(),{})
   rank.assert_not_called()
 def test_single_gpu_keeps_explicit_full_run_budget_requirement(self):
  args=SimpleNamespace(steps=None,validation_data=None,epochs=None)
  with patch.dict(os.environ,{'WORLD_SIZE':'1'}):
   with self.assertRaisesRegex(ValueError,'Full training requires explicit'):train_opcode_adapter(args,{})
if __name__=='__main__':unittest.main()
