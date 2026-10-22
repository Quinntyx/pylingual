"""Real CUDA checks for full-coverage sampling, checkpoint resume, MLM and LoRA."""
import copy
from collections import Counter
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForMaskedLM, T5Config, T5ForConditionalGeneration
from dev_scripts import full_ablation_training as training
from pylingual.lora import AdapterBank


class FullTrainingChecks(unittest.TestCase):
    def test_full_coverage_and_sampler_resume(self):
        groups = {v:[{'key':(v,i)} for i in range(5)] for v in ('3.6','3.14','3.15')}
        sampler = training.BalancedSampler(groups,17)
        first = sampler.take(15)
        self.assertEqual(len({r['key'] for r in first}),15)
        self.assertEqual(Counter(r['key'][0] for r in first),{'3.6':5,'3.14':5,'3.15':5})
        state = copy.deepcopy(sampler.state())
        expected = sampler.take(19)
        resumed = training.BalancedSampler(groups,999)
        resumed.restore(state)
        self.assertEqual(resumed.take(19),expected)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA runtime required')
    def test_cuda_training_checkpoint_resume_and_frozen_lora(self):
        torch.set_num_threads(2)
        vocab = {'[UNK]':0,'[PAD]':1,'[CLS]':2,'[SEP]':3,'[MASK]':4,'a':5,'b':6}
        backend = Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',pad_token='[PAD]',cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]')
        args = SimpleNamespace(batch_size=2,accumulation=1,eval_every=1)
        row = {'input_ids':[2,5,6,3],'attention_mask':[1,1,1,1],'labels':[2,6,3]}
        groups = {'3.14':[copy.deepcopy(row) for _ in range(4)]}
        config = T5Config(vocab_size=7,d_model=32,d_ff=64,num_layers=1,num_decoder_layers=1,num_heads=4,dropout_rate=0.1,pad_token_id=1,eos_token_id=3,decoder_start_token_id=1,use_cache=False)
        torch.manual_seed(11)
        reference = T5ForConditionalGeneration(config).cuda()
        initial = copy.deepcopy(reference.state_dict())
        root = Path(tempfile.mkdtemp(prefix='pylingual-full-training-check-'))
        expected = training.train_stage(reference,tokenizer,groups,groups,'statement',root/'reference',2,1e-3,args,41)
        interrupted = T5ForConditionalGeneration(config).cuda()
        interrupted.load_state_dict(initial)
        real_validation = training.validation_loss
        calls = [0]
        def fail_second_validation(*positional,**keyword):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError('intentional simulated training interruption')
            return real_validation(*positional,**keyword)
        with patch.object(training,'validation_loss',side_effect=fail_second_validation):
            with self.assertRaisesRegex(RuntimeError,'intentional simulated'):
                training.train_stage(interrupted,tokenizer,groups,groups,'statement',root/'resume',2,1e-3,args,41)
        restored = training.train_stage(interrupted,tokenizer,groups,groups,'statement',root/'resume',2,1e-3,args,41)
        self.assertEqual(restored['selected_step'],expected['selected_step'])
        for key,value in reference.state_dict().items():
            self.assertTrue(torch.allclose(value,interrupted.state_dict()[key],atol=1e-6,rtol=1e-6),key)
        base = {id(p):p.detach().clone() for p in interrupted.parameters()}
        bank = AdapterBank(interrupted,'test-family',('q','v'))
        bank.add('3.14',rank=2,alpha=4,dropout=0.05)
        parameters = bank.trainable('3.14')
        self.assertTrue(parameters)
        training.train_stage(interrupted,tokenizer,groups,groups,'statement',root/'adapter',2,1e-3,args,51)
        for parameter in interrupted.parameters():
            if id(parameter) in base:
                self.assertTrue(torch.equal(parameter,base[id(parameter)]))
        mlm_config = RobertaConfig(vocab_size=7,hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64,max_position_embeddings=32,pad_token_id=1,bos_token_id=2,eos_token_id=3)
        mlm = RobertaForMaskedLM(mlm_config).cuda()
        mlm_row = {**row,'labels':[0,1,2,0]}
        mlm_groups = {'3.14':[copy.deepcopy(mlm_row) for _ in range(4)]}
        result = training.train_stage(mlm,tokenizer,mlm_groups,mlm_groups,'mlm',root/'mlm',2,1e-3,args,61)
        self.assertEqual(result['steps'],2)


if __name__ == '__main__':
    unittest.main()
