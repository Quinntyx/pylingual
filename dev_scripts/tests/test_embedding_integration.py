"""Bounded real-model checks: atomic alignment, sparse input rows and package reload."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration, RobertaConfig, RobertaForTokenClassification
from pylingual.lora import AdapterBank, adapter_key
from pylingual.opcodes import OpcodeRegistry, RESERVED_TOKENS, known_token, tokenizer_fingerprint, instruction_token_indices
from pylingual.opcode_segmenter import OpcodeSegmenter
from pylingual.shared_models import load_family, load_version


def tokenizer_fixture():
    vocabulary = {word: i for i, word in enumerate(['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]', 'return', 'x', ',', '1', '2', '3.14', '3.15'])}
    backend = Tokenizer(models.WordPiece(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    names = ['LOAD_CONST', 'LOAD_FAST', 'RETURN_VALUE']
    backend.add_tokens([AddedToken(t, normalized=False, special=False) for t in [known_token(n) for n in names] + list(RESERVED_TOKENS)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')
    data = {'format': 1, 'preprocessing': 'opcode-field-placeholders-v1',
            'known_opcodes': {n: {'token': known_token(n), 'id': tokenizer.convert_tokens_to_ids(known_token(n))} for n in names},
            'reserved_slots': [{'slot': i, 'token': t, 'id': tokenizer.convert_tokens_to_ids(t)} for i, t in enumerate(RESERVED_TOKENS)]}
    data['source_generation_forbidden_ids'] = [item['id'] for item in list(data['known_opcodes'].values()) + data['reserved_slots']]
    return tokenizer, OpcodeRegistry(data, tokenizer)


def model_fixture(task, vocabulary):
    if task == 'statement':
        return T5ForConditionalGeneration(T5Config(vocab_size=vocabulary, d_model=24, d_kv=12, d_ff=48, num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0, pad_token_id=1, decoder_start_token_id=1, eos_token_id=3, tie_word_embeddings=True))
    return RobertaForTokenClassification(RobertaConfig(vocab_size=vocabulary, hidden_size=24, intermediate_size=48, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=130, pad_token_id=1, num_labels=3, hidden_dropout_prob=0, attention_probs_dropout_prob=0, id2label={0:'B',1:'I',2:'E'},label2id={'B':0,'I':1,'E':2}))


class EmbeddingIntegrationTests(unittest.TestCase):
    def bank(self, task):
        tokenizer, registry = tokenizer_fixture()
        model = model_fixture(task, len(tokenizer))
        targets = ('q','v') if task == 'statement' else ('query','value')
        originals = [(p,p.detach().clone()) for p in model.parameters()]
        bank = AdapterBank(model, 'test-backbone', targets, opcode_registry=registry, tokenizer=tokenizer, task=task)
        bank.add('3.14', rank=2, bindings={})
        bank.add('3.15', rank=2, bindings={'NEW_OP':0})
        bank.model.eval()
        return tokenizer, registry, bank, originals

    def test_input_rows_training_and_frozen_tied_head(self):
        for task in ('statement','segmentation'):
            with self.subTest(task=task):
                tokenizer,registry,bank,originals=self.bank(task)
                inputs=tokenizer(bank.bind_bytecode('NEW_OP , x','3.15'),return_tensors='pt')
                inputs.pop('token_type_ids',None)
                inputs['labels']=torch.tensor([[5,6,3]]) if task=='statement' else torch.zeros_like(inputs['input_ids'])
                parameters=bank.trainable('3.15')
                row=bank.input_updates.rows[adapter_key('3.15')].delta
                self.assertTrue(row.requires_grad)
                optimizer=torch.optim.AdamW(parameters,lr=.01)
                loss=bank.model(**inputs).loss; loss.backward()
                self.assertTrue(torch.isfinite(loss)); self.assertGreater(row.grad.abs().sum().item(),0)
                optimizer.step()
                self.assertGreater(row.abs().sum().item(),0)
                for parameter,before in originals:self.assertTrue(torch.equal(parameter,before))
                bank.select('3.14')
                self.assertEqual(bank.input_updates.active,adapter_key('3.14'))
                if task=='statement':
                    # Shared decoder embeddings and tied output projection are untouched.
                    self.assertIs(bank.model.lm_head.weight,bank.model.shared.weight)
                    with bank.use('3.15'):
                        ids=torch.tensor([[registry.slots[0]['id']]])
                        self.assertTrue(torch.equal(bank.model.shared(ids),torch.nn.functional.embedding(ids,bank.model.lm_head.weight)))
                        generated=bank.model.generate(inputs['input_ids'],attention_mask=inputs['attention_mask'],max_new_tokens=2,bad_words_ids=[[i] for i in registry.forbidden_ids])
                        self.assertFalse(set(generated[0].tolist()) & set(registry.forbidden_ids))

    def test_isolated_reused_slots_and_context_restore(self):
        tokenizer,registry,bank,_=self.bank('segmentation')
        bank.add('3.16',rank=2,bindings={'OTHER_OP':0})
        bank.input_updates.rows[adapter_key('3.15')].delta.data.fill_(.1)
        bank.input_updates.rows[adapter_key('3.16')].delta.data.fill_(.2)
        bank.select('3.14')
        with self.assertRaises(RuntimeError):
            with bank.use('3.15'):raise RuntimeError('test')
        self.assertEqual(bank.active_version,'3.14')
        self.assertTrue(torch.all(bank.input_updates.rows[adapter_key('3.16')].delta==.2))

    def test_atomic_instruction_alignment_and_batched_padding(self):
        tokenizer,registry,bank,_=self.bank('segmentation')
        texts=['LOAD_CONST , x <SEP> NEW_OP , 1 <SEP> RETURN_VALUE','LOAD_FAST , x']
        pipeline=OpcodeSegmenter(bank,tokenizer,'3.15')
        results=pipeline(texts,batch_size=2)
        self.assertEqual([len(row) for row in results],[3,1])
        self.assertEqual([item['word'] for item in results[0]],texts[0].split(' <SEP> '))
        encoded,indices=instruction_token_indices(tokenizer,bank.bind_bytecode(texts[0],'3.15'))
        self.assertEqual(set(i for i in indices if i is not None),{0,1,2})
        self.assertEqual(len(encoded['input_ids']),len(indices))
        baseline=pipeline.for_version(None,python_version='3.15')
        self.assertEqual(len(baseline([texts[0]])[0]),3)

    def test_adapter_roundtrip_and_tamper_rejection(self):
        for task in ('statement','segmentation'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                tokenizer,registry,bank,_=self.bank(task)
                bank.input_updates.rows[adapter_key('3.15')].delta.data.fill_(.15)
                path=Path(tmp); bank.save('3.15',path)
                meta=json.loads((path/'adapter.json').read_text())
                self.assertEqual(meta['format'],2)
                model=model_fixture(task,len(tokenizer))
                other=AdapterBank(model,'test-backbone',bank.targets,opcode_registry=registry,tokenizer=tokenizer,task=task)
                other.load('3.15',path)
                self.assertTrue(torch.equal(other.input_updates.rows[adapter_key('3.15')].delta,bank.input_updates.rows[adapter_key('3.15')].delta))
                meta['tokenizer_sha256']='wrong';(path/'adapter.json').write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError,'tokenizer'):other.load('3.15',path,replace=True)
                self.assertEqual(len(other.specs),1)

    def test_complete_family_runtime_reload(self):
        for task in ('statement','segmentation'):
            with self.subTest(task=task),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp);tokenizer,registry=tokenizer_fixture()
                model=model_fixture(task,len(tokenizer));model.save_pretrained(path/'backbone');tokenizer.save_pretrained(path/'tokenizer')
                (path/'tokenizer/opcode_registry.json').write_text(json.dumps(registry.data))
                targets=('q','v') if task=='statement' else ('query','value')
                bank=AdapterBank(model,'test-backbone',targets,opcode_registry=registry,tokenizer=tokenizer,task=task)
                bank.add('3.15',rank=2,bindings={'NEW_OP':0});bank.save('3.15',path/'adapters/3.15')
                metadata={'format':2,'complete':True,'smoke_only':True,'task':task,'backbone_id':'test-backbone','targets':targets,'versions':['3.15'],'adapters':{'3.15':'adapters/3.15'},'opcode_registry_sha256':registry.fingerprint,'tokenizer_sha256':tokenizer_fingerprint(tokenizer),'opcode_bindings':{'3.15':{'NEW_OP':0}},'source_prefix':''}
                (path/'family.json').write_text(json.dumps(metadata))
                loaded=load_version(path,task,'3.15',torch.device('cpu'))
                if task=='segmentation':self.assertEqual(len(loaded(['NEW_OP , x <SEP> RETURN_VALUE'])[0]),2)
                else:
                    from pylingual.models import VersionedTranslator
                    loaded=VersionedTranslator(loaded[0],loaded[1],torch.device('cpu'),'3.15','')
                    self.assertEqual(loaded._prepare_bytecode('NEW_OP , x'),'<future_opcode_00> , x')
                    self.assertEqual(set(i[0] for i in loaded.generation_kwargs['bad_words_ids']),set(registry.forbidden_ids))

    def test_backbone_publication_and_checksum_rejection(self):
        from dev_scripts.publish_shared_backbone import publish
        for task in ('statement', 'segmentation'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);tokenizer,registry=tokenizer_fixture()
                checkpoint=root/'checkpoint';tokens=root/'tokens';output=root/'family'
                model_fixture(task,len(tokenizer)).save_pretrained(checkpoint)
                tokenizer.save_pretrained(tokens)
                (tokens/'opcode_registry.json').write_text(json.dumps(registry.data))
                with self.assertRaisesRegex(ValueError,'held-out'):
                    publish(checkpoint,tokens,task,output,training_versions=['3.14','3.15'])
                publish(checkpoint,tokens,task,output,training_versions=['3.14'])
                metadata=json.loads((output/'family.json').read_text())
                self.assertTrue(metadata['complete'])
                self.assertFalse(metadata['smoke_only'])
                self.assertTrue(metadata['backbone_files_sha256'])
                self.assertEqual(load_family(output,task,torch.device('cpu'))[0].backbone_id,metadata['backbone_id'])
                # A fresh family path avoids the process-local immutable-model cache.
                import shutil
                damaged=root/'damaged';shutil.copytree(output,damaged)
                with (damaged/'backbone/config.json').open('a') as stream:stream.write(' ')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    load_family(damaged,task,torch.device('cpu'))
                forged=root/'forged';shutil.copytree(output,forged)
                altered=dict(metadata,backbone_id='wrong')
                (forged/'family.json').write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,'identity'):
                    load_family(forged,task,torch.device('cpu'))

    def test_translation_preserves_special_flagged_source_tokens(self):
        from unittest.mock import patch
        from pylingual.models import VersionedTranslator
        tokenizer,registry=tokenizer_fixture()
        tokenizer.add_special_tokens({'additional_special_tokens':['return','<mask_0>',',']})
        # Real trained BPE artifacts flag ordinary source tokens as special.
        registry=OpcodeRegistry(registry.data,tokenizer)
        model=model_fixture('statement',len(tokenizer))
        bank=AdapterBank(model,'test-backbone',('q','v'),opcode_registry=registry,tokenizer=tokenizer,task='statement')
        return_id=tokenizer.convert_tokens_to_ids('return')
        mask_id=tokenizer.convert_tokens_to_ids('<mask_0>')
        emitted=torch.tensor([[tokenizer.pad_token_id,tokenizer.cls_token_id,return_id,mask_id,registry.forbidden_ids[0],tokenizer.sep_token_id]])
        for use_adapter in (False,True):
            with self.subTest(use_adapter=use_adapter):
                translator=VersionedTranslator(bank,tokenizer,torch.device('cpu'),'3.14','',use_adapter=use_adapter)
                self.assertIn(return_id,tokenizer.all_special_ids)
                with patch.object(model,'generate',return_value=emitted):
                    translated=translator._translate_and_decode(['LOAD_FAST , x'])
                self.assertEqual(translated,['return <mask_0>'])
                for word in ('return','<mask_0>',','):
                    self.assertNotIn(tokenizer.convert_tokens_to_ids(word),translator.decode_excluded_ids)
                self.assertTrue(set(registry.forbidden_ids)<=translator.decode_excluded_ids)

    def test_normal_statement_limits_and_backbone_digest(self):
        from dev_scripts.opcode_adapter_training import encode_rows, frozen_digest
        tokenizer,registry=tokenizer_fixture()
        rows=[{'version':'3.14','file':'fixture','bytecode':'LOAD_CONST , '+('x '*600),'source':'return '*600}]
        encoded=encode_rows('statement',tokenizer,registry,rows,{})[0]
        self.assertEqual(len(encoded['input_ids']),256)
        self.assertEqual(len(encoded['labels']),256)
        _,_,bank,_=self.bank('statement')
        before=frozen_digest(bank.model)
        # The audit must not hide a base parameter accidentally marked trainable.
        base=bank.model.get_input_embeddings().weight
        base.requires_grad_(True)
        self.assertEqual(frozen_digest(bank.model),before)
        with torch.no_grad():base[0,0].add_(1)
        self.assertNotEqual(frozen_digest(bank.model),before)


if __name__=='__main__':unittest.main()
