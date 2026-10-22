"""Bounded CPU/Gloo topology-change check; not full-model quality/performance."""
import argparse,json,os
from pathlib import Path
import torch
from transformers import RobertaConfig,RobertaForMaskedLM,T5Config,T5ForConditionalGeneration,TrainingArguments,TrainerCallback,set_seed
from dev_scripts.train_universal_models import GuardedTrainer,Telemetry
from dev_scripts.loss_normalization import token_normalized_loss
class StopAtTwo(TrainerCallback):
 def on_step_end(self,args,state,control,**kwargs):
  if state.global_step==2:control.should_save=True;control.should_training_stop=True

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--task',choices=['mlm','statement'],default='mlm');p.add_argument('--global-batch',type=int,default=4)
 a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);world=int(os.environ.get('WORLD_SIZE','1'))
 assert a.global_batch%world==0;set_seed(1337)
 if a.task=='mlm':model=RobertaForMaskedLM(RobertaConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=32,pad_token_id=1,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.))
 else:model=T5ForConditionalGeneration(T5Config(vocab_size=32,d_model=16,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,pad_token_id=1,decoder_start_token_id=0,dropout_rate=0.))
 row={'input_ids':torch.tensor([0,5,6,7,8,9,10,2]),'attention_mask':torch.ones(8,dtype=torch.long),'labels':torch.tensor([0,5,6,7,8,9,10,2])}
 if a.task=='statement':row['decoder_input_ids']=torch.tensor([0,0,5,6,7,8,9,10])
 data=[row for _ in range(2*a.global_batch)]
 args=TrainingArguments(output_dir=str(a.output/'training'),use_cpu=True,per_device_train_batch_size=a.global_batch//world,max_steps=4,save_steps=1,save_total_limit=3,report_to=[],disable_tqdm=True,logging_steps=1,ddp_find_unused_parameters=False,seed=1337)
 args._pylingual_allow_repartition=a.resume
 audit=Telemetry(a.output);trainer=GuardedTrainer(model=model,args=args,train_dataset=data,compute_loss_func=token_normalized_loss,callbacks=[audit]+([] if a.resume else [StopAtTwo()]))
 audit.trainer=trainer;trainer.train(resume_from_checkpoint=str(a.output/'training/checkpoint-2') if a.resume else None)
 assert trainer.state.global_step==(4 if a.resume else 2)
 if args.process_index==0:
  (a.output/('resumed.json' if a.resume else 'initial.json')).write_text(json.dumps({'complete':True,'world_size':args.world_size,'global_step':trainer.state.global_step,'effective_batch':args.world_size*args.per_device_train_batch_size,'losses':[r['loss'] for r in trainer.state.log_history if 'loss' in r]}))
if __name__=='__main__':main()
