"""Validate checkpoint repartitioning without changing effective batch/optimizer."""
from pathlib import Path

OPTIMIZER_KEYS=('learning_rate','num_train_epochs','fp16','bf16','weight_decay','optim',
 'adam_beta1','adam_beta2','adam_epsilon','max_grad_norm','lr_scheduler_type',
 'warmup_steps','warmup_ratio','seed','data_seed')

def checkpoint_world_size(checkpoint):
 checkpoint=Path(checkpoint)
 states=list(checkpoint.glob('rng_state_*.pth'))
 if states:return len(states)
 return 1 if (checkpoint/'rng_state.pth').exists() else None

def validate_resume_layout(previous,current,current_world_size,original_world_size=None):
 for key in ('task','initialization','initial_weights_sha256','data_manifest','tokenizer_registry_fingerprint'):
  if previous[key]!=current[key]:raise ValueError('Resume identity mismatch: '+key)
 old,new=previous['training_arguments'],current['training_arguments']
 for key in OPTIMIZER_KEYS:
  if old.get(key)!=new.get(key):raise ValueError('Resume optimizer mismatch: '+key)
 recorded=previous.get('world_size')
 if recorded is not None and original_world_size is not None and recorded!=original_world_size:
  raise ValueError('Original world size contradicts recorded provenance')
 old_world=recorded if recorded is not None else original_world_size
 changed=any(old[k]!=new[k] for k in ('per_device_train_batch_size','gradient_accumulation_steps'))
 if changed and original_world_size is None:raise ValueError('Batch repartition requires explicit original world size')
 if old_world is None:old_world=current_world_size
 if old_world!=current_world_size and original_world_size is None:raise ValueError('World-size change requires explicit repartition authorization')
 if type(old_world) is not int or old_world<1 or type(current_world_size) is not int or current_world_size<1:
  raise ValueError('Positive integer world sizes required')
 if previous.get('loss_normalization')!=current.get('loss_normalization'):
  if previous.get('loss_normalization') is not None or old_world!=1:
   raise ValueError('Legacy multi-GPU checkpoint has incompatible loss normalization')
 elif previous.get('loss_normalization') is not None and previous.get('loss_normalization_sha256')!=current.get('loss_normalization_sha256'):
  raise ValueError('Loss-normalization implementation changed during resume')
 old_effective=old['per_device_train_batch_size']*old['gradient_accumulation_steps']*old_world
 new_effective=new['per_device_train_batch_size']*new['gradient_accumulation_steps']*current_world_size
 if old_effective!=new_effective:raise ValueError('Repartition changes effective batch size')
 return {'original_world_size':old_world,'world_size':current_world_size,
  'original_per_device_batch':old['per_device_train_batch_size'],
  'per_device_batch':new['per_device_train_batch_size'],'effective_batch_size':new_effective,
  'changed_from_original_layout':changed or old_world!=current_world_size}
