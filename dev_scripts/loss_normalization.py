"""Global supervised-token loss, independent of GPU count and rank padding."""
import torch

def token_normalized_loss(outputs,labels,num_items_in_batch=None):
 logits=outputs.logits
 labels=labels.to(logits.device)
 total=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100,reduction='sum')
 denominator=(labels!=-100).sum() if num_items_in_batch is None else num_items_in_batch
 denominator=torch.as_tensor(denominator,device=logits.device,dtype=torch.float32)
 if denominator.item()<=0:raise FloatingPointError('No supervised tokens in loss denominator')
 return total/denominator
