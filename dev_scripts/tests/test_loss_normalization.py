import unittest
from types import SimpleNamespace
import torch
from dev_scripts.loss_normalization import token_normalized_loss
class TokenLossNormalizationTests(unittest.TestCase):
 def test_single_rank_matches_native_mean(self):
  generator=torch.Generator().manual_seed(1);logits=torch.randn(4,3,5,generator=generator);labels=torch.tensor([[1,2,-100],[3,-100,-100],[4,1,0],[2,3,4]])
  actual=token_normalized_loss(SimpleNamespace(logits=logits),labels)
  expected=torch.nn.functional.cross_entropy(logits.reshape(-1,5),labels.reshape(-1))
  torch.testing.assert_close(actual,expected)
 def test_two_and_three_rank_gradients_equal_single_rank(self):
  for world in (2,3):
   generator=torch.Generator().manual_seed(2);logits=torch.randn(6,3,5,generator=generator,requires_grad=True)
   labels=torch.tensor([[1,2,-100],[3,-100,-100],[4,1,0],[2,3,4],[1,-100,-100],[4,2,1]])
   expected=token_normalized_loss(SimpleNamespace(logits=logits),labels);expected.backward();pieces=[];losses=[]
   for x,y in zip(logits.detach().chunk(world),labels.chunk(world)):
    x=x.clone().requires_grad_();loss=token_normalized_loss(SimpleNamespace(logits=x),y,(labels!=-100).sum())*world;loss.backward();pieces.append(x.grad/world);losses.append(loss.detach())
   with self.subTest(world=world):
    torch.testing.assert_close(torch.cat(pieces),logits.grad);torch.testing.assert_close(torch.stack(losses).mean(),expected.detach())
 def test_no_supervision_is_rejected(self):
  with self.assertRaisesRegex(FloatingPointError,'No supervised'):token_normalized_loss(SimpleNamespace(logits=torch.zeros(1,2,3)),torch.full((1,2),-100))
if __name__=='__main__':unittest.main()
