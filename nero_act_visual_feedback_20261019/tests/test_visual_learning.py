import json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
class VisualLearningTests(unittest.TestCase):
    def config(self,freeze):
        c=json.loads((ROOT/'model/run15/training_config.json').read_text())
        return dict(c,device='cpu',freeze_backbone=freeze,pretrained_backbone=False)
    def test_explicit_visual_training_updates_backbone_and_preserves_default_freeze(self):
        from training.train import FullTaskPolicy
        torch.set_num_threads(2);torch.manual_seed(42)
        policy=FullTaskPolicy(self.config(False));policy.train()
        backbone=[p for n,p in policy.named_parameters() if 'backbones' in n]
        self.assertTrue(any(p.requires_grad for p in backbone),'freeze_backbone=False must enable visual learning')
        q=torch.zeros(2,9);image=torch.rand(2,1,3,90,160)
        with torch.no_grad():target=policy(q,image)+.01
        before=backbone[0].detach().clone()
        loss=policy(q,image,target,torch.zeros(2,20,dtype=torch.bool))['loss']
        policy.optimizer.zero_grad(set_to_none=True);loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in backbone))
        policy.optimizer.step();self.assertFalse(torch.equal(before,backbone[0]))
        frozen=FullTaskPolicy(self.config(True))
        self.assertTrue(all(not p.requires_grad for n,p in frozen.named_parameters() if 'backbones' in n))
