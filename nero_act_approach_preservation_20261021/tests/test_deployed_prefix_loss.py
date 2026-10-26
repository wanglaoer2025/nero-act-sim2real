import pytest
import torch
from tests import test_cvae_training as fixtures


def test_prefix_loss_uses_physical_units_and_only_first_two_valid_points():
    from training.train import deployed_prefix_loss
    p=torch.zeros(1,20,8,requires_grad=True);target=torch.zeros_like(p)
    target[0,0,0]=.001;target[0,0,7]=.0005;target[0,1]=100;target[0,2:]=100
    pad=torch.zeros(1,20,dtype=torch.bool);pad[:,1]=True
    stats={'action_std':[1.]*8}
    loss=deployed_prefix_loss(p,target,pad,stats)
    torch.testing.assert_close(loss,torch.tensor(.25))
    loss.backward()
    assert p.grad[0,0,0]!=0 and p.grad[0,0,7]!=0
    assert torch.count_nonzero(p.grad[:,1:])==0
    assert torch.isfinite(deployed_prefix_loss(p,target,torch.ones_like(pad),stats))


def test_prefix_auxiliary_reaches_actual_zero_latent_arm_and_gripper(monkeypatch):
    from training.train import FullTaskPolicy
    from executor.model import BoundedActPolicy
    fixture=fixtures.CvaeTrainingTests();config=fixture.config(deployed_prefix_loss_weight=.01,prior_gripper_loss_weight=0.)
    torch.set_num_threads(2);policy=FullTaskPolicy(config).train()
    prior=torch.zeros(1,20,8,requires_grad=True);posterior=torch.zeros(1,20,8,requires_grad=True)
    def forward(self,q,image,actions=None,is_pad=None,**kwargs):
        if actions is not None:return dict(a_hat=posterior,kl=posterior.sum()*0)
        return dict(a_hat=prior,unbounded_a_hat=prior) if kwargs.get('return_unbounded') else prior
    monkeypatch.setattr(BoundedActPolicy,'forward',forward)
    q,image,target,pad=fixture.inputs();out=policy(q,image,target,pad)
    assert out['deployed_prefix']>0
    out['deployed_prefix'].backward()
    assert torch.count_nonzero(prior.grad[0,:2,:7])>0 and torch.count_nonzero(prior.grad[0,:2,7])>0
    assert torch.count_nonzero(prior.grad[:,2:])==0
    assert posterior.grad is None


def test_prefix_weight_rejects_nonfinite_or_negative():
    from training.train import FullTaskPolicy
    fixture=fixtures.CvaeTrainingTests()
    for v in [-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError,match='prefix'):
            FullTaskPolicy(fixture.config(deployed_prefix_loss_weight=v))
