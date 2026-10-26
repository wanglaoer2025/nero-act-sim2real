import pytest
import torch
from torch import nn

STATS={'qpos_mean':[0.]*8,'qpos_std':[1.]*8,'action_mean':[0.]*8,'action_std':[1.]*8}

def observation(width=.074,previous=.070,n=1):
    q=torch.zeros(n,15);q[:,7]=width;q[:,8]=(previous-width)/.01;return q

class TinyAct(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.action_head=nn.Linear(128,8);self.config={};self.device=torch.device('cpu')
    def forward(self,q,image):
        h=torch.ones(len(q),20,128,device=q.device)*image.flatten()[0]
        return self.model.action_head(h)


def test_zero_head_holds_causal_previous_command_and_preserves_arm():
    from model import IncrementHead,FrozenIncrementPolicy
    head=IncrementHead(128,STATS);h=torch.randn(2,20,128)
    q=observation(n=2);q[1,8]=-.2
    result=head(h,q)
    torch.testing.assert_close(result[0],torch.full((20,),.070),atol=1e-8,rtol=0)
    torch.testing.assert_close(result[1],torch.full((20,),.072),atol=1e-8,rtol=0)
    base=TinyAct();policy=FrozenIncrementPolicy(base,head,STATS,{})
    image=torch.ones(2,1,3,2,2);original=base(q,image)
    output=policy(q,image)
    assert torch.equal(output[...,:7],original[...,:7])
    assert all(not p.requires_grad for p in base.parameters())
    assert all(p.requires_grad for p in head.parameters())
    assert policy._hidden is None


def test_readout_limits_feedback_and_physical_range_and_rejects_bad_input():
    from model import IncrementHead
    head=IncrementHead(128,STATS);h=torch.ones(1,20,128)
    with torch.no_grad():head.network[2].bias.fill_(-100)
    result=head(h,observation(width=.074,previous=.065))
    torch.testing.assert_close(result,torch.full((1,20),.0645),atol=1e-8,rtol=0)
    with torch.no_grad():head.network[2].bias.fill_(100)
    result=head(h,observation(width=.0948,previous=.095))
    assert float(result.max())<=.09500001
    for q in [torch.zeros(1,14),observation(previous=.100),observation(width=.2,previous=.070),observation()*float('nan')]:
        with pytest.raises(ValueError):head(h,q)


def test_training_targets_are_causal_increments_and_padding_is_ignored():
    from model import IncrementHead,increment_loss
    head=IncrementHead(128,STATS);h=torch.ones(1,2,128);q=observation()
    target=torch.tensor([[.0698,float('nan')]]);padding=torch.tensor([[False,True]])
    loss=increment_loss(head,h,q,target,padding)
    assert float(loss)==pytest.approx(.04,abs=2e-6)
    loss.backward();assert float(head.network[2].bias.grad)>0
    with pytest.raises(ValueError):increment_loss(head,h,q,torch.tensor([[.068,.070]]),torch.zeros(1,2,dtype=torch.bool))


def test_policy_cleans_capture_after_base_failure():
    from model import IncrementHead,FrozenIncrementPolicy
    class FailingAct(TinyAct):
        def forward(self,q,image):
            super().forward(q,image);raise RuntimeError('base failed')
    policy=FrozenIncrementPolicy(FailingAct(),IncrementHead(128,STATS),STATS,{})
    with pytest.raises(RuntimeError):policy(observation(),torch.ones(1))
    assert policy._hidden is None and not policy._encoding
