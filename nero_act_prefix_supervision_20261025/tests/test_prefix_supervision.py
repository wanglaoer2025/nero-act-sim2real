from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import PrefixPolicy, prefix_arm_loss

torch.set_num_threads(2)


def test_prefix_gets_half_the_mass_and_gradient():
    p=torch.zeros(1,20,7,requires_grad=True)
    target=torch.zeros_like(p);target[:,:2]=2.
    loss=prefix_arm_loss(p,target,torch.zeros(1,20,dtype=torch.bool))
    assert loss.item()==pytest.approx(1.)
    loss.backward()
    assert p.grad[:,:2].abs().sum().item()==pytest.approx(.5)
    assert p.grad[:,2:].abs().sum().item()==0


def test_masked_nan_and_suffixless_rows_are_finite():
    p=torch.zeros(2,20,7,requires_grad=True)
    target=torch.ones_like(p)
    pad=torch.ones(2,20,dtype=torch.bool);pad[:,:2]=False
    target[pad]=float('nan')
    loss=prefix_arm_loss(p,target,pad)
    assert loss.item()==pytest.approx(1.)
    loss.backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad[pad].abs().sum()==0


def test_partial_padding_uses_only_valid_prefix_and_suffix():
    p=torch.zeros(2,20,7)
    target=torch.ones_like(p);target[:,:2]=3.
    pad=torch.zeros(2,20,dtype=torch.bool);pad[0,1:]=True;pad[1,4:]=True
    target[pad]=float('nan')
    assert prefix_arm_loss(p,target,pad).item()==pytest.approx(2.)


def stats():
    return dict(qpos_mean=[0.]*8,qpos_std=[1.]*8,action_mean=[0.]*8,action_std=[1.]*8)


def inputs():
    obs=torch.zeros(2,15);obs[:,7]=.08
    rgb=torch.rand(2,1,3,64,96)
    labels=torch.zeros(2,20,8);labels[:,:,7]=.08
    return obs,rgb,labels,torch.zeros(2,20,dtype=torch.bool)


def test_same_initial_inference_and_only_arm_objective_changes():
    a=PrefixPolicy(stats(),variant='A',pretrained=False)
    b=PrefixPolicy(stats(),variant='B',pretrained=False)
    b.load_state_dict(a.state_dict())
    assert [(n,p.requires_grad) for n,p in a.named_parameters()]==[(n,p.requires_grad) for n,p in b.named_parameters()]
    obs,rgb,labels,pad=inputs()
    a.eval();b.eval()
    with torch.no_grad():
        torch.testing.assert_close(a(obs,rgb),b(obs,rgb),rtol=0,atol=0)
    a.train();b.train()
    torch.manual_seed(731);ra=a(obs,rgb,labels,pad)
    torch.manual_seed(731);rb=b(obs,rgb,labels,pad)
    for key in ['prediction','gripper','kl','mu','logvar']:
        torch.testing.assert_close(ra[key],rb[key],rtol=0,atol=0)
    expected=(ra['prediction'][:,:,:7]-labels[:,:,:7]).abs().mean()
    torch.testing.assert_close(ra['arm'],expected)
    torch.testing.assert_close(rb['loss']-rb['arm'],ra['loss']-ra['arm'])
    rb['loss'].backward()
    assert b.model.action_head.weight.grad.abs().sum()>0


def test_invalid_variant_is_rejected():
    with pytest.raises(ValueError,match='variant'):
        PrefixPolicy(stats(),variant='anything',pretrained=False)
