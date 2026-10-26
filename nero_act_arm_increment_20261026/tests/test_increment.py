from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import ArmPolicy,encode_delta,decode_delta

torch.set_num_threads(2)


def stats():
    return dict(qpos_mean=[.2,-.1,.3,-.2,.5,-.4,.1,0.],qpos_std=[.3]*7+[1.],
                action_mean=[-.1,.2,-.3,.4,-.2,.1,.5,0.],action_std=[.2]*7+[1.])


def delta_stats():return dict(mean=[.0001]*7,std=[.002]*7)


def inputs():
    obs=torch.randn(2,15)*.1;obs[:,7]=.08;obs[:,8]=0.
    image=torch.rand(2,1,3,64,96)
    return obs,image


def test_delta_all_points_relative_to_same_current_and_no_cumsum():
    q=torch.tensor([[.2]*7]);a=q[:,None]+torch.arange(1,21)[None,:,None]*.0002
    a=a.expand(-1,-1,7);dm=torch.zeros(7);ds=torch.full((7,),.002)
    z=encode_delta(a,q,dm,ds)
    torch.testing.assert_close(decode_delta(z,q,dm,ds),a)
    assert z[0,1,0].item()==pytest.approx(.2,abs=1e-5)


def test_both_initial_readouts_restore_source_and_gripper_exactly():
    a=ArmPolicy(stats(),delta_stats(),variant='A',pretrained=False)
    cp={'model':{k:v.clone() for k,v in a.state_dict().items() if not k.startswith(('arm_readout.','delta_'))}}
    a.load_source(cp)
    b=ArmPolicy(stats(),delta_stats(),variant='B',pretrained=False);b.load_source(cp)
    obs,image=inputs()
    with torch.no_grad():
        original=a.source_prediction(obs,image)
        pa=a(obs,image);pb=b(obs,image)
    physical=lambda p:p[:,:,:7]*a.action_std[:7]+a.action_mean[:7]
    torch.testing.assert_close(physical(pa),physical(original),rtol=0,atol=1e-6)
    torch.testing.assert_close(physical(pb),physical(original),rtol=0,atol=1e-6)
    torch.testing.assert_close(pa[:,:,7],original[:,:,7],rtol=0,atol=0)
    torch.testing.assert_close(pb[:,:,7],original[:,:,7],rtol=0,atol=0)
    assert a.arm_readout.weight.numel()==b.arm_readout.weight.numel()==143*7


def test_only_new_readout_trains_base_stays_eval_z0_and_padding_safe():
    p=ArmPolicy(stats(),delta_stats(),variant='B',pretrained=False)
    cp={'model':{k:v.clone() for k,v in p.state_dict().items() if not k.startswith(('arm_readout.','delta_'))}};p.load_source(cp)
    p.train();obs,image=inputs()
    assert not p.model.training and not p.gripper_head.training
    with torch.no_grad():target=p(obs,image).detach().clone()
    target[:,:2,:7]+=.001
    pad=torch.ones(2,20,dtype=torch.bool);pad[:,:2]=False;target[pad]=float('nan')
    result=p(obs,image,target,pad)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    for name,param in p.named_parameters():
        if name.startswith('arm_readout.'):
            assert param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all()
        else:assert not param.requires_grad and param.grad is None


def test_invalid_scales_and_variants_rejected():
    with pytest.raises(ValueError,match='variant'):ArmPolicy(stats(),delta_stats(),variant='C',pretrained=False)
    with pytest.raises(ValueError,match='scale'):ArmPolicy(stats(),dict(mean=[0.]*7,std=[0.]*7),variant='B',pretrained=False)
