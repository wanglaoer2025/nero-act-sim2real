import numpy as np
import torch
from tests import test_cvae_training as fixture


def test_correct_deployed_prediction_has_zero_prior_gripper_auxiliary_loss():
    from training.train import FullTaskPolicy
    c=fixture.CvaeTrainingTests().config(prior_gripper_loss_weight=.005,prior_gripper_target_profile="feedback_preimage_v1")
    torch.manual_seed(42);p=FullTaskPolicy(c).eval()
    q,image,_,padding=fixture.CvaeTrainingTests().inputs()
    with torch.no_grad():
        target=p(q,image).clone()
        output=p(q,image,target,padding)
    assert float(output['l1'])<1e-7
    assert float(output['prior_gripper'])<1e-5, 'Auxiliary target must share the deployed nonlinear readout'


def test_inverse_target_roundtrips_bounded_prefix_and_tail():
    from executor.model import feedback_preimage,feedback_readout
    q=torch.tensor([[0.]*7+[.075]])
    stats={k:torch.zeros(8) if k.endswith('mean') else torch.ones(8) for k in ['qpos_mean','qpos_std','action_mean','action_std']}
    lower=torch.tensor([-3.]*7+[0.]);upper=torch.tensor([3.]*7+[.095])
    target=q[:,None].repeat(1,20,1);target[:,:10,7]=.067;target[:,10:,7]=.063
    inverse=feedback_preimage(target,q,stats,lower,upper,version='feedback_tanh_v2')
    actual=feedback_readout(inverse,q,stats,lower,upper,version='feedback_tanh_v2')
    torch.testing.assert_close(actual,target,atol=2e-8,rtol=0)
    assert inverse[0,0,7]<target[0,0,7]


def test_unreachable_inverse_remains_finite_and_preserves_limits():
    from executor.model import feedback_preimage,feedback_readout
    q=torch.tensor([[0.]*7+[.095]])
    stats={k:torch.zeros(8) if k.endswith('mean') else torch.ones(8) for k in ['qpos_mean','qpos_std','action_mean','action_std']}
    lower=torch.tensor([-3.]*7+[0.]);upper=torch.tensor([3.]*7+[.095])
    target=q[:,None].repeat(1,20,1);target[:,:,7]=0
    inverse=feedback_preimage(target,q,stats,lower,upper,version='feedback_tanh_v2')
    assert torch.isfinite(inverse).all()
    actual=feedback_readout(inverse,q,stats,lower,upper,version='feedback_tanh_v2')
    assert torch.all(actual[:,:10,7]>=q[:,None,7]-.00950001)
