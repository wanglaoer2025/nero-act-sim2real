import numpy as np
import pytest
import torch


def test_offline_groups_distinguish_closing_holding_opening_and_release():
    from training.grouped_gripper import classify_gripper_groups
    stages=np.array(['close','close','lift','carry','place','approach','descend','retreat','settle','release','history_context'])
    margins=np.array([.0029,.003,.004,.004,.004,0,0,0,0,.004,0])
    assert classify_gripper_groups(stages,margins).tolist()==[0,1,1,1,1,2,2,2,2,3,-1]
    for stage,gap in [(np.array(['unknown']),np.array([0.])),(stages,np.full(len(stages),np.nan)),(stages,margins[:-1])]:
        with pytest.raises(ValueError):classify_gripper_groups(stage,gap)


@pytest.mark.parametrize('profile',['legacy_physical_target','feedback_preimage_v1'])
def test_gripper_contributions_reproduce_actual_restricted_prefix_gradient(profile):
    from training.grouped_gripper import gripper_contributions
    from tests.test_restricted_history_heads import config
    from training.history import build_history_policy
    from executor.model import BoundedActPolicy
    c,q,image,_=config();c.update(gripper_supervision_profile='deployed_first2_v1',prior_gripper_target_profile=profile,deployed_prefix_loss_weight=.01)
    policy=build_history_policy(c).train();target=torch.zeros(2,20,8);padding=torch.zeros(2,20,dtype=torch.bool);padding[1,12:]=True
    parameters=list(policy.gripper_adapter.parameters())
    loss=policy(q,image,target,padding)['loss']
    expected=torch.cat([g.flatten() for g in torch.autograd.grad(loss,parameters)])
    prior=BoundedActPolicy.forward(policy,q,image,return_unbounded=True)
    contribution=gripper_contributions(policy,prior,q,target,padding)
    actual=torch.cat([g.flatten() for g in torch.autograd.grad(contribution.sum(),parameters)])
    assert contribution.shape==(2,) and expected.norm()>0
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-8)


def test_minimum_norm_balance_detects_conflicting_and_compatible_groups():
    from training.grouped_gripper import minimum_norm_weights
    np.testing.assert_allclose(minimum_norm_weights(np.eye(2)),[.5,.5],atol=1e-10)
    alpha=minimum_norm_weights(np.array([[1.],[-1.]]))
    np.testing.assert_allclose(alpha@[ [1.],[-1.] ],[0.],atol=1e-10)
    for bad in [np.empty((0,2)),np.array([[np.nan]]),np.array([1.,2.])]:
        with pytest.raises(ValueError):minimum_norm_weights(bad)
