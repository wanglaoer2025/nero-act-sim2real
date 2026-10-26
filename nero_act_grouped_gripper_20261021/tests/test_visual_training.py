import importlib
import importlib.util

import torch
import pytest

from tests.test_visual_policy import config


def api():
    assert importlib.util.find_spec('training.visual') is not None, 'Visual15D optimizer scope missing'
    return importlib.import_module('training.visual')


def test_selected_parameters_and_cvae_receive_gradients():
    m=api();torch.set_num_threads(1);torch.manual_seed(42)
    c=config();c.update(nonlinear_gripper_head=True,prior_gripper_loss_weight=.005)
    policy=m.build_visual_policy(c).train()
    q=torch.full((1,15),.2);image=torch.rand(1,1,3,90,160)
    target=torch.linspace(-.05,.05,160).reshape(1,20,8);padding=torch.zeros(1,20,dtype=torch.bool)
    output=policy(q,image,target,padding);output['loss'].backward()
    assert float(output['kl'].detach())>0
    assert float(policy.model.latent_proj.weight.grad.abs().sum())>0
    assert float(policy.model.input_proj_robot_state.weight.grad[:,9:].abs().sum())>0
    visual_grad=0;frozen=0
    for name,p in policy.named_parameters():
        if name.startswith('model.backbones.'):
            if '.layer4.' in name:visual_grad+=float(p.grad.abs().sum()) if p.grad is not None else 0
            else:assert p.grad is None;frozen+=1
    assert visual_grad>0 and frozen>0
    grouped={id(p):g['lr'] for g in policy.optimizer.param_groups for p in g['params']}
    assert grouped[id(policy.model.input_proj_robot_state.weight)]==.0001
    assert grouped[id(policy.gripper_adapter[0].weight)]==.0003


def test_legacy_heads_only_mode_cannot_freeze_new_conditioning():
    m=api();c=config()
    for change in [dict(train_arm_output_and_gripper_only=True),dict(train_nonlinear_gripper_only=True),dict(cvae_training=False)]:
        with pytest.raises(ValueError):m.build_visual_policy(dict(c,**change))
