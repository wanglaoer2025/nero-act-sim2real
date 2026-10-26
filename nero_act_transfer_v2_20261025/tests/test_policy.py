import numpy as np
import pytest
import torch

torch.set_num_threads(2)


def batch():
    obs=torch.zeros(2,15);obs[:,7]=.08
    images=torch.rand(2,1,3,64,96)
    actions=torch.zeros(2,20,8)
    actions[:,:,:7]=torch.randn(2,20,7)*.02
    actions[:,:,7]=.08-torch.arange(1,21)*.0002
    padding=torch.zeros(2,20,dtype=torch.bool)
    return obs,images,actions,padding


def model():
    from policy import ACTPolicy
    stats={k:np.zeros(8).tolist() for k in ['qpos_mean','action_mean']}
    stats.update(qpos_std=[1.]*8,action_std=[1.]*8)
    return ACTPolicy(stats,pretrained=False)


def test_cvae_decoder_last_layer_backbone_and_gripper_receive_gradients():
    p=model();p.train();obs,images,actions,pad=batch()
    result=p(obs,images,actions,pad)
    assert result['prediction'].shape==(2,20,8)
    assert result['kl'].isfinite()
    result['loss'].backward()
    names=dict(p.named_parameters())
    for prefix in ['model.encoder_action_proj','model.latent_proj',
                   'model.transformer.decoder.layers.0','model.transformer.decoder.layers.1',
                   'model.backbones.0.0.body.layer4','gripper_head.2']:
        grads=[v.grad for n,v in names.items() if n.startswith(prefix)]
        assert grads and any(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in grads),prefix
    assert all(v.grad is None for n,v in names.items() if 'body.layer1' in n)


def test_deployment_uses_zero_latent_and_never_takes_actions():
    p=model().eval();obs,images,_,_=batch()
    with torch.no_grad():
        a=p(obs,images);b=p(obs,images)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert a.shape==(2,20,8)


def test_gripper_points_share_query_previous_without_cumulative_sum():
    p=model().eval();obs,images,_,_=batch()
    with torch.no_grad():
        p.gripper_head[2].bias.fill_(.5)
        out=p(obs,images)
    expected=.08+.001*np.tanh(.5)
    torch.testing.assert_close(out[:,:,7],torch.full((2,20),expected))


def test_padded_nan_actions_do_not_pollute_posterior_or_loss():
    p=model();obs,images,actions,pad=batch()
    pad[:,15:]=True;actions[:,15:]=float('nan')
    result=p(obs,images,actions,pad)
    assert torch.isfinite(result['loss'])


def test_unreachable_gripper_prefix_labels_are_rejected():
    p=model();obs,images,actions,pad=batch()
    actions[:,0,7]=.07
    with pytest.raises(ValueError,match='increment'):
        p(obs,images,actions,pad)
