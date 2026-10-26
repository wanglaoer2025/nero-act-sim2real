import torch
from tests.test_visual_policy import config


def test_contact_adaptation_keeps_all_arm_and_encoder_weights_and_outputs():
    from training.visual_gripper_repair import build_contact_adapter
    torch.set_num_threads(1);torch.manual_seed(42)
    c=config();c.update(nonlinear_gripper_head=True,prior_gripper_loss_weight=.005)
    policy=build_contact_adapter(c).eval()
    q=torch.full((2,15),.2);image=torch.rand(2,1,3,90,160)
    before={n:p.detach().clone() for n,p in policy.state_dict().items() if not n.startswith('gripper_adapter.')}
    with torch.no_grad():old=policy(q,image).clone()
    target=old.detach().clone();target[:,:,7]-=.03
    padding=torch.zeros(2,20,dtype=torch.bool)
    loss=policy(q,image,target,padding)['loss'];policy.optimizer.zero_grad();loss.backward()
    from training.visual import gradient_evidence
    proof=gradient_evidence(policy)
    assert proof['gripper_grad_norm']>0 and proof['frozen_base_has_grad'] is False
    policy.optimizer.step()
    with torch.no_grad():new=policy(q,image)
    torch.testing.assert_close(old[:,:,:7],new[:,:,:7],rtol=0,atol=0)
    assert (old[:,:,7]-new[:,:,7]).abs().max()>0
    for n,p in before.items():torch.testing.assert_close(policy.state_dict()[n],p,rtol=0,atol=0)
