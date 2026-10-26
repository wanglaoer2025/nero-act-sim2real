import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'nero_act_grouped_gripper_20261021'))
from preservation import preservation_loss


def test_future_unexecuted_points_do_not_enter_approach_loss():
    reference=torch.zeros(2,20,8,requires_grad=True)
    prediction=torch.zeros(2,20,8);prediction[:,2:]=100
    prediction.requires_grad_()
    stats={'action_mean':[0.]*8,'action_std':[1.]*8}
    loss=preservation_loss(prediction,reference,stats)
    assert loss.item()==0
    prediction.data[:,0,0]=.001
    loss=preservation_loss(prediction,reference,stats);loss.backward()
    assert loss>0 and reference.grad is None
    assert torch.count_nonzero(prediction.grad[:,2:])==0
