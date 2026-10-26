import copy
import unittest
from pathlib import Path
import numpy as np
import torch
import yaml
from src.act.nero_kinematics import SerialArmKinematics
from src.act.nero_policy import NeroActPolicy


def specification():
    return {'links':[{'rotation':np.eye(3).tolist(),'position':[0,0,0],
        'joint_index':i,'axis':[0,0,1],'joint_position':[0,0,0]} for i in range(7)],
        'tcp_position':[1,0,0]}


class KinematicFeatureTests(unittest.TestCase):
    def test_geometry_uses_supplied_observed_joints_and_static_centres(self):
        config=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/act_nero_day5.yaml').read_text())
        config.update(state_dim=15,coordinate_conditioning=True,previous_gripper_conditioning=True,
            device='cpu',pretrained_backbone=False,dropout=0.,task_displacement_features=True,
            kinematic_state_mean=[0.]*15,kinematic_state_std=[1.]*15,
            kinematic_feature_scale_m=.1,feedback_kinematics_specification=specification())
        policy=NeroActPolicy(config)
        q=torch.zeros(1,15);q[:,8:11]=torch.tensor([1.,0.,0.]);q[:,11:14]=torch.tensor([0.,1.,0.])
        state=policy.encoder_state(q)
        self.assertEqual(state.shape,(1,21))
        torch.testing.assert_close(state[:,:15],q)
        torch.testing.assert_close(state[:,15:],torch.tensor([[0.,0.,0.,-10.,10.,0.]]))
        changed=q.clone();changed[:,0]=torch.pi/2
        torch.testing.assert_close(policy.encoder_state(changed)[:,15:],torch.tensor([[10.,-10.,0.,0.,0.,0.]]),atol=2e-6,rtol=1e-6)
        changed=q.clone();changed[:,14]=42.
        torch.testing.assert_close(policy.encoder_state(changed)[:,15:],state[:,15:])
        self.assertFalse(any(k.startswith(('kinematic_','feedback_kinematics.')) for k in policy.state_dict()))

    def test_corrupt_serialized_axes_and_duplicate_joints_reject(self):
        bad=specification();bad['links'][0]['axis']=[0,0,2]
        with self.assertRaises(ValueError):SerialArmKinematics(bad)
        bad=specification();bad['links'][0]['joint_index']=1
        with self.assertRaises(ValueError):SerialArmKinematics(bad)
        q=torch.zeros(7,dtype=torch.float64,requires_grad=True);q.data[0]=.3
        actual=SerialArmKinematics(specification())(q)
        torch.testing.assert_close(actual,torch.stack([q.sum().cos(),q.sum().sin(),q.sum()*0]))
        actual.sum().backward();self.assertTrue(torch.isfinite(q.grad).all())


if __name__=='__main__':unittest.main()
