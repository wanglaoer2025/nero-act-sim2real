import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

class PoseObjectiveTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('training.pose_objective'),
            'the training objective must explicitly supervise end-effector orientation')
        from training import pose_objective
        return pose_objective

    def test_pose_fk_matches_mujoco_position_and_rotation_at_two_independent_poses(self):
        module=self.module()
        import mujoco
        from simulation.scene import build_scene
        from src.act.nero_kinematics import SerialArmKinematics
        config=json.loads((ROOT/'scene/center.json').read_text());model=build_scene(config)
        spec=module.fk_spec_from_scene(config);kin=SerialArmKinematics(spec);data=mujoco.MjData(model)
        site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
        for q in [config['initial_robot_state']['qpos_rad'],[.1,.9,.05,1.5,-.1,.1,-1.1]]:
            data.qpos[:7]=q;mujoco.mj_forward(model,data)
            joints=torch.tensor(q,dtype=torch.float64,requires_grad=True)
            pos,rot=kin(joints,return_rotation=True)
            np.testing.assert_allclose(pos.detach(),data.site_xpos[site],atol=1e-10)
            np.testing.assert_allclose(rot.detach(),data.site_xmat[site].reshape(3,3),atol=1e-10)
            torch.testing.assert_close(kin(joints),pos)
            (pos.sum()+rot.sum()).backward();self.assertTrue(torch.isfinite(joints.grad).all())

    def test_pure_rotation_with_zero_tcp_error_is_penalized_with_correct_gradient(self):
        module=self.module()
        spec=dict(links=[dict(rotation=np.eye(3).tolist(),position=[0,0,0],joint_index=i,
            axis=[0,0,1],joint_position=[0,0,0]) for i in range(7)],tcp_position=[0,0,0])
        stats=dict(action_mean=[0.]*8,action_std=[2.]+[1.]*7)
        loss=module.CartesianPoseLoss(spec,stats)
        a=torch.zeros(1,20,8);a[:,:,0]=.05;a.requires_grad_()
        position,rotation=loss(a,torch.zeros_like(a),torch.zeros(1,20,dtype=torch.bool))
        self.assertEqual(float(position),0.)
        self.assertAlmostEqual(float(rotation),2*np.sin(.05)/np.deg2rad(1),places=5)
        rotation.backward();self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(float(a.grad[0,0,0]),0)
        self.assertEqual(loss.state_dict(),{})

    def test_padded_future_and_identical_targets_do_not_add_loss_or_nan_gradients(self):
        module=self.module()
        spec=dict(links=[dict(rotation=np.eye(3).tolist(),position=[0,0,0],joint_index=i,
            axis=[0,0,1],joint_position=[0,0,0]) for i in range(7)],tcp_position=[1,0,0])
        loss=module.CartesianPoseLoss(spec,dict(action_mean=[0.]*8,action_std=[1.]*8))
        a=torch.zeros(1,20,8);a[:,10:,0]=1.;a.requires_grad_()
        padding=torch.zeros(1,20,dtype=torch.bool);padding[:,10:]=True
        p,r=loss(a,torch.zeros_like(a),padding);self.assertEqual(float(p+r),0.)
        (p+r).backward();self.assertTrue(torch.isfinite(a.grad).all())

    def test_actual_training_adapter_uses_pose_loss_and_keeps_inference_weights_compatible(self):
        module=self.module()
        from training.train import FullTaskPolicy
        from executor.model import BoundedActPolicy
        c=json.loads((ROOT/'model/run15/training_config.json').read_text())
        c.update(device='cpu',pretrained_backbone=False,dropout=0.,pose_loss_weight=.05,
            pose_kinematics_spec=module.fk_spec_from_scene(json.loads((ROOT/'scene/center.json').read_text())),
            prefix_margin_loss_weight=0.,training_observation_noise_normalized=[0.]*8)
        torch.set_num_threads(2);p=FullTaskPolicy(c);p.eval()
        q=torch.zeros(1,9);image=torch.zeros(1,1,3,90,160);targets=torch.zeros(1,20,8)
        out=p(q,image,targets,torch.zeros(1,20,dtype=torch.bool))
        self.assertGreater(float(out['loss']),float(out['l1']))
        out['loss'].backward();self.assertTrue(torch.isfinite(p.model.action_head.weight.grad).all())
        inferred=BoundedActPolicy(c);inferred.load_state_dict(p.state_dict(),strict=True);inferred.eval()
        with torch.no_grad():torch.testing.assert_close(inferred(q,image),p(q,image))
