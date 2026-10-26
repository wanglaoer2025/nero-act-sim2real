import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
VERSION='previous_gripper_target_delta_v1'


def profile():
    return dict(chunk_size=20,state_dim=8,physical_state_dim=8,policy_input_dim=9,action_dim=8,
                task_scope='complete_pick_carry_place_retreat',gripper_command_context=VERSION)


class CommandContextTests(unittest.TestCase):
    def test_nine_model_features_never_change_eight_dimensional_physical_contract(self):
        from executor.contract import FullTaskContract
        guard=FullTaskContract(profile());state=np.r_[np.zeros(7),.074]
        raw=np.tile(state,(20,1));raw[:,0]=.005
        sent=guard.validate_prefix(raw,dict(state=state,age_s=.05),state)
        self.assertAlmostEqual(sent[0,0],.002)
        with self.assertRaises(ValueError):guard.validate_prefix(raw,dict(state=np.r_[state,0.],age_s=.05),state)
        for change in [dict(physical_state_dim=9),dict(gripper_command_context='unknown'),dict(state_dim=9),dict(policy_input_dim=8)]:
            with self.assertRaises(ValueError):FullTaskContract(dict(profile(),**change))

    def test_observation_keeps_actual_width_and_requires_real_past_command(self):
        from executor import model
        self.assertTrue(callable(getattr(model,'make_policy_observation',None)))
        stats=dict(qpos_mean=np.zeros(8),qpos_std=np.ones(8));state=np.r_[np.zeros(7),.074]
        value=model.make_policy_observation(state,stats,profile(),.065)
        self.assertEqual(value.shape,(9,));np.testing.assert_allclose(value[:8],state)
        self.assertAlmostEqual(float(value[8]),-.9,places=6)
        for past in [None,float('nan'),.096]:
            with self.assertRaises(ValueError):model.make_policy_observation(state,stats,profile(),past)
        self.assertEqual(model.make_policy_observation(state,stats,dict(state_dim=8)).shape,(8,))

    def test_dataset_reads_previous_action_or_initial_snapshot_never_future_labels(self):
        self.assertIsNotNone(importlib.util.find_spec('training.command_context'))
        from training.command_context import ReferenceDataset
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'episode_0.hdf5';actions=np.zeros((6,8),np.float32);actions[:,7]=.070+np.arange(6)*.0002
            with h5py.File(path,'w') as f:
                f.attrs['scene_config']=json.dumps(dict(initial_simulation_snapshot=dict(previous_target=[0.]*7+[.065])))
                f['action']=actions;f['observations/qpos']=np.zeros((6,7),np.float32)
                f['observations/gripper_state']=np.full((6,1),.074,np.float32)
                f['observations/images/camera_0']=np.zeros((6,90,160,3),np.uint8)
            stats=dict(qpos_mean=np.zeros(8,np.float32),qpos_std=np.ones(8,np.float32),action_mean=np.zeros(8,np.float32),action_std=np.ones(8,np.float32))
            data=ReferenceDataset([path],['camera_0'],stats,sample_starts=[(0,0),(0,2)])
            self.assertAlmostEqual(float(data[0][1][-1]),-.9,places=6)
            before=data[1][1].clone();self.assertAlmostEqual(float(before[-1]),-.38,places=5)
            with h5py.File(path,'r+') as f:f['action'][2:,7]=.09
            torch.testing.assert_close(data[1][1],before)

    def test_state_projection_expansion_preserves_legacy_prediction_and_learns_context(self):
        self.assertIsNotNone(importlib.util.find_spec('training.command_context'))
        from training.command_context import expand_context_state_dict
        from executor.model import BoundedActPolicy
        config=json.loads((ROOT/'model/candidate_run12_full_contract_recovery/training_config.json').read_text())
        config.update(device='cpu',pretrained_backbone=False);torch.set_num_threads(1)
        old=BoundedActPolicy(config).eval();new=BoundedActPolicy(dict(config,**profile())).eval()
        new.load_state_dict(expand_context_state_dict(old.state_dict()),strict=True)
        q=torch.zeros(1,8);image=torch.zeros(1,1,3,90,160)
        with torch.no_grad():torch.testing.assert_close(old(q,image),new(torch.cat([q,torch.ones(1,1)],1),image),atol=1e-6,rtol=1e-5)
        new(torch.cat([q,torch.ones(1,1)],1),image).square().mean().backward()
        self.assertGreater(float(new.model.input_proj_robot_state.weight.grad[:,-1].abs().sum()),0.)

    def test_live_worker_uses_submitted_reference_instead_of_measured_opening(self):
        from executor.runtime import LiveWorker
        worker=object.__new__(LiveWorker);state=[0.]*7+[.074];receipt=time.monotonic()
        worker.session=SimpleNamespace(spin=lambda _:None,actual=lambda:state,
            last_target=np.r_[np.zeros(7),.065],latest={'/feedback/joint_states':{'receipt':receipt}})
        worker.process=SimpleNamespace(stdin=io.StringIO());worker.records=[]
        worker.receive=lambda _:dict(type='prediction',chunk=np.tile(state,(20,1)).tolist(),details=dict(
            observed_monotonic_s=receipt,feedback_state=state,camera_received_monotonic_s=time.monotonic(),previous_gripper_target_m=.065))
        worker.predict();request=json.loads(worker.process.stdin.getvalue())
        self.assertEqual(request['previous_gripper_target_m'],.065)
        self.assertNotEqual(request['previous_gripper_target_m'],state[7])

    def test_frozen_worker_builds_the_same_nine_dimensional_observation(self):
        from executor.policy_worker import FrozenAct
        worker=object.__new__(FrozenAct);worker.config=profile();worker.device='cpu'
        worker.stats={k:np.zeros(8,np.float32) if k.endswith('mean') else np.ones(8,np.float32) for k in ['qpos_mean','qpos_std','action_mean','action_std']}
        seen=[]
        def predict(q,image):seen.append(q.clone());return torch.zeros(1,20,8)
        worker.model=predict
        worker.predict(np.zeros((90,160,3),np.uint8),np.r_[np.zeros(7),.074],previous_gripper_target_m=.065)
        self.assertEqual(seen[0].shape,(1,9));self.assertAlmostEqual(float(seen[0][0,-1]),-.9,places=6)
        with self.assertRaises(ValueError):worker.predict(np.zeros((90,160,3),np.uint8),np.r_[np.zeros(7),.074])

    def test_noise_preserves_the_same_known_past_command(self):
        from unittest.mock import patch
        from training.train import FullTaskPolicy
        from executor.model import make_policy_observation
        config=json.loads((ROOT/'model/candidate_run12_full_contract_recovery/training_config.json').read_text())
        config.update(profile());config.update(device='cpu',pretrained_backbone=False)
        policy=FullTaskPolicy(config).train();stats=config['constraint_loss_stats']
        actual=np.array(config['real_initial_state']);actual[7]=.074
        q=torch.from_numpy(make_policy_observation(actual,stats,config,.065))[None];seen=[]
        def predict(self,qpos,image):seen.append(qpos);return torch.zeros(1,20,8,requires_grad=True)
        with patch('training.train.BoundedActPolicy.forward',predict):
            policy(q,torch.zeros(1,1,3,90,160),torch.zeros(1,20,8),torch.zeros(1,20,dtype=torch.bool))
        width=seen[0][:,7]*stats['qpos_std'][7]+stats['qpos_mean'][7]
        torch.testing.assert_close(width+seen[0][:,8]*.01,torch.tensor([.065]),atol=1e-7,rtol=1e-6)

    def test_command_context_can_learn_the_release_deadband_progress(self):
        from training.train import complete_windows
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'episode_0.hdf5'
            with h5py.File(path,'w') as f:
                a=np.zeros((50,8),np.float32);a[:,7]=.065+np.arange(50)*.00025
                f['action']=a;f['observations/gripper_state']=np.full((50,1),.074)
                f['simulation/stage']=np.array([b'release']*50)
            self.assertIn((0,0),complete_windows([path],20,command_context=True))
            self.assertNotIn((0,0),complete_windows([path],20))

    def test_initialization_only_weights_can_never_be_exported(self):
        from training.export import export_model
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'initializer.ckpt';torch.save(dict(config={},initialization_only=True),path)
            with self.assertRaisesRegex(ValueError,'initializer'):export_model(path,Path(directory)/'bad_model')


if __name__=='__main__':unittest.main()
