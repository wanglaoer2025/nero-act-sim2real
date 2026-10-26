import unittest
import numpy as np
import torch
VERSION="previous_gripper_target_delta_v1"
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

