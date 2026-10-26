"""A preservation term must not overwhelm the task gradient by hundreds of times."""
import sys
from pathlib import Path
import unittest

import torch
import numpy as np
from objective import pregrasp_goal, preservation_scale, executed_objective

sys.path.insert(0, str(Path(__file__).resolve().parent))


class PreservationBudgetTests(unittest.TestCase):
    def test_preservation_gradient_stays_below_half_the_task_gradient(self):
        # These norm values come from the frozen failing-batch diagnostic.
        parameter = torch.zeros(2, requires_grad=True)
        task = .418131*parameter[0]
        preservation = 1869.72061*parameter[1]
        scale, norms = preservation_scale(task, preservation, [parameter])
        (task+scale*preservation).backward()
        self.assertAlmostEqual(float(parameter.grad[0]), .418131, places=6)
        self.assertLessEqual(abs(float(parameter.grad[1])), .5*.418131+1e-6)

    def test_opposing_preservation_cannot_reverse_task_learning(self):
        parameter = torch.tensor(0., requires_grad=True)
        task = 2*parameter; preservation = -2000*parameter
        scale, _ = preservation_scale(task, preservation, [parameter])
        (task+scale*preservation).backward()
        self.assertGreater(float(parameter.grad), 0.)

    def test_zero_task_gradient_cannot_trigger_large_preservation_update(self):
        parameter = torch.tensor(0., requires_grad=True)
        scale, _ = preservation_scale(parameter*0., parameter*2000, [parameter])
        self.assertEqual(scale, 0.)

    def test_off_target_low_pose_requires_vertical_clearance_in_teacher(self):
        # The failing actor is57mm above the table and78mm away from the cup.
        goal, role = pregrasp_goal([-.570, .021, .057], [-.633, -.026, 0.])
        self.assertEqual(role, 'clearance')
        self.assertGreaterEqual(goal[2], .20)
        np.testing.assert_allclose(goal[:2], [-.570, .021], atol=1e-12)

    def test_high_pose_aligns_before_descending_and_aligned_pose_descends(self):
        goal, role = pregrasp_goal([-.570, .021, .255], [-.633, -.026, 0.])
        self.assertEqual(role, 'align')
        np.testing.assert_allclose(goal, [-.633, -.026, .255])
        goal, role = pregrasp_goal([-.633, -.026, .155], [-.633, -.026, 0.])
        self.assertEqual(role, 'descend')
        np.testing.assert_allclose(goal, [-.633, -.026, .055])

    def test_only_executed_arm_points_receive_supervision(self):
        import json
        import h5py
        from learning import ROOT
        from training.pose_objective import CartesianPoseLoss, fk_spec_from_scene
        manifest=json.loads((ROOT/'data/manifest.json').read_text())
        with h5py.File(manifest['episodes'][0]['path']) as f:
            scene=json.loads(f.attrs['scene_config'])
        stats=dict(action_mean=np.zeros(8),action_std=np.ones(8))
        fk=CartesianPoseLoss(fk_spec_from_scene(scene),stats)
        prediction=torch.zeros((1,20,8),requires_grad=True)
        target=torch.full((1,20,8),.001)
        padding=torch.zeros((1,20),dtype=torch.bool)
        loss,_=executed_objective(prediction,target,padding,stats,fk)
        loss.backward()
        self.assertGreater(float(prediction.grad[:,:2].abs().sum()),0.)
        self.assertEqual(float(prediction.grad[:,2:].abs().sum()),0.)


if __name__ == '__main__':
    unittest.main()
