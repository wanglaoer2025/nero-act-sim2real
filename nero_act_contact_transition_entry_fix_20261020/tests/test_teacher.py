import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class TeacherTests(unittest.TestCase):
    def test_release_teacher_refuses_unsupported_cup(self):
        from simulation.teacher import check_stage_start
        with self.assertRaisesRegex(RuntimeError,'supported'):
            check_stage_start('release',dict(supported=False,cup_upright=1.))
        with self.assertRaisesRegex(RuntimeError,'upright'):
            check_stage_start('release',dict(supported=True,cup_upright=.8))
        check_stage_start('release',dict(supported=True,cup_upright=1.))

    def test_retreat_requires_actual_open_fingers(self):
        from simulation import teacher
        self.assertTrue(callable(getattr(teacher, 'check_stage_start', None)))
        with self.assertRaises(RuntimeError):
            teacher.check_stage_start('retreat', {'gripper_width': .082, 'bilateral': False})
        teacher.check_stage_start('retreat', {'gripper_width': .095, 'bilateral': False})

    def test_contact_aware_expert_width_reserves_raw_prefix_margin(self):
        from simulation.teacher import expert_gripper_target
        config = dict(control_hz=20, gripper_speed_limit_m_s=.004,
                      teacher_gripper_feedback_lead_m=.008)
        self.assertGreaterEqual(expert_gripper_target(.065, .067, config, measured_width=.075), .0669999)

    def test_snapshot_corrections_have_explicit_partial_start_stages(self):
        for mode, first in [('grasp_to_retreat', 'close'), ('carry_to_retreat', 'carry'),
                            ('place_to_retreat', 'place'), ('release_to_retreat','release'), ('descend_to_retreat','descend')]:
            config = copy.deepcopy(self.config);config['continuation'] = mode
            self.assertEqual(self.waypoints(config)[0][0], first)
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                self.run_teacher(config)

    def test_delayed_expert_gripper_can_use_slower_rate_with_same_full_opening(self):
        from simulation import teacher
        self.assertTrue(callable(getattr(teacher, 'expert_gripper_target', None)))
        config = dict(control_hz=20, gripper_speed_limit_m_s=.008)
        q = .095
        for _ in range(100):
            old = q
            q = teacher.expert_gripper_target(.065, q, config)
            self.assertLessEqual(abs(q-old), .00040001)
        self.assertAlmostEqual(q, .065, places=6)

    def test_teacher_uses_tighter_configured_lead_without_rebasing_history(self):
        from simulation import teacher
        self.assertTrue(callable(getattr(teacher, 'expert_arm_target', None)), 'configurable expert lead missing')
        config = dict(control_hz=20, joint_speed_limit_rad_s=.04, teacher_command_lead_limit_rad=.01)
        value = teacher.expert_arm_target(np.full(7, .5), np.full(7, .009), np.zeros(7), config)
        np.testing.assert_allclose(value, .01, atol=1e-8)
        with self.assertRaises(RuntimeError):
            teacher.expert_arm_target(np.full(7, .5), np.full(7, .02), np.zeros(7), config)

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('simulation.teacher'),
                             'complete physical teacher is missing')
        from simulation.teacher import task_waypoints, run_teacher
        self.waypoints, self.run_teacher = task_waypoints, run_teacher
        self.config = json.loads((ROOT/'scene/config.json').read_text())

    def test_plan_contains_actual_transfer_release_and_retreat(self):
        points = self.waypoints(self.config)
        names = [point[0] for point in points]
        self.assertEqual(names, ['approach', 'descend', 'close', 'lift',
                                 'carry', 'place', 'release', 'retreat'])
        carry = points[4][1]
        self.assertAlmostEqual(carry[1], -.1575674779566919)
        self.assertAlmostEqual(carry[2], .255)
        self.assertGreater(points[-1][2], .085)

    def test_unreachable_or_invalid_opening_is_rejected(self):
        config = copy.deepcopy(self.config)
        config['gripper_open_m'] = .070
        with self.assertRaises(ValueError):
            self.waypoints(config)

    def test_continuation_does_not_relabel_full_grasp_data(self):
        config = copy.deepcopy(self.config)
        config['continuation'] = 'retreat_only'
        self.assertEqual([x[0] for x in self.waypoints(config)], ['retreat'])

    def test_lift_path_keeps_orientation_and_vertical_tcp(self):
        import mujoco
        from simulation import teacher
        from simulation.scene import build_scene, tcp_in_link7
        self.assertTrue(callable(getattr(teacher, 'cartesian_path', None)),
                        'joint-only interpolation does not preserve the grasp pose')
        model = build_scene(self.config)
        rotation = np.asarray(self.config['ik_goal_rotation_link7'])@tcp_in_link7(model)[1]
        start = teacher.inverse_kinematics(model, [-.51842272, .00846063, .055],
                                          np.asarray(self.config['initial_robot_state']['qpos_rad']), rotation)
        path = teacher.cartesian_path(model, start, [-.51842272, .00846063, .255], rotation)
        data = mujoco.MjData(model)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
        for q in path:
            data.qpos[:7] = q
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.site_xpos[site][:2], [-.51842272, .00846063], atol=.0002)
            np.testing.assert_allclose(data.site_xmat[site].reshape(3, 3), rotation, atol=.002)

    def test_robot_table_collision_is_not_hidden_by_valid_cup_contacts(self):
        import mujoco
        from simulation.scene import build_scene
        from simulation.teacher import observe
        config = copy.deepcopy(self.config)
        config['table_top_z_m'] = .2
        model = build_scene(config)
        data = mujoco.MjData(model)
        data.qpos[:7] = config['initial_robot_state']['qpos_rad']
        data.qpos[7:9] = [.0475, -.0475]
        mujoco.mj_forward(model, data)
        measurement = observe(model, data, config)
        self.assertTrue(measurement.get('unexpected_robot_contact', False),
                        'arm/table penetration must invalidate a physical rollout')

    @unittest.skipUnless(os.environ.get('NERO_PHYSICS_TEST') == '1', 'full physical rollout opt-in')
    def test_low_force_full_physical_transfer(self):
        result = self.run_teacher(self.config)
        self.assertTrue(result['success'], {k: v for k, v in result.items() if k != 'trace'})
        self.assertGreater(result['max_lift_m'], .06)
        self.assertGreater(result['maximum_horizontal_transfer_m'], .1)
        self.assertGreaterEqual(result['stable_released_frames'], 100)
        self.assertFalse(result['constraint_attachment_used'])

    @unittest.skipUnless(os.environ.get('NERO_PHYSICS_TEST') == '1', 'physical correction rollout opt-in')
    def test_retreat_correction_is_explicitly_partial_and_preserves_supported_cup(self):
        config = copy.deepcopy(self.config)
        config['continuation'] = 'retreat_only'
        config['initial_robot_state']['qpos_rad'] = [.47, .65, -.1, 1.51, .29, .38, -.52]
        result = self.run_teacher(config)
        self.assertEqual(result['task'], 'retreat_correction')
        self.assertFalse(result['full_task_verified'])
        self.assertTrue(result['success'], {k: v for k, v in result.items() if k != 'trace'})
        self.assertTrue(result['last_observation']['supported'])
        self.assertEqual([x['stage'] for x in result['stages']], ['retreat'])


if __name__ == '__main__':
    unittest.main()
