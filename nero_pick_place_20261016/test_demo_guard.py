import unittest
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace

from run_demo import authorize_stage, guard_carried_cup, close_gate_before_cancel, main, check_model_tcp, wait_for_control_graph, camera_health_barrier


class DemoGuardTests(unittest.TestCase):
    def test_camera_start_waits_for_fresh_feedback_without_reopening_or_ignoring_faults(self):
        camera = SimpleNamespace(open=Mock())
        original_open = camera.open
        health = Mock(side_effect=[TimeoutError('startup gap'), {}, {}])
        with patch('run_demo.time.sleep'):
            camera_health_barrier(camera, health)
            camera.open()
        self.assertEqual(health.call_count, 3)
        original_open.assert_called_once()
        camera2 = SimpleNamespace(open=Mock())
        camera_health_barrier(camera2, Mock(side_effect=RuntimeError('motor fault')))
        with self.assertRaisesRegex(RuntimeError, 'motor fault'):
            camera2.open()
        camera3 = SimpleNamespace(open=Mock())
        camera_health_barrier(camera3, Mock(side_effect=TimeoutError('stale')), timeout_s=0)
        with self.assertRaises(TimeoutError):
            camera3.open()
    def test_chinese_confirmation_log_works_in_ros_ascii_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / 'confirmation.json')
            code = ('from pathlib import Path; from run_demo import write_new; '
                    f'write_new(Path({target!r}), {{"action": "\\u4e0b\\u964d"}})')
            env = dict(os.environ, LC_ALL='C', PYTHONCOERCECLOCALE='0', PYTHONUTF8='0')
            result = subprocess.run([sys.executable, '-X', 'utf8=0', '-c', code],
                                    env=env, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('下降', Path(target).read_text(encoding='utf-8'))
    def test_discovery_wait_keeps_unique_node_requirement(self):
        names = [(name, '/') for name in ('agx_arm_ctrl_single_node', 'control_gate',
                 'controller_manager', 'arm_controller', 'gripper_controller', 'move_group')]
        node = Mock()
        node.get_node_names_and_namespaces.side_effect = [[('client', '/')], names]
        session = SimpleNamespace(node=node, rclpy=SimpleNamespace(spin_once=Mock()))
        wait_for_control_graph(session)
        session.rclpy.spin_once.assert_called_once()
        node.get_node_names_and_namespaces.side_effect = None
        node.get_node_names_and_namespaces.return_value = names + [('move_group', '/other')]
        with self.assertRaises(RuntimeError):
            wait_for_control_graph(session)
        node.get_node_names_and_namespaces.return_value = []
        with self.assertRaises(TimeoutError):
            wait_for_control_graph(session, timeout_s=0)
    def test_only_the_explicitly_accepted_approximate_tcp_is_allowed(self):
        check_model_tcp({'tcp_offset_flange_xyz_m': [.152, 0, -.0235], 'approximate': True})
        for xyz in [[.135, 0, 0], [.152, 0, 0], [float('nan'), 0, -.0235]]:
            with self.assertRaises(ValueError):
                check_model_tcp({'tcp_offset_flange_xyz_m': xyz, 'approximate': True})
        with self.assertRaises(ValueError):
            check_model_tcp({'tcp_offset_flange_xyz_m': [.152, 0, -.0235], 'approximate': False})
    def test_known_colliding_historical_profile_cannot_start_execution(self):
        with patch('sys.argv', ['run_demo.py', '--execute', '--targets', '/unused/targets',
                                '--output', '/unused/output']):
            with patch.object(Path, 'mkdir', side_effect=AssertionError('execution accessed files')):
                with self.assertRaises(PermissionError):
                    main()

    def test_operator_must_supply_exact_stage_token(self):
        with patch('builtins.input', return_value='GRASP'):
            self.assertTrue(authorize_stage('close', 'GRASP'))
        for answer in ['NEXT', '', 'grasp']:
            with patch('builtins.input', return_value=answer):
                self.assertFalse(authorize_stage('close', 'GRASP'))

    def test_eof_does_not_authorize(self):
        with patch('builtins.input', side_effect=EOFError):
            self.assertFalse(authorize_stage('release', 'RELEASE'))

    def test_stale_target_cannot_be_authorized_by_a_correct_token(self):
        with patch('builtins.input', return_value='GRASP'):
            with self.assertRaises(ValueError):
                authorize_stage('close', 'GRASP', targets={'capture_wall_s': 0}, now=100)

    def test_sweep_checks_only_the_three_held_cup_stages(self):
        session = SimpleNamespace(joint_state=lambda: 'state', plan_to_pose=Mock(return_value='trajectory'),
                                  plan_cartesian=Mock(return_value='trajectory'))
        keys = ('pregrasp_high_pose', 'grasp_pose', 'lift_pose', 'carry_pose', 'place_pose', 'retreat_pose')
        plan = {'flange': {key: [index]*6 for index, key in enumerate(keys)}}
        plan['flange']['lift_pose'] = plan['flange']['pregrasp_high_pose']
        plan['flange']['retreat_pose'] = plan['flange']['carry_pose']
        with patch('cup_sweep.validate_cup_trajectory', return_value={'fk_samples': 1}) as sweep:
            report = {}
            guard_carried_cup(session, plan, report)
            for key in keys:
                session.plan_to_pose(plan['flange'][key], .01)
            self.assertEqual(sweep.call_count, 3)
            self.assertEqual([row['stage'] for row in report['cup_sweep_checks']],
                             ['lift_pose', 'carry_pose', 'place_pose'])
            session.plan_cartesian.assert_called_once()

    def test_action_cancel_still_runs_when_gate_service_fails(self):
        service = SimpleNamespace(SetBool=SimpleNamespace(Request=lambda **kw: SimpleNamespace(**kw)))
        client = Mock()
        client.wait_for_service.return_value = False
        original_cancel = Mock()
        session = SimpleNamespace(node=SimpleNamespace(create_client=Mock(return_value=client)),
                                  cancel=original_cancel)
        with patch.dict('sys.modules', {'std_srvs.srv': service}):
            report = {}
            close_gate_before_cancel(session, report)
            session.cancel()
        original_cancel.assert_called_once()
        self.assertIn('gate_close_error', report)


if __name__ == '__main__':
    unittest.main()
