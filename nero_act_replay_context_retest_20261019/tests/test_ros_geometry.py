import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np


class RosGeometryTests(unittest.TestCase):
    def test_collision_queries_bound_inflight_and_stop_after_invalid_batch(self):
        from executor import ros_execution
        self.assertTrue(callable(getattr(ros_execution, 'validate_in_batches', None)),
                        'unbounded ROS service bursts can overflow the depth10 queues')
        from types import SimpleNamespace
        pending, largest, calls = set(), [0], []
        def submit(row):
            self.assertLess(len(pending), 8)
            index = len(calls);calls.append(row);pending.add(index)
            largest[0] = max(largest[0],len(pending));return index
        def wait(index):
            pending.remove(index);return SimpleNamespace(valid=True)
        self.assertEqual(ros_execution.validate_in_batches(list(range(41)), submit, wait),41)
        self.assertEqual(largest[0],8);self.assertEqual(calls,list(range(41)));self.assertFalse(pending)
        calls.clear()
        with self.assertRaises(RuntimeError):
            ros_execution.validate_in_batches(list(range(41)), submit,
                lambda index: SimpleNamespace(valid=False))
        self.assertEqual(len(calls),8,'no later collision batches may be submitted after rejection')

    def test_checks_include_the_measured_start_and_all_intermediate_states(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.ros_execution'))
        from executor.ros_execution import checked_interpolation
        start = np.zeros(8);end = np.r_[np.full(7, .002), .0005]
        rows = checked_interpolation(start, [end])
        np.testing.assert_allclose(rows[-1], end)
        self.assertLessEqual(np.max(abs(np.diff(np.vstack([start, rows]), axis=0)[:, :7])), .000500001)
        self.assertGreaterEqual(len(rows), 4)

    def test_fk_matches_the_selected_tcp_and_returns_a_proper_rotation(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.ros_execution'))
        from executor.ros_execution import RobotGeometry
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'scene/config.json').read_text())
        geometry = RobotGeometry(config)
        q = np.r_[config['initial_robot_state']['qpos_rad'], .095]
        pose = geometry.pose(q)
        np.testing.assert_allclose(pose['tcp'], [-.526158966865, -.157513558396, .255033769543], atol=1e-9)
        np.testing.assert_allclose(pose['rotation'].T@pose['rotation'], np.eye(3), atol=1e-10)


if __name__ == '__main__':unittest.main()
