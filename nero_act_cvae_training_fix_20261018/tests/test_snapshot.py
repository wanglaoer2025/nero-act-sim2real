import importlib.util
import json
from pathlib import Path
import unittest
import mujoco
import numpy as np
from simulation.scene import build_scene
from simulation.nero_domain_randomization import NeroServo


class SnapshotTests(unittest.TestCase):
    def test_restored_integration_and_delayed_servo_produce_identical_next_physics(self):
        self.assertIsNotNone(importlib.util.find_spec('simulation.snapshot'))
        from simulation.snapshot import capture, restore
        cfg = json.loads((Path(__file__).resolve().parents[1]/'scene/config.json').read_text())
        cfg['action_delay_steps'] = 2
        model = build_scene(cfg);data = mujoco.MjData(model)
        data.qpos[:7] = cfg['initial_robot_state']['qpos_rad'];data.qpos[7:9] = [.0475, -.0475]
        mujoco.mj_forward(model, data);servo = NeroServo(model, data, cfg)
        previous = np.r_[data.qpos[:7], .095]
        for _ in range(5):servo.command(previous[:7], previous[7]);servo.apply();mujoco.mj_step(model, data)
        saved = capture(model, data, servo, previous)
        other_model = build_scene(cfg);other = mujoco.MjData(other_model)
        other_servo = NeroServo(other_model, other, cfg)
        restored_previous = restore(other_model, other, other_servo, saved)
        np.testing.assert_array_equal(restored_previous, previous)
        np.testing.assert_array_equal(other.qpos, data.qpos)
        np.testing.assert_array_equal(other.qvel, data.qvel)
        for _ in range(25):
            servo.apply();mujoco.mj_step(model, data)
            other_servo.apply();mujoco.mj_step(other_model, other)
        np.testing.assert_allclose(other.qpos, data.qpos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(other.qvel, data.qvel, atol=1e-12, rtol=0)
        bad = dict(saved, servo_queue=[])
        with self.assertRaises(ValueError):restore(other_model, other, other_servo, bad)


if __name__ == '__main__':unittest.main()
