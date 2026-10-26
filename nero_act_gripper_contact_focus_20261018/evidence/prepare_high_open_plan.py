"""Offline only: reuse the existing empty-gripper preparation interpolation."""
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), '/home/user/wq/nero_empty_demo_20261016']
from air_prepare import preparation_rows
from executor.contract import FullTaskContract
from simulation.scene import build_scene
from simulation.teacher import observe


def main():
    model_dir = ROOT/'model/run26_current_fixed_scene'
    scene = json.loads((model_dir/'scene_config.json').read_text())
    config = json.loads((model_dir/'training_config.json').read_text())
    feedback = ROOT/'evidence/current_readonly_connection_check.json'
    start = np.asarray(json.loads(feedback.read_text())['feedback']['positions'])
    target = start.copy()
    target[7] = scene['initial_robot_state']['gripper_total_width_m']
    contract = FullTaskContract(config)
    rows = np.asarray(preparation_rows(start, target,
        np.column_stack([contract.lower, contract.upper])))
    path = np.vstack([start, rows])
    assert np.array_equal(path[:, :7], np.broadcast_to(start[:7], path[:, :7].shape))
    assert np.max(abs(np.diff(path[:, 7]))) <= .0005
    assert np.all(np.diff(path[:, 7]) >= 0) and np.array_equal(rows[-1], target)
    assert np.max(abs(target[:7]-scene['initial_robot_state']['qpos_rad'])) <= .005
    model = build_scene(scene)
    data = mujoco.MjData(model)
    observations = []
    for state in path:
        data.qpos[:7] = state[:7]
        data.qpos[7:9] = [state[7]/2, -state[7]/2]
        mujoco.mj_forward(model, data)
        measurement = observe(model, data, scene)
        assert not measurement['unexpected_contact']
        assert not measurement['bilateral']
        assert measurement['tcp_xyz'][2] >= scene['table_top_z_m']+.245
        observations.append(measurement)
    output = ROOT/'evidence/high_open_preparation_plan'
    output.mkdir(exist_ok=False)
    (output/'rows.json').write_text(json.dumps(path.tolist(), indent=2))
    report = dict(mode='offline_plan_only', hardware_connected=False,
        motion_sent=False, gripper_sent=False, execution_permission=False,
        recheck_fresh_feedback_and_whole_robot_collision_before_execution=True,
        empty_gripper_and_onsite_clearance_confirmation_required=True,
        source_feedback_sha256=hashlib.sha256(feedback.read_bytes()).hexdigest(),
        reused_interpolation='/home/user/wq/nero_empty_demo_20261016/air_prepare.py:preparation_rows',
        reused_source_sha256=hashlib.sha256(Path('/home/user/wq/nero_empty_demo_20261016/air_prepare.py').read_bytes()).hexdigest(),
        start=start.tolist(), target=target.tolist(), duration_s=len(rows)/20,
        checked_kinematic_states=len(path), arm_commands_unchanged=True,
        max_gripper_step_m=float(np.max(abs(np.diff(path[:, 7])))),
        minimum_tcp_z_m=min(o['tcp_xyz'][2] for o in observations),
        model_contact_checks_passed=True, real_clearance_verified=False,
        force_limit_n=1., task_policy_unchanged=True)
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
