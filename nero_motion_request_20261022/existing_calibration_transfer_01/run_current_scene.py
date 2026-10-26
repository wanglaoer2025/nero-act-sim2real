"""Complete offline ACT development trial; no robot transport is imported or used."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ACT = Path('/home/user/wq/nero_act_transfer_resume_20261021')
sys.path.insert(0, str(ACT))
from checked_runtime import load_projected
from simulation.supervised_rollout import rollout
from simulation.visual_evaluation import full_task_passed
from pipeline import snapshot, require_unchanged


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    preview_path = HERE/'live_preview_table/report.json'
    preview = json.loads(preview_path.read_text())
    records = preview['records']
    if preview['error'] is not None or len(records) != 10:
        raise ValueError('Ten fresh finite observed inputs required for this development case')
    if preview['can_tx_before'] != preview['can_tx_after']:
        raise ValueError('Read-only preview observed a CAN TX change')
    states = np.asarray([r['feedback_state'] for r in records])
    if (states.shape != (10, 8) or not np.isfinite(states).all()
            or np.ptp(states[:, :7], axis=0).max() > .0003):
        raise ValueError('Stationary finite eight-field initial state required')
    scene_path = ACT/'results/unqualified_nominal_full_development/scene_configuration.json'
    scene = json.loads(scene_path.read_text())
    original = copy.deepcopy(scene)
    cup = np.median([r['cup_bottom_m'] for r in records], axis=0)
    target = np.median([r['target_m'] for r in records], axis=0)
    state = np.median(states, axis=0)
    scene['pick_xy_m'] = cup[:2].tolist()
    scene['place_xy_m'] = target[:2].tolist()
    scene['visual_marker_xy_m'] = target[:2].tolist()
    scene['initial_robot_state'] = dict(qpos_rad=state[:7].tolist(), gripper_total_width_m=float(state[7]))
    scene['hardware_deployment_allowed'] = False
    scene['source_geometry_evidence'] = str(preview_path)
    assert scene['camera_calibration'] == original['camera_calibration']
    assert scene['tcp_link7_xyz_m'] == original['tcp_link7_xyz_m']
    assert scene['table_top_z_m'] == original['table_top_z_m']
    for name in ['joint_speed_limit_rad_s', 'finger_force_limit_n', 'lift_height_m']:
        assert scene[name] == original[name]
    hashes = snapshot([Path(__file__), scene_path, preview_path,
                       ACT/'checked_runtime.py', ACT/'training/projected_A/policy_last.ckpt'])
    policy, stats = load_projected()
    hashes.update(snapshot([Path(sys.modules['simulation.supervised_rollout'].__file__),
                            Path(sys.modules['simulation.visual_evaluation'].__file__)]))
    print(json.dumps(dict(stage='starting_complete_current_scene_development',
                          cup_xy_m=scene['pick_xy_m'], target_xy_m=scene['place_xy_m'],
                          pick_shift_from_training_mm=(1000*(cup[:2]-original['pick_xy_m'])).tolist(),
                          retained_TCP=scene['tcp_link7_xyz_m'], hardware_commands_sent=False)), flush=True)
    started = time.monotonic()
    queries = [0]

    def report_query(module, args, result):
        queries[0] += 1
        if queries[0] % 200 == 0:
            print(json.dumps(dict(stage='ACT_simulation', queries=queries[0],
                                  runtime_s=round(time.monotonic()-started, 2))), flush=True)

    hook = policy.register_forward_hook(report_query)
    try:
        scenario = dict(name='current_observed_scene_retained_calibration_development',
                        seed=191899, query_stride=2, servo_delay_steps=0,
                        observation_age_s=.05, jitter=False, force=False, scene=copy.deepcopy(scene))
        result = rollout(scene, policy, stats, scenario, HERE/'current_scene_full',
                         runtime_profile='history_common_v1')
    finally:
        hook.remove()
    require_unchanged(hashes)
    summary = dict(mode='current_observed_scene_retained_calibration_development',
                   full_task_passed=full_task_passed(result), result=result,
                   source_sha256=hashes, original_qualification_passed=False,
                   geometry_independently_verified=False, retained_calibration_used=True,
                   extra_optimizer_updates=0, manual_measurements_requested=False,
                   hardware_commands_sent=False, hardware_deployment_allowed=False)
    (HERE/'current_scene_full_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['result', 'source_sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
