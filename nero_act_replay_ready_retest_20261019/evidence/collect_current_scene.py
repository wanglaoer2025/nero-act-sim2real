"""Offline, physically replayed demonstrations for the measured fixed cup scene."""
import copy
import json
from pathlib import Path

import numpy as np

from simulation.collect import collect_episode, verify_replay
from executor.ros_execution import RobotGeometry

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent/'nero_act_replay_current_cup_retest_20261019'
OUT = ROOT/'training_data_current_cup'
OUT.mkdir(exist_ok=False)
observed = json.loads((BASE/'evidence/retry_readonly_02/report.json').read_text())
scene = json.loads((ROOT/'model/source_run11/scene_config.json').read_text())
scene['pick_xy_m'] = np.median([r['cup_xyz'][:2] for r in observed['observations']], axis=0).tolist()
scene['place_xy_m'] = np.median([r['target_xyz'][:2] for r in observed['observations']], axis=0).tolist()
(OUT/'scene_config.json').write_text(json.dumps(scene, indent=2))
full, corrections, rejected = [], [], []
def save(config, episode, split, origin):
    path = OUT/f'episode_{episode}.hdf5'
    result = collect_episode(config, path)
    replay = verify_replay(path)
    item = dict(path=str(path), status='valid', split=split, frames=result['frames'],
                full_task_verified=result['full_task_verified'], teacher_success=True,
                replay=replay, prefix_contract_audit=result['prefix_contract_audit'], origin=origin)
    if result['full_task_verified']:
        full.append(item)
    else:
        item.update(task_role=result['task'], correction_verified=True)
        corrections.append(item)
    print(json.dumps(dict(episode=episode, **item)), flush=True)

try:
    for index in range(4):
        config = copy.deepcopy(scene)
        rng = np.random.default_rng(2026101900+index)
        if index:
            config['pick_xy_m'] = (np.asarray(config['pick_xy_m'])+rng.uniform(-.001, .001, 2)).tolist()
            config['initial_robot_state']['qpos_rad'] = (
                np.asarray(config['initial_robot_state']['qpos_rad'])+rng.uniform(-.001, .001, 7)).tolist()
        config['appearance']['light_scale'] = [.98, 1.02, .96, 1.04][index]
        save(config, 200+index, 'train' if index < 2 else 'val',
             dict(kind='complete_current_cup_teacher', seed=2026101900+index))
    failed = BASE/'evidence/retry_cup_prototype/recorded_post_stop_source'
    snapshots = json.loads((failed/'physical_snapshots.json').read_text())
    measures = json.loads((failed/'measurements.json').read_text())
    for episode, minimum, mode, split in [
            (208, 900, 'grasp_to_retreat', 'train'),
            (209, 1600, 'grasp_to_retreat', 'val')]:
        index = next(i for i, step in enumerate(snapshots['query_steps']) if step >= minimum)
        snapshot = snapshots['snapshots'][index]
        q = np.asarray(snapshot['qpos'])
        config = copy.deepcopy(scene)
        config['pick_xy_m'] = q[9:11].tolist()
        config['initial_robot_state'] = dict(qpos_rad=q[:7].tolist(), gripper_total_width_m=float(q[7]-q[8]))
        config['ik_goal_rotation_link7'] = RobotGeometry(config).pose(np.r_[q[:7], q[7]-q[8]])['rotation'].tolist()
        config.update(initial_simulation_snapshot=snapshot, continuation=mode,
                      joint_speed_limit_rad_s=.02, teacher_command_lead_limit_rad=.015,
                      gripper_speed_limit_m_s=.004, teacher_gripper_feedback_lead_m=.008)
        config['dwell_seconds']['close'] = config['dwell_seconds']['release'] = 8
        save(config, episode, split, dict(kind='failed_policy_exact_snapshot_correction',
             source=str(failed), step=snapshots['query_steps'][index], continuation=mode))
except BaseException as exc:
    rejected.append(dict(error=f'{type(exc).__name__}: {exc}'))
finally:
    (OUT/'full_manifest.json').write_text(json.dumps(dict(episodes=full, rejected=rejected), indent=2))
    (OUT/'correction_manifest.json').write_text(json.dumps(dict(episodes=corrections, rejected=rejected), indent=2))
if rejected:
    raise RuntimeError(rejected)
