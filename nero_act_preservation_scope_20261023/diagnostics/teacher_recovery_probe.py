"""Zero-update development diagnostic. Teacher success is never ACT success."""
import copy
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/'nero_act_dense_response_20261023'))
import evaluate_dense
evaluation = evaluate_dense.evaluation
from executor.model import physical_actions
from executor.ros_execution import RobotGeometry
from simulation.collect import collect_episode, verify_replay, verify_action_contract
from training.legacy_visual_data import derive_episode, training_profile, sha
from training.visual_dataset import VisualReferenceDataset


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    evaluation.ROOT = ROOT
    policy, stats = evaluation.load_candidate()
    source = ROOT/'results/restored_current'
    scene = json.loads((source/'scene_configuration.json').read_text())
    saved = json.loads((source/'physical_snapshots.json').read_text())['snapshots']
    with np.load(source/'rollout.npz') as stream:
        states = stream['feedback_at_query']
    geometry = RobotGeometry(scene)
    poses = np.array([geometry.pose(state)['tcp'] for state in states])
    distances = np.linalg.norm(poses[:,:2]-scene['pick_xy_m'], axis=1)
    nearest = int(distances.argmin())
    later = np.flatnonzero((np.arange(len(distances))>nearest) & (distances>=distances[nearest]+.01))
    assert len(later), 'Outward drift was not observed; do not invent a drift state'
    chosen = [('nearest_alignment', nearest), ('outward_drift', int(later[0])), ('terminal', len(states)-1)]
    output = ROOT/'diagnostics/teacher_recovery'
    output.mkdir(exist_ok=False)
    bound = evaluation.snapshot([Path(__file__), source/'physical_snapshots.json', source/'rollout.npz',
                                 ROOT/'training/pilot_A/policy_last.ckpt'])
    report = dict(optimizer_updates=0, diagnostic_only=True, new_training=False,
                  teacher_success_is_not_ACT_success=True, nominal_scene_only=True,
                  hardware_commands_sent=False, records=[], source_sha256=bound)
    for index, (name, query) in enumerate(chosen, 7000):
        snapshot = saved[query]
        config = copy.deepcopy(scene)
        config['initial_simulation_snapshot'] = snapshot
        config['initial_robot_state'] = dict(qpos_rad=snapshot['qpos'][:7],
            gripper_total_width_m=snapshot['qpos'][7]-snapshot['qpos'][8])
        config['generalization'] = dict(index=index, split='diagnostic',
            source_group=f'zero_update_{name}_{query}', parent_query=query,
            actual_saved_physics_and_servo_state=True)
        physical = output/f'{name}.hdf5'
        record = dict(name=name, query=query, tcp_m=poses[query].tolist(),
                      source_xy_error_m=float(distances[query]))
        try:
            result = collect_episode(config, physical)
            replay, contract = verify_replay(physical), verify_action_contract(physical)
            assert result['success'] and replay['passed'] and contract['passed']
            visual = output/f'{name}_visual.hdf5'
            derived = derive_episode(physical, visual, expected_sha=sha(physical), profile=training_profile(config))
            dataset = VisualReferenceDataset([visual], ['camera_0'], stats, policy_config=policy.config,
                image_size=tuple(policy.config['image_size']), sample_starts=[(0,0)])
            image, current, target, padding, _ = dataset[0]
            assert not padding[:2].any()
            with torch.inference_mode():
                actor = physical_actions(policy(current[None].cuda(), image[None].cuda()), stats,
                                         bounded_gripper=True)[0].cpu().numpy()
            teacher = physical_actions(target, stats, bounded_gripper=True).numpy()
            actor_poses = np.array([geometry.pose(row)['tcp'] for row in actor[:2]])
            teacher_poses = np.array([geometry.pose(row)['tcp'] for row in teacher[:2]])
            record.update(teacher_complete_success=True, teacher_frames=replay['frames'],
                replay=replay, contract=contract, derived=derived,
                maximum_prefix_position_error_m=float(np.linalg.norm(actor_poses-teacher_poses,axis=1).max()),
                actor_prefix_delta_m=(actor_poses-poses[query]).tolist(),
                teacher_prefix_delta_m=(teacher_poses-poses[query]).tolist(),
                previous_gripper_target_m=snapshot['previous_target'][7],
                measured_gripper_width_m=config['initial_robot_state']['gripper_total_width_m'])
        except (ValueError, RuntimeError, AssertionError) as exc:
            record.update(teacher_complete_success=False, error=str(exc))
        report['records'].append(record)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    evaluation.require_unchanged(bound)
    report['complete'] = len(report['records'])==3
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
