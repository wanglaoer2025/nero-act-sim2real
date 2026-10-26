"""Physical teacher corrections only. Never imports ROS, SDK, or a CAN sender."""
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

from protocol import offsets

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'nero_act_position_response_repair_20261023'
sys.path.insert(0, str(SOURCE))
import prepare as source
from simulation.collect import verify_replay, verify_action_contract
from training.legacy_visual_data import derive_episode, training_profile, sha


def main():
    output = ROOT / 'data'
    output.mkdir(exist_ok=False)
    poses = source.definitions()
    protocol = dict(poses=poses, offsets_train_m=offsets(), original_validation_offsets_unchanged=True,
                    partial_teacher_only=True, failed_ACT_actions_used_as_labels=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    manifest = dict(complete=False, episodes=[], rejected=[], hardware_deployment_allowed=False,
                    failed_ACT_actions_used_as_labels=False)
    index = 5000
    started = time.monotonic()
    for pose in poses:
        for dx, dy in offsets():
            index += 1
            scene = copy.deepcopy(pose['scene'])
            scene.update(initial_robot_state=dict(qpos_rad=pose['q'], gripper_total_width_m=.095),
                         action_delay_steps=0, joint_speed_limit_rad_s=.03,
                         teacher_command_lead_limit_rad=.016, gripper_speed_limit_m_s=.004,
                         teacher_gripper_feedback_lead_m=.008)
            scene.pop('arm_disturbance', None)
            scene.pop('initial_simulation_snapshot', None)
            scene['pick_xy_m'] = (np.array(scene['pick_xy_m']) + [dx, dy]).tolist()
            group = f'dense_{pose["key"]}_{dx:+.3f}_{dy:+.3f}'
            scene['generalization'] = dict(index=index, split='train', source_group=group,
                pose_key=pose['key'], offset_xy_m=[dx, dy],
                parent_pose_sha256=pose['source_sha256'], synthetic_stationary_initialization=True)
            physical = output / 'physical' / f'episode_{index}.hdf5'
            physical.parent.mkdir(exist_ok=True)
            try:
                result = source.collect_prefix(scene, physical)
                replay, contract = verify_replay(physical), verify_action_contract(physical)
                if not replay['passed'] or not contract['passed']:
                    raise ValueError('Physical correction replay or action contract failed')
                report = derive_episode(physical, output / 'visual' / physical.name,
                    expected_sha=sha(physical), profile=training_profile(scene))
                manifest['episodes'].append(dict(report, id=index, status='valid', split='train',
                    source_group=group, pose_key=pose['key'], offset_xy_m=[dx, dy], role=result['role'],
                    full_task_verified=False, physical_path=str(physical.resolve()),
                    physical_sha256=sha(physical), replay=replay, prefix_contract_audit=contract))
            except (RuntimeError, ValueError) as exc:
                manifest['rejected'].append(dict(id=index, pose_key=pose['key'], offset_xy_m=[dx, dy],
                                               error=str(exc)))
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(dict(pose=pose['key'], accepted=len(manifest['episodes']),
            rejected=len(manifest['rejected']), elapsed_s=time.monotonic()-started)), flush=True)
    roles = {r['role'] for r in manifest['episodes']}
    if not {'align', 'clearance'} <= roles:
        raise ValueError('Both alignment and vertical-clearance data required')
    manifest['complete'] = True
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
