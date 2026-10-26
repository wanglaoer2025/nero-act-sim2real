"""Offline physical demonstrations for the existing 15D position-conditioned ACT."""
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'nero_act_transfer_resume_20261021'))
import checked_runtime  # establishes the same sealed simulation/training import chain
from simulation.collect import collect_episode, verify_replay, verify_action_contract
from training.legacy_visual_data import derive_episode, training_profile, sha

SOURCE = ROOT.parent/'nero_motion_request_20261022/existing_calibration_transfer_01'


def main():
    scene = json.loads((SOURCE/'current_scene_full/scene_configuration.json').read_text())
    old = json.loads((SOURCE/'reference_layout_control/scene_configuration.json').read_text())
    # Complete physical labels use the established uniform4mm/s demonstration timing.
    scene.update(joint_speed_limit_rad_s=.03, teacher_command_lead_limit_rad=.016,
                 gripper_speed_limit_m_s=.004, teacher_gripper_feedback_lead_m=.008)
    scene['dwell_seconds'].update(close=8, release=8)
    # Split before collecting. Validation positions are never training/statistics inputs.
    specs = [
        ('current', 1., [0., 0.], [0., 0.], 'train'),
        ('bridge_1', .33, [0., 0.], [0., 0.], 'train'),
        ('bridge_2', .67, [0., 0.], [0., 0.], 'train'),
        ('source_left', 1., [-.015, 0.], [.005, -.01], 'train'),
        ('source_right', 1., [.015, 0.], [-.005, .01], 'train'),
        ('source_back', 1., [0., -.015], [.01, .005], 'train'),
        ('source_front', 1., [0., .015], [-.01, -.005], 'train'),
        ('target_shift', 1., [.01, -.01], [.015, -.015], 'train'),
        ('heldout_1', 1., [.007, .008], [-.007, .006], 'val'),
        ('heldout_2', 1., [-.008, -.006], [.006, -.007], 'val'),
        ('heldout_bridge', .5, [.004, -.004], [-.003, .004], 'val'),
    ]
    configs = []
    for index, (name, alpha, pick_delta, place_delta, split) in enumerate(specs):
        config = copy.deepcopy(scene)
        for key, delta in [('pick_xy_m', pick_delta), ('place_xy_m', place_delta)]:
            config[key] = ((1-alpha)*np.array(old[key])+alpha*np.array(scene[key])+delta).tolist()
        config['visual_marker_xy_m'] = config['place_xy_m']
        config['generalization'] = dict(index=1000+index, split=split,
            source_group='position_adaptation_'+name, seed=222610+index,
            role='development_targeted' if split == 'train' else 'nearby_unseen_layout',
            additional_manual_measurements=False)
        configs.append(config)
    output = ROOT/'data'; output.mkdir(exist_ok=True)
    spec_path = output/'scene_protocol.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != configs:
            raise ValueError('Frozen scene protocol changed')
    else:
        spec_path.write_text(json.dumps(configs, indent=2)+'\n')
    profile = training_profile(scene)
    rows = []
    for index, config in enumerate(configs):
        started = time.monotonic()
        physical = output/'physical'/f'episode_{1000+index}.hdf5'
        visual = output/'visual'/physical.name
        physical.parent.mkdir(exist_ok=True)
        if not physical.exists():
            collect_episode(config, physical)
        result = json.loads(physical.with_suffix('.result.json').read_text())
        if not result['success'] or not result['full_task_verified']:
            raise ValueError('Failed physical teacher cannot become supervision')
        replay = verify_replay(physical)
        contract = verify_action_contract(physical)
        if visual.exists():
            raise FileExistsError('Do not silently reuse partially derived visual data: '+str(visual))
        report = derive_episode(physical, visual, expected_sha=sha(physical), profile=profile)
        row = dict(report, id=1000+index, status='valid', full_task_verified=True,
                   split=config['generalization']['split'],
                   source_group=config['generalization']['source_group'],
                   physical_sha256=sha(physical), replay=replay,
                   prefix_contract_audit=contract,
                   elapsed_s=time.monotonic()-started)
        rows.append(row)
        (output/'manifest.json').write_text(json.dumps(dict(complete=len(rows)==len(configs),
            hardware_deployment_allowed=False, episodes=rows), indent=2)+'\n')
        print(json.dumps(dict(episode=row['id'], split=row['split'], frames=result['frames'],
            valid_visual=report['valid'], elapsed_s=row['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
