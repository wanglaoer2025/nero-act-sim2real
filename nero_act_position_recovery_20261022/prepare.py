"""Correct development-only ACT-visited states with the physical offline teacher."""
import copy
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
FIRST = ROOT.parent/'nero_act_position_adaptation_20261022'
sys.path.insert(0, str(ROOT.parent/'nero_act_transfer_resume_20261021'))
import checked_runtime
from simulation.collect import collect_episode, verify_replay, verify_action_contract
from training.legacy_visual_data import derive_episode, training_profile, sha
from training.visual_dataset import audit_visual_episode


def main():
    output = ROOT/'data'; output.mkdir(exist_ok=False)
    original = json.loads((FIRST/'data/manifest.json').read_text())
    rows = []
    physical_proofs = []
    for row in original['episodes']:
        row = dict(row, physical_path=str(FIRST/'data/physical'/f'episode_{row["id"]}.hdf5'))
        rows.append(row)
        physical_proofs.append(dict(id=row['id'], prefix_contract=verify_action_contract(row['physical_path']), replay=row['replay']))
    specs = []
    for case, partial in [('current_development', False), ('reference_layout', True)]:
        folder = FIRST/'results'/case
        base = json.loads((folder/'scene_configuration.json').read_text())
        snapshots = json.loads((folder/'physical_snapshots.json').read_text())
        measurements = json.loads((folder/'measurements.json').read_text())
        if not partial:
            eligible = [i for i, step in enumerate(snapshots['query_steps'])
                        if step >= 500 and measurements[step]['tcp_xyz'][2] >= .12
                        and not measurements[step]['bilateral']]
            indices = [eligible[i] for i in np.linspace(0, len(eligible)-1, 4).astype(int)]
        else:
            eligible = [i for i, step in enumerate(snapshots['query_steps'])
                        if measurements[step]['bilateral'] and not measurements[step]['supported']]
            indices = [eligible[-1], eligible[max(0, len(eligible)-101)]]
        for query in indices:
            config = copy.deepcopy(base); saved = snapshots['snapshots'][query]
            eid = 1100+len(specs)
            config.update(initial_simulation_snapshot=saved,
                initial_robot_state=dict(qpos_rad=saved['qpos'][:7],
                    gripper_total_width_m=saved['qpos'][7]-saved['qpos'][8]),
                joint_speed_limit_rad_s=.03, teacher_command_lead_limit_rad=.016,
                gripper_speed_limit_m_s=.004, teacher_gripper_feedback_lead_m=.008)
            config['dwell_seconds'].update(close=8, release=8)
            if partial:
                config['continuation'] = 'place_to_retreat'
            config['generalization'] = dict(index=eid, split='train',
                source_group=f'development_recovery_{eid}', seed=222700+eid,
                parent_case=case, parent_query=query,
                label_source='physical_teacher_not_failed_ACT_action',
                validation_layout_used=False)
            specs.append(config)
    (output/'scene_protocol.json').write_text(json.dumps(specs, indent=2)+'\n')
    for config in specs:
        started = time.monotonic(); eid = config['generalization']['index']
        physical = output/'physical'/f'episode_{eid}.hdf5'; physical.parent.mkdir(exist_ok=True)
        result = collect_episode(config, physical)
        replay = verify_replay(physical); contract = verify_action_contract(physical)
        visual = output/'visual'/physical.name
        report = derive_episode(physical, visual, expected_sha=sha(physical), profile=training_profile(config))
        # Interrupted sources carry an actual submitted command different from measured opening.
        with h5py.File(visual, 'r+') as stream:
            stream.attrs['initial_previous_gripper_target_m'] = config['initial_simulation_snapshot']['previous_target'][7]
            stream.attrs['initial_command_source'] = 'exact_physical_snapshot.previous_target'
            stream.attrs['recovery_export_code_sha256'] = sha(__file__)
        audit_visual_episode(visual)
        report['sha256'] = sha(visual)
        row = dict(report, id=eid, split='train', source_group=config['generalization']['source_group'],
            status='valid', full_task_verified=bool(result['full_task_verified']),
            partial_correction=bool(config.get('continuation')), physical_path=str(physical.resolve()),
            physical_sha256=sha(physical), prefix_contract_audit=contract, replay=replay)
        rows.append(row); physical_proofs.append(dict(id=eid, prefix_contract=contract, replay=replay))
        complete = len(rows) == 17
        (output/'manifest.json').write_text(json.dumps(dict(complete=complete, episodes=rows,
            hardware_deployment_allowed=False, failed_ACT_actions_used_as_labels=False), indent=2)+'\n')
        (ROOT/'physical_audit_progress.json').write_text(json.dumps(dict(complete=complete,
            episodes=physical_proofs, accepted=len(rows)), indent=2)+'\n')
        print(json.dumps(dict(id=eid, teacher_success=result['success'],
            full_task_verified=result['full_task_verified'], partial_correction=row['partial_correction'],
            replay=replay['passed'], elapsed_s=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
