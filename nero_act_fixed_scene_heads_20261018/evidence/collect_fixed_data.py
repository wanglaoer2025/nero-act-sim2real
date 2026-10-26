"""One declared offline fixed-scene dataset; physical failures are retained and stop collection."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation.collect import collect_episode, verify_replay
from simulation.refresh_observations import derive_episode, require_fresh_inputs
from training.recoveries import validate_recovery_scene

SOURCE = ROOT.parent/'nero_act_gripper_contact_focus_20261018'
DATA = ROOT/'training_data/fixed_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    DATA.mkdir(exist_ok=False)
    center = json.loads((SOURCE/'model/run26_current_fixed_scene/scene_config.json').read_text())
    (DATA/'scene_config.json').write_text(json.dumps(center, indent=2))
    declaration = dict(full_domains=['nominal', 'delay_100ms', 'delay_500ms', 'force'],
        variants_per_domain=3, train_variants=[0, 1], validation_variants=[2],
        fixed_object_positions=True, initial_joint_jitter_rad=.001,
        initial_joint_seeds=list(range(270000,270012)), partial_roles=['release_to_retreat'],
        partial_origins=['force_0', 'delay_100ms'], physical_limits_unchanged=True,
        no_hardware=True, no_automatic_retries=True)
    (DATA/'predeclared.json').write_text(json.dumps(declaration, indent=2))
    manifest = dict(complete=False, episodes=[], rejected=[], scope='current_fixed_scene',
        source_model=str(SOURCE/'model/run26_current_fixed_scene'), hardware_connected=False)
    def save():
        (DATA/'manifest.json').write_text(json.dumps(manifest, indent=2))
    save()
    specifications = []
    for domain, delay in [('nominal',0), ('delay_100ms',2), ('delay_500ms',10), ('force',0)]:
        for variant in range(3):
            index = 5000+len(specifications)
            config = copy.deepcopy(center)
            config['action_delay_steps'] = delay
            rng = np.random.default_rng(270000+index-5000)
            config['initial_robot_state']['qpos_rad'] = (
                np.array(center['initial_robot_state']['qpos_rad'])+rng.uniform(-.001,.001,7)).tolist()
            if domain == 'force':
                config['arm_disturbance'] = dict(seed=270000+index, max_torque_nm=.15,
                    period_steps=47, duration_steps=3, start_step=90)
            specifications.append((index, 'val' if variant == 2 else 'train', domain, config, None))
    for index, split, name in [(5100,'train','force_0'), (5101,'val','delay_100ms')]:
        origin = SOURCE/'evidence/run26_current_scene14'/name
        record = json.loads((origin/'physical_snapshots.json').read_text())
        config = json.loads((origin/'scene_configuration.json').read_text())
        snapshot = record['snapshots'][-1]
        query_step = record['query_steps'][-1]
        config.update(continuation='release_to_retreat', initial_simulation_snapshot=snapshot,
            initial_robot_state=dict(qpos_rad=snapshot['qpos'][:7],
                gripper_total_width_m=snapshot['qpos'][7]-snapshot['qpos'][8]),
            joint_speed_limit_rad_s=.02, teacher_command_lead_limit_rad=.015,
            gripper_speed_limit_m_s=.004, teacher_gripper_feedback_lead_m=.008,
            dwell_seconds=dict(config['dwell_seconds'], close=8, release=8),
            correction_source=dict(origin=str(origin), query_step=query_step, split=split,
                source_group=f'fixed_release_{name}', role='partial_not_full'))
        validate_recovery_scene(config,json.loads((origin/'scene_configuration.json').read_text()),snapshot)
        provenance=dict(origin=str(origin), query_step=query_step,
            snapshot_sha256=sha(origin/'physical_snapshots.json'),
            scene_sha256=sha(origin/'scene_configuration.json'),
            protocol_sha256=sha(origin.parent/'protocol.json'))
        specifications.append((index, split, 'release_recovery', config, provenance))
    for index, split, domain, config, provenance in specifications:
        raw = DATA/'raw'/f'episode_{index}.hdf5'
        raw.parent.mkdir(exist_ok=True)
        try:
            result = collect_episode(config, raw)
            assert result['success'] and result['prefix_contract_audit']['passed']
            replay = verify_replay(raw)
            assert replay['passed']
            derived = DATA/f'episode_{index}.hdf5'
            proof = derive_episode(raw, derived, expected_source_sha256=sha(raw))
            assert proof['physical_arrays_and_original_attributes_identical']
            manifest['episodes'].append(dict(id=index, path=str(derived), split=split,
                source_group=f'fixed_{index}', domain=domain, sha256=sha(derived),
                raw_path=str(raw), raw_sha256=sha(raw), task=result['task'],
                full_task_verified=result.get('full_task_verified',False),
                teacher_success=True, replay=replay,
                prefix_contract_audit=result['prefix_contract_audit'],
                provenance=provenance, frames=result['frames']))
            save()
            print(json.dumps(dict(id=index, split=split, domain=domain,
                success=True, frames=result['frames'])),flush=True)
        except BaseException as exc:
            manifest['rejected'].append(dict(id=index, split=split, domain=domain, error=str(exc)))
            save()
            raise
    paths = [Path(row['path']) for row in manifest['episodes']]
    require_fresh_inputs(paths)
    assert len(paths)==14 and sum(row['full_task_verified'] for row in manifest['episodes'])==12
    assert {r['source_group'] for r in manifest['episodes'] if r['split']=='train'}.isdisjoint(
        {r['source_group'] for r in manifest['episodes'] if r['split']=='val'})
    manifest['complete']=True
    save()
    (DATA/'split.json').write_text(json.dumps({key:[r['id'] for r in manifest['episodes']
        if r['split']==key] for key in ['train','val']},indent=2))


if __name__ == '__main__':
    main()
