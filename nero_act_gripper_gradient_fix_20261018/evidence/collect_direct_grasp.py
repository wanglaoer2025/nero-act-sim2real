"""Validated offline closing supervision from actual frozen ACT grasp states."""
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from executor.evaluation_integrity import verify_frozen_artifacts
from simulation.collect import collect_episode, verify_replay
from simulation.refresh_observations import derive_episode, require_fresh_inputs
from simulation.scene import build_scene
from training.recoveries import validate_corrections

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'nero_act_cvae_training_fix_20261018'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = ROOT/'training_data/direct_grasp_raw_v1'
    fresh = ROOT/'training_data/direct_grasp_fresh_v1'
    raw.mkdir(parents=True, exist_ok=False)
    fresh.mkdir(exist_ok=False)
    for index, evaluation, case, split in [
        (4001, 'run21_nominal', 'cup_+0_+0', 'train'),
        (4002, 'run21_validation42', 'validation42_grasp', 'val')]:
        source = OLD/'evaluation'/evaluation
        origin = source/case
        protocol = json.loads((source/'protocol.json').read_text())
        checkpoint = OLD/'model/run21_cvae_posterior/policy.ckpt'
        verify_frozen_artifacts(checkpoint, protocol, source_root=OLD)
        scene = json.loads((origin/'scene_configuration.json').read_text())
        record = json.loads((origin/'physical_snapshots.json').read_text())
        if split == 'train':
            model = build_scene(scene)
            data = mujoco.MjData(model)
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
            heights = []
            for snapshot in record['snapshots']:
                data.qpos[:] = snapshot['qpos']
                mujoco.mj_forward(model, data)
                heights.append(float(data.site_xpos[site, 2]))
            j = int(np.argmin(abs(np.array(heights)-.055)))
            assert abs(heights[j]-.055) < .002
        else:
            events = json.loads((origin/'result.json').read_text())['simulated_events']
            step = next(e['physics_step'] for e in events if e['kind'] == 'GRASP')
            j = record['query_steps'].index(step)
        snapshot = record['snapshots'][j]
        assert snapshot['qpos'][7]-snapshot['qpos'][8] > .094
        config = copy.deepcopy(scene)
        config.update(continuation='grasp_to_retreat', initial_simulation_snapshot=snapshot,
            initial_robot_state=dict(qpos_rad=snapshot['qpos'][:7],
                gripper_total_width_m=snapshot['qpos'][7]-snapshot['qpos'][8]),
            joint_speed_limit_rad_s=.02, teacher_command_lead_limit_rad=.015,
            gripper_speed_limit_m_s=.004, teacher_gripper_feedback_lead_m=.008,
            correction_source=dict(origin=str(origin), query_step=record['query_steps'][j],
                split=split, source_group='run21_direct_grasp_'+case, role='partial_not_full',
                reason='physically verified closure at actual ACT grasp pose; no alignment takeover'))
        config['dwell_seconds'].update(close=8, release=8)
        path = raw/f'episode_{index}.hdf5'
        manifest = dict(complete=False, rejected=[], episodes=[], source_evaluation=str(source),
            source_protocol_sha256=sha(source/'protocol.json'),
            source_checkpoint_sha256=protocol['checkpoint_sha256'])
        try:
            path.with_suffix('.scene.json').write_text(json.dumps(config, indent=2))
            result = collect_episode(config, path)
            replay = verify_replay(path)
            assert result['success'] and result['physical_success'] and replay['passed']
            assert not result['full_task_verified'] and result['max_lift_m'] > .06
            assert result['maximum_horizontal_transfer_m'] > .1
            row = dict(id=index, path=str(path), sha256=sha(path), status='valid', split=split,
                source_group=config['correction_source']['source_group'], teacher_success=True,
                full_task_verified=False, correction_verified=True, task='grasp_to_retreat_correction',
                origin=str(origin), query_step=record['query_steps'][j],
                snapshot_sha256=sha(origin/'physical_snapshots.json'),
                scene_sha256=sha(origin/'scene_configuration.json'), replay=replay,
                prefix_contract_audit=result['prefix_contract_audit'])
            derived = derive_episode(path, fresh/path.name, expected_source_sha256=row['sha256'])
            row.update(path=derived['target'], sha256=derived['target_sha256'], observation_derivation=derived)
            manifest.update(complete=True, episodes=[row])
            validate_corrections(manifest, [])
            require_fresh_inputs([Path(row['path'])])
            verify_frozen_artifacts(checkpoint, protocol, source_root=OLD)
            (fresh/f'{split}_manifest.json').write_text(json.dumps(manifest, indent=2))
            print(json.dumps(dict(id=index, split=split, success=True, lift_m=result['max_lift_m'],
                transfer_m=result['maximum_horizontal_transfer_m'], replay_error=replay['maximum_qpos_error'])), flush=True)
        except Exception as exc:
            manifest['rejected'].append(dict(error=str(exc)))
            manifest['complete'] = False
            (fresh/f'{split}_manifest.failed.json').write_text(json.dumps(manifest, indent=2))
            raise


if __name__ == '__main__':
    main()
