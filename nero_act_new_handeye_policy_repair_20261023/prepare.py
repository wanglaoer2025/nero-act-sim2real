"""New-camera complete physical labels with exact saved-state recovery; offline only."""
import json
from pathlib import Path
import sys
import time

import h5py

from protocol import plan_scenes

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'nero_act_new_calibration_20261023'
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from simulation.collect import collect_episode,verify_replay,verify_action_contract
from training.legacy_visual_data import derive_episode,training_profile,sha
from training.visual_dataset import audit_visual_episode,validate_visual_groups


def main():
    output = ROOT/'data'
    output.mkdir(exist_ok=False)
    source_scene = SOURCE/'scene_relocalized_after_return.json'
    source_snapshots = SOURCE/'after_return_validation/hypothetical_95mm_start/physical_snapshots.json'
    scene = json.loads(source_scene.read_text())
    snapshots = json.loads(source_snapshots.read_text())['snapshots']
    specs = plan_scenes(scene,snapshots)
    (output/'protocol.json').write_text(json.dumps(specs,indent=2)+'\n')
    manifest = dict(complete=False,episodes=[],rejected=[],hardware_deployment_allowed=False,
        failed_ACT_actions_used_as_labels=False,
        source_sha256={str(p):sha(p) for p in [source_scene,source_snapshots,Path(__file__),ROOT/'protocol.py']})
    for row in specs:
        started = time.monotonic()
        physical = output/'physical'/f'episode_{row["id"]}.hdf5'
        physical.parent.mkdir(exist_ok=True)
        config = row['scene']
        try:
            result = collect_episode(config,physical)
            replay,contract = verify_replay(physical),verify_action_contract(physical)
            if not result['full_task_verified'] or not replay['passed'] or not contract['passed']:
                raise ValueError('Only full verified physical trajectories are labels')
            report = derive_episode(physical,output/'visual'/physical.name,
                                   expected_sha=sha(physical),profile=training_profile(config))
            # The causal initializer is the saved submitted target, not the measured width.
            previous = config.get('initial_simulation_snapshot',{}).get('previous_target')
            initial_command = previous[7] if previous is not None else config['initial_robot_state']['gripper_total_width_m']
            with h5py.File(report['path'],'r+') as f:
                f.attrs['initial_previous_gripper_target_m'] = float(initial_command)
                f.attrs['initial_command_source'] = 'saved_servo_previous_target' if previous is not None else 'explicit_simulation_initialization'
            audit_visual_episode(report['path'])
            report['sha256'] = sha(report['path'])
            manifest['episodes'].append(dict(report,id=row['id'],role=row['role'],split=row['split'],
                source_group=row['source_group'],physical_path=str(physical),physical_sha256=sha(physical),
                replay=replay,prefix_contract_audit=contract,full_task_verified=True,
                exact_dynamic_snapshot_preserved=row['role']=='recovery'))
            print(json.dumps(dict(id=row['id'],role=row['role'],split=row['split'],accepted=True,
                visual_valid=report['valid'],elapsed_s=time.monotonic()-started)),flush=True)
        except (ValueError,RuntimeError) as exc:
            manifest['rejected'].append(dict(id=row['id'],role=row['role'],split=row['split'],error=str(exc)))
            print(json.dumps(manifest['rejected'][-1]),flush=True)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    train = [Path(r['path']) for r in manifest['episodes'] if r['split']=='train']
    val = [Path(r['path']) for r in manifest['episodes'] if r['split']=='val']
    if len(train)<7 or len(val)!=2 or not any(r['role']=='center' for r in manifest['episodes']):
        raise ValueError('At least7 full training scenes including center and both independent validation scenes required')
    manifest['split_audit'] = validate_visual_groups(train,val)
    manifest['complete'] = True
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
