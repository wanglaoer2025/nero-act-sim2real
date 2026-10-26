"""Re-derive all training/independent images with the same current-frame observer."""
import hashlib
import json
from pathlib import Path
import sys

import h5py

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nero_act_new_handeye_policy_repair_20261023'
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from training import legacy_visual_data as visual
from training.visual_dataset import audit_visual_episode,validate_visual_groups
from card_fragments import CardFragmentsEstimator


def main():
    output=ROOT/'data'
    output.mkdir(exist_ok=False)
    old=json.loads((PREVIOUS/'data/manifest.json').read_text())
    collected=json.loads((ROOT/'physical_manifest.json').read_text())
    if not old['complete'] or not collected['complete']:
        raise ValueError('Audited physical collection must be complete')
    rows=[dict(row,path=row['physical_path'],sha256=row['physical_sha256'])
          for row in old['episodes'] if row['split']=='train']+collected['episodes']
    visual.NominalImageEstimator=CardFragmentsEstimator
    code_files=[Path(visual.__file__),ROOT/'card_fragments.py',Path(__file__)]
    code_hashes={str(p):visual.sha(p) for p in code_files}
    code_digest=hashlib.sha256(json.dumps(code_hashes,sort_keys=True).encode()).hexdigest()
    manifest=dict(complete=False,episodes=[],source_sha256=dict(code_hashes),
        failed_ACT_actions_used_as_labels=False,hardware_deployment_allowed=False,
        development_case_ids=[9005,9006],independent_case_ids=[9200,9201],
        observer='current_RGB_nominal_card_fragments_v1')
    for p in [ROOT/'physical_manifest.json',ROOT/'physical_protocol.json',PREVIOUS/'data/manifest.json']:
        manifest['source_sha256'][str(p)]=visual.sha(p)
    for row in rows:
        source=Path(row['path'])
        if visual.sha(source)!=row['sha256']:
            raise ValueError('Physical source changed')
        with h5py.File(source) as f:
            config=json.loads(f.attrs['scene_config'])
        report=visual.derive_episode(source,output/'visual'/source.name,
            expected_sha=row['sha256'],profile=visual.training_profile(config))
        previous=config.get('initial_simulation_snapshot',{}).get('previous_target')
        initial=previous[7] if previous is not None else config['initial_robot_state']['gripper_total_width_m']
        with h5py.File(report['path'],'r+') as f:
            f.attrs['initial_previous_gripper_target_m']=float(initial)
            f.attrs['initial_command_source']='saved_servo_previous_target' if previous is not None else 'explicit_simulation_initialization'
            f.attrs['observation_code_sha256']=code_digest
            f.attrs['observation_code_components']=json.dumps(code_hashes,sort_keys=True)
            f.attrs['target_image_observer']='current_RGB_nominal_card_fragments_v1'
        audit_visual_episode(report['path'])
        report['sha256']=visual.sha(report['path'])
        manifest['episodes'].append(dict(report,id=row['id'],role=row['role'],split=row['split'],
            source_group=config['generalization']['source_group'],physical_path=str(source),
            physical_sha256=row['sha256'],full_task_verified=row['full_task_verified'],
            remaining_task_physical_success=row.get('physical_success',True)))
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(dict(id=row['id'],role=row['role'],split=row['split'],
            valid=report['valid'],stages=report['stages_valid'])),flush=True)
    train=[Path(r['path']) for r in manifest['episodes'] if r['split']=='train']
    val=[Path(r['path']) for r in manifest['episodes'] if r['split']=='val']
    assert len(train)==11 and len(val)==2
    manifest['split_audit']=validate_visual_groups(train,val)
    manifest['complete']=True
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
