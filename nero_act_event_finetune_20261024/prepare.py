"""Generate event-terminated complete physical experts; offline and append-only."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'nero_act_grouped_gripper_20261021'
sys.path.insert(0,str(BASE))
import simulation.collect as collector
from training.legacy_visual_data import derive_episode,training_profile
from training.visual_dataset import audit_visual_episode
from teacher_event import run_teacher


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output=ROOT/'data';output.mkdir(exist_ok=False)
    source=ROOT.parent/'nero_act_new_handeye_policy_repair_20261023/data/protocol.json'
    protocol=[copy.deepcopy(r) for r in json.loads(source.read_text()) if r['role']!='recovery' and r['split']=='train']
    assert len(protocol)==5
    for row in protocol:
        row['scene']['generalization']['source_group']='event_train_'+str(row['id'])
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    hashes={str(p.resolve()):sha(p) for p in [source,Path(__file__),ROOT/'teacher_event.py',ROOT/'event_gate.py']}
    manifest=dict(complete=False,episodes=[],rejected=[],source_sha256=hashes,
        failed_ACT_actions_used_as_labels=False,hardware_commands_sent=False,hardware_deployment_allowed=False)
    collector.run_teacher=run_teacher
    for row in protocol:
        started=time.monotonic();physical=output/'physical'/f'episode_{row["id"]}.hdf5'
        physical.parent.mkdir(exist_ok=True)
        try:
            result=collector.collect_episode(row['scene'],physical)
            replay=collector.verify_replay(physical);contract=collector.verify_action_contract(physical)
            if not result['full_task_verified'] or not replay['passed'] or not contract['passed']:
                raise ValueError('Complete expert, exact replay and original prefix contract required')
            report=derive_episode(physical,output/'visual'/physical.name,
                expected_sha=sha(physical),profile=training_profile(row['scene']))
            audit_visual_episode(report['path'])
            with h5py.File(physical) as f:
                stages=f['simulation/stage'].asstr()[:]
                dwell={s:float(np.count_nonzero(stages==s)/20) for s in ['close','release']}
            manifest['episodes'].append(dict(report,id=row['id'],role=row['role'],split='train',
                physical_path=str(physical),physical_sha256=sha(physical),replay=replay,
                prefix_contract_audit=contract,full_task_verified=True,contact_stage_seconds=dwell))
            print(json.dumps(dict(id=row['id'],accepted=True,contact_stage_seconds=dwell,
                frames=result['frames'],elapsed_s=time.monotonic()-started)),flush=True)
        except (ValueError,RuntimeError) as exc:
            manifest['rejected'].append(dict(id=row['id'],error=str(exc)))
            print(json.dumps(manifest['rejected'][-1]),flush=True)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if manifest['rejected']:raise RuntimeError('Physical/visual mechanism failed; no training')
    manifest['complete']=True
    # Bind actual shared simulation/observation code and rigid assets, not just the wrapper.
    for directory in [BASE/'simulation',BASE/'training',BASE/'executor']:
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix in {'.py','.xml','.stl','.obj','.msh'}:
                hashes[str(p.resolve())]=sha(p)
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Preparation source changed')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
