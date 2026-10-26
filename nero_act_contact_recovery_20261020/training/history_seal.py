"""Freeze reviewed code, untouched numeric16 and shared A/B schedules before training."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import torch

from training.history import prepare_batch_orders,freeze_history_protocol
from training.integrity import require_unchanged,snapshot
from training.rebase_initializer import paired_config

ROOT=Path(__file__).resolve().parents[1]


def runtime_sources(root):
    return sorted(p for folder in ['training','executor','simulation','calibration','vision','policy_source']
                  for p in (Path(root)/folder).rglob('*.py'))


def seal_history_experiment(data_dir:Path,initializer:Path,verification_path:Path)->dict:
    verification_path=Path(verification_path);verification=json.loads(verification_path.read_text())
    if (verification.get('complete') is not True or verification.get('failed')!=0
            or not verification.get('source_sha256') or verification.get('tests_passed',0)<=0):
        raise ValueError('Complete green suite with exact tested sources required')
    require_unchanged(verification['source_sha256'])
    if not set(map(lambda p:str(p.resolve()),runtime_sources(ROOT)))<=verification['source_sha256'].keys():
        raise ValueError('Suite evidence omits runtime source files')
    destination=ROOT/'evidence/pre_optimization_freeze.json'
    if destination.exists():raise FileExistsError(destination)
    data_dir,initializer=Path(data_dir).resolve(),Path(initializer).resolve()
    if hashlib.sha256(initializer.read_bytes()).hexdigest()!='b252dd578b4bf5626566ec5b60a47be0bdcfee30e9eab990cde05ac122bdea71':
        raise ValueError('Different initialization requires a new declared experiment')
    accepted=json.loads((data_dir/'data_acceptance.json').read_text());manifest=json.loads((data_dir/'manifest.json').read_text())
    settings=json.loads((ROOT/'config/history_experiment.json').read_text())
    additional=settings.get('additional_failure_episode_ids',[])
    if (not accepted.get('complete') or not manifest.get('complete') or len(manifest.get('episodes',[]))!=66+len(additional)
            or accepted['manifest_sha256']!=hashlib.sha256((data_dir/'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('Complete verified66 physical history episodes required')
    for row in manifest['episodes']:
        if snapshot([row['path']])[str(Path(row['path']).resolve())]!=row['sha256']:raise ValueError('Prepared history data changed')
    runtime=json.loads((ROOT/'config/history_runtime_contract.json').read_text())
    protocol=ROOT/'evaluation/history_protocol.json'
    if not protocol.exists():freeze_history_protocol(ROOT/'evaluation/protocol.json',runtime,protocol)
    else:
        frozen=json.loads(protocol.read_text())
        if (frozen.get('history_runtime_contract')!=runtime
                or frozen.get('origin_protocol_sha256')!=hashlib.sha256((ROOT/'evaluation/protocol.json').read_bytes()).hexdigest()):
            raise ValueError('Existing numeric protocol differs from declared source/runtime')
    source=torch.load(initializer,map_location='cpu',weights_only=False)
    stats=pickle.loads((data_dir/'dataset_stats.pkl').read_bytes())
    config=paired_config(dict(source['config'],image_size=[180,320]),stats)
    config['transition_sampling_profile']=settings.get('transition_sampling_profile','none')
    order_file=data_dir/'batch_order_manifest.json'
    if not order_file.exists():prepare_batch_orders(data_dir,config)
    pipeline=json.loads((ROOT/'evidence/pipeline_status.json').read_text())
    recovery_manifest=ROOT/pipeline['recovery_batch']/'manifest.json'
    if hashlib.sha256(recovery_manifest.read_bytes()).hexdigest()!=manifest['recovery_manifest_sha256']:
        raise ValueError('The accepted recovery batch changed')
    recovery=json.loads(recovery_manifest.read_text())
    data_files=[Path(row['path']) for row in manifest['episodes']]
    data_files += [p for p in data_dir.iterdir() if p.is_file() and p.suffix in ('.json','.pkl','.npy')]
    data_files += [recovery_manifest,initializer,initializer.parent/'dataset_stats.pkl',verification_path]
    data_files += [Path(r[k]) for r in recovery['episodes'] for k in ['path','context_path']]
    data_files += [Path(r['path']) for r in recovery['base_episodes']]
    original_capture=ROOT/'results/original_failure_capture'
    data_files += [original_capture/name for name in ['runtime_checkpoints.json','scene_configuration.json','result.json','reference_verification.json']]
    if recovery.get('verified_parent_batch'):data_files.append(Path(recovery['verified_parent_batch']))
    source_files=runtime_sources(ROOT)+[p for p in (ROOT/'simulation/assets').rglob('*') if p.is_file()]
    source_files += [ROOT/'simulation/scene_template.xml',ROOT/'evaluation/protocol.json',protocol]
    source_files += list((ROOT/'config').glob('*.json'))+list((ROOT/'scene').glob('*.json'))
    source_files += [ROOT/'evidence/baseline_development_protocol.json',ROOT/'evidence/retained_calibration_sources.json',
                     ROOT/'evidence/complete_recovery_v3.py']
    original=json.loads((ROOT/'evidence/source_inventory.json').read_text());require_unchanged(original)
    result=dict(schema='history_pre_optimization_v1',complete=True,data_dir=str(data_dir),
        source_sha256=snapshot(sorted(set(source_files))),data_sha256=snapshot(sorted(set(data_files))),
        original_source_files_unchanged=len(original),created_unix_s=time.time(),
        independent16_evaluated=False,optimizer_steps_before_seal=0,hardware_commands_sent=False)
    require_unchanged(result['source_sha256']);require_unchanged(result['data_sha256'])
    temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(destination)
    print(json.dumps(dict(sealed=True,source_files=len(result['source_sha256']),data_files=len(result['data_sha256']),
                         optimizer_steps_before_seal=0)),flush=True)
    return result


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir','initializer','verification']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();seal_history_experiment(a.data_dir,a.initializer,a.verification);return 0


if __name__=='__main__':raise SystemExit(main())
