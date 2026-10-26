"""Second, separately bounded1000-update repair; preserve the existing70-tensor scope."""
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nero_act_new_handeye_policy_repair_20261023'
_spec=importlib.util.spec_from_file_location('_new_camera_training_base',PREVIOUS/'train_adaptation.py')
base=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=base
_spec.loader.exec_module(base)
from contact_batches import contact_batches,STAGES

base.ROOT=ROOT
base.INITIALIZER=PREVIOUS/'training/pilot_A/policy_last.ckpt'
base.INITIALIZER_SHA='1a28f4fddd3095791403028b639b99f9775b91caab45c950c2d0f518955bc0f0'
base.PROFILE='new_handeye_contact_recovery_v2'
historical=base.historical


def prepare_dataset(policy,stats):
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    if not manifest['complete'] or manifest['failed_ACT_actions_used_as_labels'] is not False:
        raise ValueError('Completed audited physical labels required')
    rows=manifest['episodes']
    train=[r for r in rows if r['split']=='train']
    val=[r for r in rows if r['split']=='val']
    if {r['id'] for r in val}!={9200,9201} or any(r['id'] in {9005,9006} for r in rows):
        raise ValueError('Development failures cannot become independent validation')
    audit=base.validate_visual_groups([Path(r['path']) for r in train],[Path(r['path']) for r in val])
    historical.require_unchanged(manifest['source_sha256'])
    historical.require_unchanged({r[k]:r[h] for r in rows for k,h in [('path','sha256'),('physical_path','physical_sha256')]})
    samples=[]
    pools={f'{kind}:{stage}':[] for kind in ['center','other'] for stage in STAGES}
    pools.update({'contact:close':[],'contact:lift':[]})
    for i,row in enumerate(train):
        base.audit_visual_episode(row['path'])
        if not row['full_task_verified'] and not row['remaining_task_physical_success']:
            raise ValueError('A partial recovery must physically finish its remaining task')
        with h5py.File(row['path']) as f:
            valid=np.flatnonzero(f['observations/visual_valid'][:])
            stages=f['simulation/stage'].asstr()[:]
        for t in valid:
            stage=stages[t]
            if stage not in STAGES:
                continue
            if row['role'] in {'grasp','lift'}:
                if stage not in {'close','lift'}:
                    continue
                key=f'contact:{stage}'
            else:
                key=f'{"center" if row["role"]=="center" else "other"}:{stage}'
            pools[key].append(len(samples))
            samples.append((i,int(t)))
    dataset=base.VisualReferenceDataset([Path(r['path']) for r in train],['camera_0'],stats,
        policy_config=policy.config,image_size=tuple(policy.config['image_size']),sample_starts=samples)
    batches=contact_batches(pools)
    (ROOT/'sampling_audit.json').write_text(json.dumps(dict(audit,training_episodes=len(train),
        independent_validation_episodes=len(val),validation_rows_in_training=0,samples=len(samples),
        pool_sizes={k:len(v) for k,v in pools.items()},updates=1000,batch_size=32,
        exact_contact_batch_fraction=.5,all_original_stages_in_every_batch=True,
        statistics_recomputed=False),indent=2)+'\n')
    return dataset,batches,rows


base.prepare_dataset=prepare_dataset
load_candidate=base.load_candidate


if __name__=='__main__':
    base.main()
