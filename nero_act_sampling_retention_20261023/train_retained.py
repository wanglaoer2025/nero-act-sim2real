"""Repeat the sealed bounded trainer with the omitted original corrections restored."""
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np

from retention import STAGES,make_batches

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nero_act_full_flow_repair_20261023'
sys.path.insert(0,str(PREVIOUS))
spec=importlib.util.spec_from_file_location('sealed_full_flow_trainer',PREVIOUS/'train_repair.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
from training.visual_dataset import VisualReferenceDataset,audit_visual_episode


def prepare_data(policy,stats):
    old,_,validation,scene,files=trainer.prior.prepare_data(policy,stats)
    old_manifest_path=trainer.prior.ROOT/'data/manifest.json'
    old_manifest=json.loads(old_manifest_path.read_text())
    original={Path(r['path']).resolve():r for r in old_manifest['episodes'] if r['split']=='train'}
    new_manifest_path=PREVIOUS/'data/manifest.json';manifest=json.loads(new_manifest_path.read_text())
    if not manifest['complete'] or manifest['failed_ACT_actions_used_as_labels'] is not False:
        raise ValueError('Audited new physical corrections required')
    rows=manifest['episodes'];paths=list(old.paths);new_start=len(paths)
    trainer.require_unchanged({r['path']:r['sha256'] for r in rows})
    trainer.require_unchanged({r['physical_path']:r['physical_sha256'] for r in rows})
    for row in rows:
        if row['split']!='train' or not row['replay']['passed'] or not row['prefix_contract_audit']['passed']:
            raise ValueError('Invalid or validation episode in optimization')
        audit_visual_episode(row['path']);paths.append(Path(row['path']))
    pools={key:[] for key in ['new_correction','retained_align','retained_clearance']+
        [f'{kind}:{stage}' for stage in STAGES for kind in ['new','old']]}
    cache={}
    for i,path in enumerate(paths):
        with h5py.File(path) as f:
            if f.attrs['split']!='train':raise ValueError('Validation rows may not enter training')
            cache[i]=(f['simulation/stage'].asstr()[:],np.flatnonzero(f['observations/visual_valid'][:]))
    samples=[];retained_paths=set()
    for i,t in old.sample_starts:
        path=paths[i].resolve();stage=cache[i][0][t]
        if path in original:
            key='retained_clearance' if original[path]['role']=='clearance' else 'retained_align'
            retained_paths.add(path)
        elif stage in STAGES:key='old:'+stage
        else:continue
        index=len(samples);samples.append((i,t));pools[key].append(index)
    if retained_paths!=set(original):raise ValueError('An original training intervention was still omitted')
    for offset,row in enumerate(rows):
        i=new_start+offset;stages,valid=cache[i]
        for t in valid:
            if row['role']!='correction' and stages[t] not in STAGES:continue
            key='new_correction' if row['role']=='correction' else 'new:'+stages[t]
            index=len(samples);samples.append((i,int(t)));pools[key].append(index)
    dataset=VisualReferenceDataset(paths,['camera_0'],stats,policy_config=policy.config,
        image_size=tuple(policy.config['image_size']),sample_starts=samples)
    batches=make_batches(pools,steps=1000,seed=231019)
    sampled_paths={paths[samples[i][0]].resolve() for i in np.unique(batches)}
    if not set(original)<=sampled_paths:raise ValueError('Loaded original interventions never sampled')
    (ROOT/'sampling_audit.json').write_text(json.dumps(dict(pool_sizes={k:len(v) for k,v in pools.items()},
        retained_original_training_episodes=len(original),actually_sampled_original_episodes=len(set(original)&sampled_paths),
        original_validation_episodes_optimized=0,batch_size=32,updates=1000,
        role_draws={k:int(np.isin(batches,v).sum()) for k,v in pools.items()}),indent=2)+'\n')
    files+=[old_manifest_path,new_manifest_path,PREVIOUS/'data/protocol.json',ROOT/'current_scene.json',PREVIOUS/'train_repair.py']
    files += [Path(row[key]) for row in rows for key in ['path','physical_path']]
    return dataset,batches,validation,scene,files


trainer.ROOT=ROOT
trainer.prepare_data=prepare_data

if __name__=='__main__':trainer.main()
