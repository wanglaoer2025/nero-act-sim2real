"""One data-only comparison, same initializer, optimizer, loss and fixed1000 updates."""
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'nero_act_sampling_retention_20261023'
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location('sealed_retention_trainer', PREVIOUS/'train_retained.py')
retained = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retained)
trainer = retained.trainer
from retention import STAGES, make_batches
from training.visual_dataset import VisualReferenceDataset, audit_visual_episode, validate_visual_groups
from protocol import classify_partition


def prepare_data(policy, stats):
    old, _, validation, scene, files = retained.prepare_data(policy, stats)
    original_manifest = json.loads((trainer.prior.ROOT/'data/manifest.json').read_text())
    original = {Path(r['path']).resolve(): r for r in original_manifest['episodes'] if r['split']=='train'}
    current_manifest = json.loads((ROOT.parent/'nero_act_full_flow_repair_20261023/data/manifest.json').read_text())
    current = {Path(r['path']).resolve(): r for r in current_manifest['episodes']}
    manifest_path = ROOT/'data/manifest.json'
    dense = json.loads(manifest_path.read_text())
    if not dense['complete'] or dense['failed_ACT_actions_used_as_labels'] is not False:
        raise ValueError('Completed physical teacher data required')
    rows = dense['episodes']
    trainer.require_unchanged({r['path']: r['sha256'] for r in rows})
    trainer.require_unchanged({r['physical_path']: r['physical_sha256'] for r in rows})
    paths = list(old.paths)
    samples = list(old.sample_starts)
    pools = {key: [] for key in ['new_correction','retained_align','retained_clearance'] +
             [f'{kind}:{stage}' for stage in STAGES for kind in ['new','old']]}
    for index, (i, t) in enumerate(samples):
        path = paths[i].resolve()
        with h5py.File(path) as f:
            stage = f['simulation/stage'].asstr()[t]
        if path in original:
            key = 'retained_clearance' if original[path]['role']=='clearance' else 'retained_align'
        elif path in current:
            key = 'new_correction' if current[path]['role']=='correction' else 'new:'+stage
        else:
            key = 'old:'+stage
        pools[key].append(index)
    for row in rows:
        if (row['split']!='train' or classify_partition(row['offset_xy_m'])!='train'
                or not row['replay']['passed'] or not row['prefix_contract_audit']['passed']):
            raise ValueError('Dense data violates partition or physical replay contract')
        path = Path(row['path'])
        audit_visual_episode(path)
        with h5py.File(path) as f:
            if f.attrs['split']!='train':
                raise ValueError('Held-out episode cannot enter optimization')
            valid = np.flatnonzero(f['observations/visual_valid'][:])
        i = len(paths)
        paths.append(path)
        key = 'retained_clearance' if row['role']=='clearance' else 'retained_align'
        for t in valid:
            pools[key].append(len(samples))
            samples.append((i, int(t)))
    validate_visual_groups(paths, [Path(r['path']) for r in original_manifest['episodes'] if r['split']=='val'])
    batches = make_batches(pools, steps=1000, seed=231019)
    sampled = {paths[samples[index][0]].resolve() for index in np.unique(batches)}
    if not set(original)<=sampled:
        raise ValueError('Original interventions omitted again')
    dense_paths = {Path(r['path']).resolve() for r in rows}
    if not dense_paths<=sampled:
        raise ValueError('Loaded dense episode was never sampled')
    (ROOT/'sampling_audit.json').write_text(json.dumps(dict(
        pool_sizes={k:len(v) for k,v in pools.items()},
        original_episodes_sampled=len(set(original)&sampled),
        dense_episodes_sampled=len(dense_paths&sampled),
        held_out_episodes_optimized=0, batch_size=32, updates=1000,
        role_draws={k:int(np.isin(batches,v).sum()) for k,v in pools.items()}), indent=2)+'\n')
    files += [manifest_path, ROOT/'data/protocol.json', PREVIOUS/'train_retained.py', PREVIOUS/'retention.py']
    files += [Path(r[key]) for r in rows for key in ['path','physical_path']]
    dataset = VisualReferenceDataset(paths, ['camera_0'], stats, policy_config=policy.config,
        image_size=tuple(policy.config['image_size']), sample_starts=samples)
    return dataset, batches, validation, scene, files


retained.ROOT = ROOT
trainer.ROOT = ROOT
trainer.prepare_data = prepare_data

if __name__ == '__main__':
    trainer.main()
