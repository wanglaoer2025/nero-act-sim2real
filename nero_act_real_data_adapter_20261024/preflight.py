"""Verify converted targets and ACT batch compatibility without weight updates."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch

from adapter import FutureStateDataset, file_snapshot, verify_source


def verify_conversion(row):
    source = verify_source(row['path'], numeric=True)
    if source != Path(row['source_path']).resolve():
        raise ValueError('Conversion source differs from manifest')
    with h5py.File(source, 'r') as raw, h5py.File(row['path'], 'r') as view:
        lead, n = int(row['lead_steps']), len(view['action'])
        if (n != len(raw['timestamp']) - lead or view.attrs['split'] != row['split']
                or not np.array_equal(view['action'][:, :7], raw['observations/qpos'][lead:])
                or not np.array_equal(view['action'][:, 7], raw['action'][:n, 7])
                or not np.array_equal(view['observations/qpos'][:], raw['observations/qpos'][:n])
                or not np.array_equal(view['observations/gripper_state'][:], raw['observations/gripper_state'][:n])
                or not np.array_equal(view['timestamp'][:], raw['timestamp'][:n])
                or not np.array_equal(view['joint_target_feedback_timestamp'][:], raw['timestamp'][lead:])
                or not np.array_equal(view['gripper_intent_timestamp'][:], raw['timestamp'][:n])
                or not np.array_equal(view['full_chunk_query_valid'][:], np.arange(n) < n - 19)):
            raise ValueError('Converted target/state/time arrays do not match original feedback')
        checked = 0
        for camera in raw['observations/images']:
            image = view['observations/images/' + camera]
            if (not image.is_virtual or image.shape != (n, *raw['observations/images/' + camera].shape[1:])
                    or not np.array_equal(image[0], raw['observations/images/' + camera][0])):
                raise ValueError('Virtual image mapping differs from source')
            checked += 1
    return dict(arrays_exact=True, image_pairs_checked=checked, frames=n,
                full_chunk_queries=max(0, n - 19))


def cpu_model_probe(dataset, project):
    # Reuse the existing ACT implementation in a fresh preflight process.
    sys.path.insert(0, str(project))
    from src.act import nero_policy
    if Path(nero_policy.__file__).resolve() != project / 'src/act/nero_policy.py':
        raise ValueError('Unexpected ACT implementation on import path')
    torch.set_num_threads(2)
    torch.manual_seed(24)
    config = dict(state_dim=8, action_dim=8, camera_names=['camera_0'], chunk_size=20,
                  hidden_dim=128, dim_feedforward=512, nheads=8, enc_layers=2,
                  dec_layers=2, dropout=.1, lr=1e-5, lr_backbone=1e-6,
                  weight_decay=1e-4, backbone='resnet18', pretrained_backbone=False,
                  device='cpu', kl_weight=10.)
    policy = nero_policy.NeroActPolicy(config).eval()
    def fingerprint():
        h = hashlib.sha256()
        for key, value in policy.state_dict().items():
            h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    samples = [dataset[0], dataset[len(dataset) // 2]]
    images, states, targets, padding = [torch.stack([r[i] for r in samples]) for i in range(4)]
    before = fingerprint()
    with torch.no_grad():
        predicted = policy(states, images)
        losses = policy(states, images, targets, padding)
    if (tuple(predicted.shape) != (2, 20, 8) or not torch.isfinite(predicted).all()
            or not torch.isfinite(losses['loss']) or fingerprint() != before):
        raise ValueError('ACT interface probe failed or changed model state')
    return dict(image_shape=list(images.shape), state_shape=list(states.shape),
                prediction_shape=list(predicted.shape), finite_loss=float(losses['loss']),
                optimizer_steps=0, checkpoint_loaded=False, weights_and_buffers_unchanged=True,
                scope='Freshly initialized existing ACT; interface only, no learned performance claim',
                policy_source=str(Path(nero_policy.__file__).resolve()))


def current15_rejection(path):
    group = Path(__file__).parents[1] / 'nero_act_grouped_gripper_20261021'
    script = '''import json,sys
from pathlib import Path
from training.visual_dataset import audit_visual_episode
try:
    audit_visual_episode(Path(sys.argv[1]))
except ValueError as exc:
    print(json.dumps({'rejected':True,'reason':str(exc)}))
else:
    raise RuntimeError('Legacy future-state view unexpectedly passed strict15 audit')
'''
    env = dict(os.environ, PYTHONPATH=str(group))
    result = subprocess.run([sys.executable, '-c', script, str(path)], cwd=group,
                            env=env, capture_output=True, text=True, check=True, timeout=60)
    return json.loads(result.stdout.strip().splitlines()[-1])


def run_preflight(manifest_path, project):
    manifest_path = Path(manifest_path).resolve()
    project = Path(project).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete']:
        raise ValueError('Preparation did not complete')
    proofs = [verify_conversion(r) for r in manifest['episodes']]
    source_sets = {split: {str(Path(r['source_path']).resolve()) for r in manifest['episodes'] if r['split'] == split}
                   for split in ('train', 'val', 'reserve')}
    if (source_sets['train'] & source_sets['val'] or source_sets['train'] & source_sets['reserve']
            or source_sets['val'] & source_sets['reserve']
            or sum(map(len, source_sets.values())) != len(manifest['episodes'])):
        raise ValueError('Duplicate source or split leakage in manifest')
    statistics = json.loads((manifest_path.parent / 'statistics.json').read_text())
    if set(statistics['source_paths']) != source_sets['train'] or statistics['split'] != 'train':
        raise ValueError('Statistics provenance does not match train split')
    train = FutureStateDataset([r['path'] for r in manifest['episodes'] if r['split'] == 'train'], statistics)
    val = FutureStateDataset([r['path'] for r in manifest['episodes'] if r['split'] == 'val'], statistics)
    model = cpu_model_probe(train, project)
    rejected = current15_rejection(manifest['episodes'][0]['path'])
    for row in manifest['episodes']:
        if file_snapshot(row['source_path']) != row['source_snapshot']:
            raise ValueError('Source changed during preflight')
    return dict(preflight_passed=True, episodes=len(proofs),
                split_counts=dict(Counter(r['split'] for r in manifest['episodes'])),
                converted_frames=sum(r['frames'] for r in proofs),
                full_chunk_queries=sum(r['full_chunk_queries'] for r in proofs),
                exact_target_state_time_matches=len(proofs),
                vds_frame_pairs_checked=sum(r['image_pairs_checked'] for r in proofs),
                train_query_count_stride4=len(train), val_query_count_stride4=len(val),
                same_source_split_overlap=False, normalization_uses_train_only=True,
                original_file_snapshots_unchanged=True, cpu_model_interface=model,
                original_current15_gate=rejected, current_visual15_ready=False,
                independent_position_generalization_passed=False,
                physical_task_acceptance=False, training_updates=0, hardware_commands_sent=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/home/user/projects/nero'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = run_preflight(args.manifest, args.project)
    with args.report.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
