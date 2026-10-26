"""In-memory adapter fitting on verified training windows; never exports weights."""
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from executor.model import BoundedActPolicy, physical_actions
from training.command_context import ReferenceDataset
from training.rebase_initializer import rebase_initializer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'nero_act_cvae_training_fix_20261018/model/run21_cvae_posterior'
DIRECT = ROOT.parent / 'nero_act_gripper_gradient_fix_20261018/training_data/direct_grasp_fresh_v1'


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    payload = torch.load(SOURCE / 'policy.ckpt', map_location='cpu', weights_only=False)
    config = dict(payload['config'], device='cpu', pretrained_backbone=False,
        nonlinear_gripper_head=True, train_nonlinear_gripper_only=True,
        train_gripper_head_only=False, gripper_adapter_lr=.001)
    with (SOURCE / 'dataset_stats.pkl').open('rb') as stream:
        stats = pickle.load(stream)
    paired = {key: np.asarray(value, np.float32)
              for key, value in config['constraint_loss_stats'].items()}
    policy = BoundedActPolicy(config).eval()
    policy.load_state_dict(rebase_initializer(payload['model'], paired, paired, config), strict=True)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.gripper_adapter.parameters():
        parameter.requires_grad_(True)
    records = []
    direct = json.loads((DIRECT / 'train_manifest.json').read_text())['episodes'][0]
    records.append(dict(path=direct['path'], start=0, role='actual_failed_grasp'))
    manifest = json.loads((ROOT.parent / 'nero_act_transition_fix_20261018/training_data/critical_rgb_v1/generalized_v1/manifest.json').read_text())
    for row in [row for row in manifest['episodes'] if row['split'] == 'train'][:8]:
        with h5py.File(row['path'], 'r') as stream:
            stages = stream['simulation/stage'].asstr()[:]
        records.append(dict(path=row['path'], start=0, role='initial_open'))
        for stage in ['close', 'lift', 'release']:
            first = int(np.flatnonzero(stages == stage)[0])
            for offset in [0, 10]:
                records.append(dict(path=row['path'], start=first+offset, role=stage))
    features, bases, targets, samples = [], [], [], []
    captured = []
    handle = policy.model.action_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    for row in records:
        sample = ReferenceDataset([Path(row['path'])], config['camera_names'], stats,
                                  fixed_start_ts=row['start'])[0]
        image, q, target, padding, _ = sample
        assert not padding[:20].any()
        with torch.no_grad():
            out = policy(q[None], image[None], return_unbounded=True)
        features.append(captured.pop())
        bases.append(out['unbounded_a_hat'][..., 7:8])
        targets.append(target[None, :20, 7:8])
        samples.append((q[None], image[None], out['a_hat'][..., :7].clone()))
    handle.remove()
    feature, base, target = map(torch.cat, [features, bases, targets])
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    optimizer = torch.optim.AdamW(policy.gripper_adapter.parameters(), lr=.001, weight_decay=0.)
    initial = float((base-target).abs().mean())
    for step in range(1000):
        optimizer.zero_grad(set_to_none=True)
        prediction = base + policy.gripper_adapter(feature)
        loss = torch.nn.functional.smooth_l1_loss(prediction, target, beta=.05)
        loss.backward()
        optimizer.step()
    widths = []
    arm_error = 0.
    for row, (q, image, old_arm) in zip(records, samples):
        with torch.no_grad():
            out = policy(q, image, return_unbounded=True)
            physical = physical_actions(out['a_hat'], stats, bounded_gripper=True)
            raw = out['unbounded_a_hat'][..., 7]*stats['action_std'][7]+stats['action_mean'][7]
        arm_error = max(arm_error, float((old_arm-out['a_hat'][..., :7]).abs().max()))
        widths.append(dict(role=row['role'], start=row['start'], path=row['path'],
            bounded_first10_mm=(physical[0, :10, 7]*1000).tolist(),
            raw_mae_mm=float(((raw-targets[len(widths)][..., 0]*stats['action_std'][7]
                -stats['action_mean'][7]).abs().mean())*1000)))
    changed = [key for key, value in policy.state_dict().items() if not torch.equal(before[key], value)]
    frozen = all(key.startswith('gripper_adapter.') for key in changed) and arm_error == 0.
    grasp = min(widths[0]['bounded_first10_mm']) < 94.9
    checks = {role: max(row['raw_mae_mm'] for row in widths if row['role'] == role)
              for role in sorted({row['role'] for row in widths})}
    passed = frozen and grasp and all(value < 1. for value in checks.values())
    report = dict(passed=passed, probe_only_not_task_success=True, weights_saved=False,
        training_windows=len(records), optimization_steps=1000, future_labels_only_in_loss=True,
        all_windows_from_training_groups=True, unchanged_arm_output_max=arm_error,
        changed_parameters=changed, frozen_base_passed=frozen,
        initial_raw_l1=initial, final_raw_l1=float((base+policy.gripper_adapter(feature)-target).abs().mean().detach()),
        worst_raw_mae_mm_by_role=checks, records=widths, hardware_connected=False)
    (ROOT / 'evidence/nonlinear_probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2))
    if not passed:
        raise RuntimeError('Adapter mechanism probe failed; do not start full training')


if __name__ == '__main__':
    main()
