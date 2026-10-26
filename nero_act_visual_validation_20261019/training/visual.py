"""Offline visual15D ACT training; no ROS, SDK, hardware or runtime authorization."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from executor.contract import VISUAL_PROFILE
from training.train import FullTaskPolicy, save_checkpoint
from training.command_context import expand_visual_state_dict
from training.rebase_initializer import paired_config, rebase_initializer
from training.visual_dataset import VisualReferenceDataset, validate_visual_groups, visual_statistics
from training.integrity import snapshot, require_unchanged
from training.transition_sampling import critical_windows

ROOT = Path(__file__).resolve().parents[1]


def build_visual_policy(config):
    if (config.get('visual_observation_profile') != VISUAL_PROFILE
            or config.get('policy_input_dim') != 15 or config.get('cvae_training') is not True
            or config.get('nonlinear_gripper_head') is not True
            or any(config.get(k) for k in ['train_arm_output_and_gripper_only',
                                           'train_nonlinear_gripper_only','train_gripper_head_only'])):
        raise ValueError('Visual15D training requires active CVAE, geometry/arm branches and nonlinear gripper; no heads-only mode')
    config = dict(config,freeze_backbone=True)
    policy = FullTaskPolicy(config)
    groups = {'policy':[], 'visual':[], 'gripper':[]}
    for name,parameter in policy.named_parameters():
        if name.startswith('model.backbones.'):
            parameter.requires_grad_('.layer4.' in name)
            group = 'visual'
        elif name.startswith('gripper_adapter.'):
            group = 'gripper'
        else:
            group = 'policy'
        if parameter.requires_grad:
            groups[group].append(parameter)
    if not all(groups.values()):
        raise ValueError('Expected trainable policy, ResNet layer4 and gripper parameter groups')
    rates = dict(policy=1e-4, visual=1e-5, gripper=3e-4)
    policy.optimizer = torch.optim.AdamW([
        dict(params=values,lr=rates[name],weight_decay=0. if name=='gripper' else float(config.get('weight_decay',0.)))
        for name,values in groups.items()])
    policy.declared_training_epochs = int(config.get('num_epochs', 20))
    return policy


def validate_formal_config(config):
    if (config.get('seed') != 42 or int(config.get('batch_size', 0)) <= 0
            or int(config.get('num_epochs', 0)) <= 0
            or config.get('image_size') not in ([90, 160], [180, 320])
            or any(config.get(k) for k in ['train_arm_output_and_gripper_only',
                    'train_nonlinear_gripper_only', 'train_gripper_head_only'])):
        raise ValueError('Explicit seed42, positive epochs/batch, supported resolution and full ACT training required')


def initialize_visual(source, stats, overrides):
    config = dict(source['config'])
    config.update(policy_input_dim=15, visual_observation_profile=VISUAL_PROFILE,
        physical_state_dim=8, state_dim=8, action_dim=8, chunk_size=20,
        cvae_training=True, kl_weight=.001, prior_gripper_loss_weight=.005,
        nonlinear_gripper_head=True, train_arm_output_and_gripper_only=False,
        train_nonlinear_gripper_only=False, train_gripper_head_only=False,
        prefix_margin_loss_weight=0., reset_action_head=False,
        joint_target_lead_steps=None, joint_lookahead_steps=0, alignment_offset=0,
        coordinate_conditioning=False, previous_gripper_conditioning=False,
        task_displacement_features=False, hardware_deployment_allowed=False,
        training_revision='visual_relative15_retained_calibration_full_ACT_v1',
        checkpoint_selection='minimum_zero_latent_validation_total_loss',
        trainable_parameter_scope='ACT_CVAE_state_projection_arm_output_gripper_and_resnet_layer4')
    config.update(overrides)
    config = paired_config(config, stats)
    old = {k: np.asarray(v, np.float32) for k, v in source['config']['constraint_loss_stats'].items()}
    old.update(geometry_mean=np.zeros(6, np.float32), geometry_std=np.ones(6, np.float32))
    expanded = expand_visual_state_dict(source['model'])
    weights = rebase_initializer(expanded, old, stats, config)
    return config, weights


def visual_windows(paths):
    candidates = critical_windows(paths, 20, command_context=True)
    masks = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            masks.append(f['observations/visual_valid'][:].astype(bool))
    result = [(i, t) for i, t in candidates if masks[i][t]]
    if not result or set(i for i, _ in result) != set(range(len(paths))):
        raise ValueError('Every episode needs valid visual training inputs')
    return result


def gradient_evidence(policy):
    def norm(value):
        return float(value.grad.detach().norm()) if value.grad is not None else 0.
    values = dict(posterior_grad_norm=norm(policy.model.latent_proj.weight),
        geometry_projection_grad_norm=float(policy.model.input_proj_robot_state.weight.grad[:, 9:].norm()),
        layer4_grad_norm=sum(norm(p) for n, p in policy.named_parameters()
                             if n.startswith('model.backbones.') and '.layer4.' in n),
        frozen_backbone_has_grad=any(p.grad is not None for n, p in policy.named_parameters()
                             if n.startswith('model.backbones.') and '.layer4.' not in n))
    if (any(not np.isfinite(values[k]) or values[k] <= 0 for k in
            ['posterior_grad_norm', 'geometry_projection_grad_norm', 'layer4_grad_norm'])
            or values['frozen_backbone_has_grad']):
        raise RuntimeError('Required ACT/geometry/visual gradients or frozen-layer invariant failed: '+str(values))
    return values


def run_epoch(policy, loader, device, *, training, output=None, epoch=0):
    policy.train(training)
    sums = {}; count = 0
    for step, batch in enumerate(loader, 1):
        image, state, action, padding = [v.to(device) for v in batch[:4]]
        with torch.set_grad_enabled(training):
            losses = policy(state, image, action, padding)
            if not torch.isfinite(losses['loss']):
                raise RuntimeError('Nonfinite ACT loss')
            if training:
                policy.optimizer.zero_grad(set_to_none=True)
                losses['loss'].backward()
                if epoch == 1 and step == 1:
                    proof = gradient_evidence(policy)
                    if output is not None:
                        (output/'artifacts/first_optimizer_gradients.json').write_text(json.dumps(proof, indent=2))
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), .1)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite ACT gradient')
                policy.optimizer.step()
        n = len(image); count += n
        for key, value in losses.items():
            if key != 'a_hat': sums[key] = sums.get(key, 0.)+float(value.detach())*n
        with torch.no_grad():
            difference = (losses['a_hat']-action).abs()
            valid = (~padding).to(difference.dtype)
            scale = torch.as_tensor(policy.loss_stats['action_std'], device=device)
            physical = difference*scale
            for name, value in [('arm_mae_rad', (physical[:, :, :7].mean(-1)*valid).sum()/valid.sum().clamp_min(1)),
                                ('gripper_mae_m', (physical[:, :, 7]*valid).sum()/valid.sum().clamp_min(1))]:
                sums[name] = sums.get(name, 0.)+float(value)*n
        if output is not None and training and (step == 1 or step % 25 == 0):
            status = dict(stage='optimizing', epoch=epoch, total_epochs=policy.declared_training_epochs,
                optimizer_steps_this_epoch=step, batches_this_epoch=len(loader), samples_seen_this_epoch=count,
                loss=float(losses['loss'].detach()), hardware_deployment_allowed=False,
                updated_unix_s=time.time())
            (output/'status.json').write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
    if not count: raise ValueError('Empty training/validation loader')
    return {k: v/count for k, v in sums.items()}


def finalize_training(output, config, history, protected):
    if (not history or len(history) != config['num_epochs']
            or [r['epoch'] for r in history] != list(range(1, config['num_epochs']+1)) or not protected):
        raise ValueError('All declared epochs and protected inputs must complete before finalization')
    output = Path(output)
    require_unchanged(protected)
    outputs = list((output/'checkpoints').glob('policy*.ckpt'))+[output/'checkpoints/dataset_stats.pkl']
    proof = dict(finalized=True, declared_epochs=config['num_epochs'], actual_epochs=len(history),
        protected_sha256=protected, output_sha256=snapshot(outputs), hardware_deployment_allowed=False)
    (output/'artifacts/training_integrity.json').write_text(json.dumps(proof, indent=2))
    return proof


def complete_training(output, config, history, protected, summary):
    finalize_training(output, config, history, protected)
    (output/'artifacts/training_summary.json').write_text(json.dumps(summary, indent=2))
    (output/'status.json').write_text(json.dumps(dict(stage='training_complete_evaluation_pending', **summary), indent=2))


def train_visual(data_dir, initializer, settings, output):
    validate_formal_config(settings)
    data_dir, initializer, output = map(lambda p: Path(p).resolve(), [data_dir, initializer, output])
    if output.exists(): raise FileExistsError(output)
    manifest = json.loads((data_dir/'manifest.json').read_text())
    split = json.loads((data_dir/'split.json').read_text())
    if not manifest.get('complete') or manifest.get('visual_profile') != VISUAL_PROFILE:
        raise ValueError('Complete explicit visual observation dataset required')
    rows = manifest['episodes']
    if (len(rows) != 48 or len(split['train']) != 36 or len(split['val']) != 12
            or len({r['id'] for r in rows}) != 48):
        raise ValueError('Expected existing48 full demonstrations with36/12 groups')
    from training.generalized import validate_manifest
    validate_manifest(manifest, split)
    for row in rows:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Derived visual demonstration changed')
    paths = {key: [Path(r['path']) for r in rows if r['id'] in split[key]] for key in ['train', 'val']}
    groups = validate_visual_groups(paths['train'], paths['val'])
    stats = visual_statistics(paths['train'])
    source = torch.load(initializer, map_location='cpu', weights_only=False)
    scene = json.loads((data_dir/'scene_config.json').read_text())
    from training.pose_objective import fk_spec_from_scene
    settings = dict(settings, scene_config=scene, pose_kinematics_spec=fk_spec_from_scene(scene),
                    pose_loss_weight=.05, calibration_sha256=manifest['calibration_sha256'])
    config, weights = initialize_visual(source, stats, settings)
    config.update(dataset_dir=str(data_dir), manifest=str(data_dir/'manifest.json'),
        split_file=str(data_dir/'split.json'), normalization_source='new_training_groups_only',
        training_integrity_required=True, checkpoint_dir=str(output/'checkpoints'),
        artifact_dir=str(output/'artifacts'), training_log=str(output/'training.csv'))
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device(config.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Formal training requested CUDA but it is unavailable')
    output.mkdir(parents=True)
    (output/'checkpoints').mkdir(); (output/'artifacts').mkdir()
    (output/'config.json').write_text(json.dumps(config, indent=2))
    (output/'split.json').write_text(json.dumps(split, indent=2))
    with (output/'checkpoints/dataset_stats.pkl').open('wb') as f: pickle.dump(stats, f)
    source_files = [p for folder in ['training', 'executor', 'calibration', 'vision', 'simulation', 'policy_source']
                    for p in (ROOT/folder).rglob('*.py')]
    protected = snapshot(source_files+[p for values in paths.values() for p in values]+[
        initializer, data_dir/'manifest.json', data_dir/'split.json', data_dir/'calibration_profile.json',
        data_dir/'scene_config.json', ROOT/'evaluation/protocol.json',
        ROOT/'evidence/retained_calibration_sources.json',
        output/'config.json', output/'checkpoints/dataset_stats.pkl'])
    (output/'artifacts/training_inputs_before.json').write_text(json.dumps(protected, indent=2))
    loaders = {}
    for role in paths:
        samples = visual_windows(paths[role])
        if role == 'val': samples = list(dict.fromkeys(samples))
        dataset = VisualReferenceDataset(paths[role], config['camera_names'], stats,
            policy_config=config, image_size=config['image_size'], sample_starts=samples)
        loaders[role] = DataLoader(dataset, batch_size=config['batch_size'], shuffle=role == 'train',
            num_workers=int(config.get('data_workers', 2)), pin_memory=device.type == 'cuda')
    policy = build_visual_policy(config).to(device)
    policy.load_state_dict(weights, strict=True)
    # optimizer groups are explicit and differ from inherited run27 heads-only mode.
    history = []; best = float('inf'); started = time.monotonic()
    (output/'artifacts/data_groups.json').write_text(json.dumps(dict(groups=groups,
        train_windows=len(loaders['train'].dataset), val_windows=len(loaders['val'].dataset)), indent=2))
    with (output/'training.csv').open('w') as stream:
        writer = None
        for epoch in range(1, config['num_epochs']+1):
            train = run_epoch(policy, loaders['train'], device, training=True, output=output, epoch=epoch)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
                torch.manual_seed(42)
                val = run_epoch(policy, loaders['val'], device, training=False)
            row = dict(epoch=epoch, **{'train_'+k: v for k, v in train.items()},
                       **{'val_'+k: v for k, v in val.items()}, elapsed_s=time.monotonic()-started)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); stream.flush(); history.append(row)
            save_checkpoint(output/'checkpoints/policy_last.ckpt', policy, config, split, epoch, row)
            if val['loss'] < best:
                best = val['loss']
                save_checkpoint(output/'checkpoints/policy_best_val.ckpt', policy, config, split, epoch, row)
            print(json.dumps(row), flush=True)
    summary = dict(complete=True, best_zero_latent_val_loss=best, actual_epochs=len(history),
                   last_metrics=history[-1], elapsed_s=time.monotonic()-started, hardware_deployment_allowed=False)
    complete_training(output, config, history, protected, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--initializer', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        train_visual(args.data_dir, args.initializer, json.loads(args.config.read_text()), args.output)
    except Exception as exc:
        if args.output.exists():
            (args.output/'failure.json').write_text(json.dumps(dict(error=str(exc), hardware_deployment_allowed=False), indent=2))
        raise


if __name__ == '__main__':
    main()
