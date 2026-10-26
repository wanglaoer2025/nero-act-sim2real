"""One declared 1000-update pilot, using physical complete-task labels only."""
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from learning import ROOT, load_for_learning, prefix_objective
from pipeline import PARENT, snapshot, require_unchanged
from training.visual_dataset import VisualReferenceDataset, validate_visual_groups
from training.transition_sampling import critical_windows
from training.approach_preservation import ApproachPreservation, validate_approach_replay

STAGES = ('approach', 'descend', 'close', 'lift', 'carry', 'place', 'release', 'retreat')


def prepare_data(stats, config):
    new_manifest = ROOT/'data/manifest.json'
    new = json.loads(new_manifest.read_text())
    old_manifest = PARENT/'training_data/uniform_complete_v1/manifest.json'
    old = json.loads(old_manifest.read_text())
    if not new.get('complete'):
        raise ValueError('Finish physical and visual audits before training')
    rows = [r for r in new['episodes']+old['episodes'] if r['split'] == 'train']
    val_rows = [r for r in new['episodes'] if r['split'] == 'val']
    paths, val_paths = [[Path(r['path']) for r in group] for group in [rows, val_rows]]
    validate_visual_groups(paths, val_paths)
    require_unchanged({str(Path(r['path']).resolve()): r['sha256'] for r in rows+val_rows})
    candidates = sorted(set(critical_windows(paths, 20, command_context=True)))
    stages, valid = [], []
    for path in paths:
        with h5py.File(path) as stream:
            stages.append(stream['simulation/stage'].asstr()[:])
            valid.append(stream['observations/visual_valid'][:].astype(bool))
    samples = [(i, t) for i, t in candidates if valid[i][t] and stages[i][t] in STAGES]
    pools = {(family, stage): [] for family in ['new', 'retained', 'recovery'] for stage in STAGES}
    for index, (i, t) in enumerate(samples):
        family = ('recovery' if 1100 <= rows[i]['id'] < 1200 else
                  'new' if 1000 <= rows[i]['id'] < 1011 else 'retained')
        pools[family, stages[i][t]].append(index)
    if any(not pool for pool in pools.values()):
        raise ValueError('Both families must cover all eight task stages')
    rng = np.random.default_rng(42)
    # Each batch:8 new,8 retained,16 correction samples; stages only choose offline labels.
    batches = np.array([np.concatenate([rng.choice(pool, 2 if family == 'recovery' else 1, replace=True)
                                       for (family, stage), pool in pools.items()]) for _ in range(1000)])
    dataset = VisualReferenceDataset(paths, ['camera_0'], stats, policy_config=config,
        image_size=tuple(config['image_size']), sample_starts=samples)
    val_candidates = sorted(set(critical_windows(val_paths, 20, command_context=True)))
    val_valid = []
    for path in val_paths:
        with h5py.File(path) as stream:
            val_valid.append(stream['observations/visual_valid'][:].astype(bool))
    val_samples = [(i, t) for i, t in val_candidates if val_valid[i][t]]
    val_dataset = VisualReferenceDataset(val_paths, ['camera_0'], stats, policy_config=config,
        image_size=tuple(config['image_size']), sample_starts=val_samples)
    physical = [Path(r.get('physical_path', ROOT/'data/physical'/f'episode_{r["id"]}.hdf5')) for r in new['episodes']]
    require_unchanged({str(p.resolve()): r['physical_sha256'] for p, r in zip(physical, new['episodes'])})
    return dataset, batches, val_dataset, [new_manifest, old_manifest, ROOT/'physical_audit_progress.json']+physical+paths+val_paths, dict(
        train_groups=sorted({r['source_group'] for r in rows}),
        val_groups=sorted({r['source_group'] for r in val_rows}), samples=len(samples),
        val_samples=len(val_samples), pool_counts={str(k): len(v) for k, v in pools.items()})


def validate(policy, dataset, stats):
    loader = DataLoader(dataset, batch_size=32, num_workers=2,
                        multiprocessing_context='spawn', shuffle=False)
    totals = np.zeros(3); count = 0
    with torch.no_grad():
        for image, current, target, padding, _ in loader:
            loss, parts = prefix_objective(policy(current.cuda(), image.cuda()),
                                           target.cuda(), padding.cuda(), stats)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite validation objective')
            n = len(image); count += n
            totals += n*np.array([loss.item(), parts['arm'].item(), parts['gripper'].item()])
    return dict(zip(['total', 'arm', 'gripper'], (totals/count).tolist()), samples=count)


def main():
    torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)
    output = ROOT/'training/pilot_A'
    output.mkdir(parents=True, exist_ok=False)
    policy, stats, groups = load_for_learning()
    dataset, batches, val_dataset, inputs, data_report = prepare_data(stats, policy.config)
    np.save(output/'batch_order.npy', batches, allow_pickle=False)
    files = inputs+[ROOT/'learning.py', Path(__file__), ROOT/'prepare.py', ROOT/'evaluate.py',
                    ROOT/'plan.json', ROOT/'data/scene_protocol.json', output/'batch_order.npy']
    files += [Path(module.__file__) for module in tuple(sys.modules.values())
              if getattr(module, '__file__', None)
              and str(module.__file__).startswith(str(ROOT.parent)+'/')
              and str(module.__file__).endswith('.py')]
    files += [PARENT/'training_data/uniform_complete_v1/dataset_stats.pkl']
    first = ROOT.parent/'nero_act_position_adaptation_20261022'
    files += [first/'training/pilot_A/policy_last.ckpt', first/'training/pilot_A/integrity.json']
    files += [first/'results'/case/name for case in ['current_development','reference_layout']
              for name in ['physical_snapshots.json','scene_configuration.json','result.json']]
    hashes = snapshot(files)
    preservation = dict(policy.config['approach_preservation'], weight=.1, batch_size=4)
    hashes.update(validate_approach_replay(preservation['replay_path']))
    keeper = ApproachPreservation(preservation, stats, 'cuda')
    (output/'data_audit.json').write_text(json.dumps(data_report, indent=2)+'\n')
    before = validate(policy, val_dataset, stats)
    print(json.dumps(dict(stage='validation_before', result=before)), flush=True)
    frozen = {name: value.detach().cpu().clone() for name, value in policy.named_parameters()
              if not value.requires_grad}
    optimizer = torch.optim.AdamW(groups, weight_decay=0.)
    loader = DataLoader(dataset, batch_sampler=batches.tolist(), num_workers=2,
                        multiprocessing_context='spawn', pin_memory=True)
    started = time.monotonic(); updates = 0
    initial_rates = [g['lr'] for g in optimizer.param_groups]
    for updates, (image, current, target, padding, _) in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = policy(current.cuda(non_blocking=True), image.cuda(non_blocking=True))
        loss, parts = prefix_objective(prediction, target.cuda(), padding.cuda(), stats)
        retention_loss = .1*keeper.loss(policy, updates-1)
        loss = loss+retention_loss
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite pilot objective')
        loss.backward()
        if updates == 1:
            evidence = dict(position_geometry_gradient=float(
                policy.base.model.input_proj_robot_state.weight.grad[:, 9:].norm()),
                transformer_gradient=sum(float(p.grad.norm()) for p in
                    policy.base.model.transformer.parameters() if p.grad is not None),
                frozen_parameters_with_grad=[name for name, p in policy.named_parameters()
                                             if name in frozen and p.grad is not None])
            if evidence['position_geometry_gradient'] <= 0 or evidence['frozen_parameters_with_grad']:
                raise ValueError('Position learning or freeze contract failed')
            (output/'first_optimizer_gradients.json').write_text(json.dumps(evidence, indent=2)+'\n')
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],
                                      1., error_if_nonfinite=True)
        for group, rate in zip(optimizer.param_groups, initial_rates):
            group['lr'] = rate*(1-.9*(updates-1)/999)
        optimizer.step()
        if updates == 1 or updates % 50 == 0:
            status = dict(complete=False, updates=updates, limit=1000,
                loss=loss.item(), arm_loss=parts['arm'].item(), gripper_loss=parts['gripper'].item(),
                retention_loss=retention_loss.item(),
                elapsed_s=time.monotonic()-started, hardware_commands_sent=False)
            (output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
            print(json.dumps(status), flush=True)
    if updates != 1000:
        raise ValueError('Declared1000-update budget was not completed exactly')
    changed_frozen = [name for name, p in policy.named_parameters()
                      if name in frozen and not torch.equal(p.detach().cpu(), frozen[name])]
    if changed_frozen:
        raise ValueError('Frozen visual/CVAE parameters changed: '+str(changed_frozen))
    require_unchanged(hashes)
    after = validate(policy, val_dataset, stats)
    checkpoint = output/'policy_last.ckpt'
    torch.save(dict(profile='position_recovery_pilot_v2', actual_updates=updates,
                    model=policy.state_dict(), stats=stats, config=policy.config,
                    hardware_deployment_allowed=False), checkpoint)
    proof = dict(complete=True, actual_updates=updates, formal_epochs=0,
                 initializer_first_pilot_updates=1000, cumulative_new_optimizer_updates=2000,
                 checkpoint_selection='fixed_last_no_validation_selection',
                 source_sha256=hashes, checkpoint_sha256=snapshot([checkpoint]),
                 frozen_parameters_unchanged=True, validation_before=before, validation_after=after,
                 hardware_commands_sent=False, hardware_deployment_allowed=False)
    (output/'integrity.json').write_text(json.dumps(proof, indent=2)+'\n')
    (output/'status.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(dict(stage='training_complete', actual_updates=updates,
                          validation_before=before, validation_after=after)), flush=True)


if __name__ == '__main__':
    main()
