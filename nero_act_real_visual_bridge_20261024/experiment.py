"""Fixed200-update CPU A/B: real8 auxiliary supervision, unchanged base15 weights."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

from bridge import (VisualResidual, AuxiliaryHead, auxiliary_loss, physical_state,
                    decoded_auxiliary, projected_features, attach_visual_adapter)

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
GROUP = WORKSPACE / 'nero_act_grouped_gripper_20261021'
REAL = WORKSPACE / 'nero_act_real_data_adapter_20261024'
EVENT = WORKSPACE / 'nero_act_event_finetune_20261024'
CHECKPOINT = WORKSPACE / 'nero_act_new_handeye_policy_repair_20261023/training/pilot_A/policy_last.ckpt'
CHECKPOINT_SHA = '1a28f4fddd3095791403028b639b99f9775b91caab45c950c2d0f518955bc0f0'
STAGES = ('approach', 'descend', 'close', 'lift', 'carry', 'place', 'release', 'retreat')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def event_query_starts(actions):
    if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) < 20:
        raise ValueError('Full20-point recorded targets required')
    events = np.flatnonzero(np.abs(np.diff(actions[:, 7])) > 1e-5) + 1
    close = [int(i) for i in events if actions[i, 7] < actions[i - 1, 7]]
    opened = [int(i) for i in events if close and i > close[0] and actions[i, 7] > actions[i - 1, 7]]
    if not close or not opened:
        raise ValueError('Both close and release intent events required')
    c, o, end = close[0], opened[0], len(actions) - 20
    return sorted({min(end, max(0, int(t))) for t in (0, c-4, c-1, c, c+2, c+20, o-2, o, o+2, end)})


def stage_partition(stages, valid=None):
    valid = np.ones(len(stages), bool) if valid is None else np.asarray(valid)
    if valid.shape != (len(stages),) or valid.dtype != bool:
        raise ValueError('Paired boolean visual validity required')
    train, check = [], []
    for stage in STAGES:
        indices = np.flatnonzero((stages == stage) & valid)
        if not len(indices):
            continue
        if len(indices) < 6:
            raise ValueError('Six distinct queries per preservation stage required')
        selected = indices[np.linspace(0, len(indices)-1, 6).round().astype(int)].tolist()
        train.extend(selected[:3]); check.extend(selected[3:])
    if not train or set(train) & set(check):
        raise ValueError('Disjoint stage rehearsal/check queries required')
    return train, check


def validate_real_partition(rows):
    seen = set()
    for row in rows:
        source = str(Path(row['source_path']).resolve())
        if source in seen:
            raise ValueError('Duplicate real source across split entries')
        seen.add(source)


def acceptance(a, b, preservation):
    values = [a['loss'], b['loss'], a['gripper_intent_accuracy'], b['gripper_intent_accuracy'],
              preservation['maximum_arm_change_rad'], preservation['maximum_gripper_change_m']]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError('Finite nonnegative gate metrics required')
    gain = (a['loss'] - b['loss']) / max(a['loss'], 1e-12)
    gates = dict(real_validation_gain=gain >= .05,
                 gripper_intent_nonregression=b['gripper_intent_accuracy'] >= a['gripper_intent_accuracy'],
                 main15_all20_preserved=preservation['maximum_arm_change_rad'] <= .001
                 and preservation['maximum_gripper_change_m'] <= .001)
    return dict(gates=gates, real_validation_gain_fraction=gain,
                mechanism_gate_passed=all(gates.values()), policy_adopted=False,
                full_flow_tested=False, hardware_deployment_allowed=False)


def load_baseline():
    if digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Exact1a28 checkpoint required')
    sys.path.insert(0, str(GROUP))
    from executor.model import BoundedActPolicy
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    construction = dict(payload['config'], device='cpu', pretrained_backbone=False)
    base = BoundedActPolicy(construction)
    base.config = copy.deepcopy(payload['config'])
    spec = importlib.util.spec_from_file_location('_bridge_increment_architecture',
             WORKSPACE / 'nero_act_command_increment_20261021/model.py')
    increment = importlib.util.module_from_spec(spec); spec.loader.exec_module(increment)
    head = increment.IncrementHead(construction['hidden_dim'], payload['stats'])
    policy = increment.FrozenIncrementPolicy(base, head, payload['stats'], payload['config'])
    policy.load_state_dict(payload['model'], strict=True)
    policy.requires_grad_(False).eval()
    return policy, payload['stats']


def cache_features(policy, images):
    values = []
    with torch.no_grad():
        for start in range(0, len(images), 4):
            values.append(projected_features(policy, images[start:start+4]).detach())
    return torch.cat(values)


def prepare_real(policy):
    spec = importlib.util.spec_from_file_location('_bridge_real_adapter', REAL / 'adapter.py')
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    integrity = json.loads((REAL / 'integrity.json').read_text())
    for row in integrity['assets']:
        if digest(row['path']) != row['sha256']:
            raise ValueError('Frozen real-data adapter asset changed: ' + row['path'])
    manifest = json.loads((REAL / 'data/manifest.json').read_text())
    statistics = json.loads((REAL / 'data/statistics.json').read_text())
    validate_real_partition(manifest['episodes'])
    caches, identities = {}, []
    for split in ('train', 'val'):
        images, states, targets, masks, rows = [], [], [], [], []
        for entry in manifest['episodes']:
            if entry['split'] != split:
                continue
            legacy.verify_source(entry['path'], numeric=True)
            with h5py.File(entry['path'], 'r') as f:
                starts = event_query_starts(f['action'][:])
            for start in starts:
                image, state, action, padding, _ = legacy.read_window(entry['path'], start, statistics)
                if padding.any():
                    raise ValueError('Selected real window crossed its episode tail')
                images.append(image); states.append(state); masks.append(padding)
                targets.append(action * action.new_tensor(statistics['action_std']) + action.new_tensor(statistics['action_mean']))
                rows.append(dict(path=entry['path'], source_path=entry['source_path'], split=split, start=start))
        state8 = torch.stack(states)
        caches[split] = dict(features=cache_features(policy, torch.stack(images)), state=state8,
                             measured=physical_state(state8, statistics), targets=torch.stack(targets),
                             padding=torch.stack(masks))
        identities.extend(rows)
        print(json.dumps({'cached_real_split': split, 'queries': len(rows)}), flush=True)
    if set(statistics['source_paths']) != {r['source_path'] for r in identities if r['split'] == 'train'}:
        raise ValueError('Real train-only normalization provenance mismatch')
    return caches, identities


def baseline_predictions(policy, current, image):
    with torch.no_grad():
        return torch.cat([policy(current[i:i+1], image[i:i+1]) for i in range(len(current))])


def prepare_preservation(policy, stats):
    from training.visual_dataset import VisualReferenceDataset, audit_visual_episode
    manifest = json.loads((EVENT / 'data/manifest.json').read_text())
    if not manifest['complete'] or manifest['rejected']:
        raise ValueError('Complete current-camera expert corpus required')
    paths, slots, roles, labels = [], [], [], []
    for i, row in enumerate(manifest['episodes']):
        path = Path(row['path'])
        if digest(path) != row['sha256'] or not row['full_task_verified']:
            raise ValueError('Frozen verified preservation source required')
        audit_visual_episode(path)
        paths.append(path)
        with h5py.File(path, 'r') as f:
            stages = f['simulation/stage'].asstr()[:]
            valid = f['observations/visual_valid'][:].astype(bool)
            bound = f['observations/visual_error_bound_m'][:]
            valid &= np.isfinite(bound) & (bound >= 0) & np.isfinite(f['observations/visual_relative_m'][:]).all(1)
        train, check = stage_partition(stages, valid)
        for role, selected in (('rehearsal', train), ('check', check)):
            for t in selected:
                slots.append((i, t)); roles.append(role); labels.append(str(stages[t]))
    dataset = VisualReferenceDataset(paths, ['camera_0'], stats, policy_config=policy.config,
                                    image_size=tuple(policy.config['image_size']), sample_starts=slots)
    samples = [dataset[i] for i in range(len(dataset))]
    image, current = [torch.stack([s[i] for s in samples]).contiguous() for i in (0, 1)]
    train = np.flatnonzero(np.array(roles) == 'rehearsal').tolist()
    check = np.flatnonzero(np.array(roles) == 'check').tolist()
    features = cache_features(policy, image[train])
    checks = [dict(name='multistage_current_camera', current=current[check], image=image[check],
                   reference=baseline_predictions(policy, current[check], image[check]),
                   stages=[labels[i] for i in check])]
    replay_proof = json.loads((EVENT / 'preservation/proof.json').read_text())
    for path, expected in replay_proof['replay_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Frozen visited-state replay changed')
    with np.load(EVENT / 'preservation/current_camera.npz', allow_pickle=False) as f:
        old_current = torch.from_numpy(f['current'].copy()).float()
        old_image = torch.from_numpy(f['image'].copy()).float().contiguous()
        old_target = torch.from_numpy(f['target'].copy()).float()
    old_cpu = baseline_predictions(policy, old_current, old_image)
    archive_delta = ((old_cpu - old_target) * old_cpu.new_tensor(stats['action_std'])).abs()
    checks.append(dict(name='visited_approach96', current=old_current, image=old_image,
                       reference=old_cpu, stages=['visited_approach'] * len(old_current)))
    # Confirm the standalone cache extractor uses the same normalized visual path.
    captured = []
    handle = policy.base.model.input_proj.register_forward_hook(lambda m, args, output: captured.append(output.detach()))
    baseline_predictions(policy, current[:1], image[:1]); handle.remove()
    with torch.no_grad():
        cache_delta = float((captured[0] - projected_features(policy, image[:1])).abs().max())
    if cache_delta > 1e-6:
        raise ValueError('Feature cache and original visual path disagree')
    index = [dict(path=str(paths[i]), start=int(t), role=role, stage=stage)
             for (i, t), role, stage in zip(slots, roles, labels)]
    return features, checks, dict(indices=index, feature_path_max_abs_difference=cache_delta,
                                 cpu_vs_archived_cuda_max_arm_rad=float(archive_delta[:, :, :7].max()),
                                 cpu_vs_archived_cuda_max_gripper_m=float(archive_delta[:, :, 7].max()),
                                 cpu_rebaselined=True)


def assess_preservation(policy, adapter, checks, stats):
    record = []; overall = np.zeros(2)
    handle = attach_visual_adapter(policy, adapter)
    try:
        for group in checks:
            predicted = baseline_predictions(policy, group['current'], group['image'])
            delta = ((predicted - group['reference']) * predicted.new_tensor(stats['action_std'])).abs()
            maximum = [float(delta[:, :, :7].max()), float(delta[:, :, 7].max())]
            overall = np.maximum(overall, maximum)
            stages = {}
            for stage in sorted(set(group['stages'])):
                selected = [i for i, x in enumerate(group['stages']) if x == stage]
                stages[stage] = dict(queries=len(selected), maximum_arm_change_rad=float(delta[selected, :, :7].max()),
                                     maximum_gripper_change_m=float(delta[selected, :, 7].max()))
            record.append(dict(name=group['name'], queries=len(predicted), all20=True,
                               maximum_arm_change_rad=maximum[0], maximum_gripper_change_m=maximum[1], stages=stages))
    finally:
        handle.remove()
    return dict(maximum_arm_change_rad=float(overall[0]), maximum_gripper_change_m=float(overall[1]), records=record)


def metrics(adapter, head, data, *, shuffle=False):
    with torch.no_grad():
        features = data['features'].roll(1, 0) if shuffle else data['features']
        prediction = head(adapter(features), data['state'])
        loss = auxiliary_loss(prediction, data['measured'], data['targets'], data['padding'])
        decoded = decoded_auxiliary(prediction, data['measured'])
        error = (decoded - data['targets']).abs()
        accurate = (decoded[:, :, 7] >= .0475) == (data['targets'][:, :, 7] >= .0475)
        return dict(loss=float(loss), joint_mae_deg=float(error[:, :, :7].mean() * 180/np.pi),
                    gripper_intent_mae_mm=float(error[:, :, 7].mean() * 1000),
                    gripper_intent_accuracy=float(accurate.float().mean()), queries=len(prediction))


def run_variant(name, policy, stats, real, keep_features, checks, batch_order, initial_head, initial_adapter, output):
    folder = output / name; folder.mkdir()
    head, adapter = AuxiliaryHead(), VisualResidual()
    head.load_state_dict(initial_head); adapter.load_state_dict(initial_adapter)
    adapter.requires_grad_(name == 'B')
    groups = [dict(params=list(head.parameters()), lr=1e-3)]
    if name == 'B': groups.append(dict(params=list(adapter.parameters()), lr=1e-4))
    optimizer = torch.optim.AdamW(groups, weight_decay=0.)
    scale = keep_features.std((0, 2, 3)).clamp_min(1e-3)[None, :, None, None]
    before = metrics(adapter, head, real['val'])
    statuses = []; started = time.monotonic()
    for update, indices in enumerate(batch_order, 1):
        selected = torch.as_tensor(indices, dtype=torch.long)
        prediction = head(adapter(real['train']['features'][selected]), real['train']['state'][selected])
        task = auxiliary_loss(prediction, real['train']['measured'][selected],
                              real['train']['targets'][selected], real['train']['padding'][selected])
        keep_indices = (torch.arange(4) + (update-1)*4) % len(keep_features)
        reference = keep_features[keep_indices]
        preservation = ((adapter(reference) - reference)/scale).square().mean()
        loss = task + preservation
        if not torch.isfinite(loss): raise ValueError('Nonfinite bridge objective')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        parameters = [p for group in groups for p in group['params']]
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 50 == 0:
            status = dict(variant=name, update=update, limit=200, task=float(task.detach()),
                          feature_preservation=float(preservation.detach()), elapsed_s=time.monotonic()-started)
            statuses.append(status); write_json(folder/'status.json', status)
            print(json.dumps(status), flush=True)
    if update != 200: raise ValueError('Exactly200 updates required')
    adapter.eval(); head.eval()
    holdout = metrics(adapter, head, real['val'])
    preservation = assess_preservation(policy, adapter, checks, stats)
    result = dict(updates=update, initial_validation=before, train=metrics(adapter, head, real['train']),
                  validation=holdout, shuffled_validation=metrics(adapter, head, real['val'], shuffle=True),
                  preservation=preservation, trainable_tensor_count=sum(len(g['params']) for g in groups))
    torch.save(dict(profile='real_visual_auxiliary_sidecar_v1', variant=name, adapter=adapter.state_dict(),
                    auxiliary_head=head.state_dict(), initializer_sha256=CHECKPOINT_SHA, actual_updates=200,
                    hardware_deployment_allowed=False), folder/'sidecar_last.pt')
    write_json(folder/'results.json', result); write_json(folder/'training_log.json', statuses)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New experiment directory')
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(24); np.random.seed(24)
    protocol = dict(updates_per_variant=200, variants=['A', 'B'], batch_size=16, seed=24,
                    checkpoint_sha256=CHECKPOINT_SHA, compute_device='cpu', original_config_device='cuda',
                    gate=dict(minimum_validation_gain_fraction=.05, maximum_arm_change_rad=.001,
                              maximum_gripper_change_m=.001, gripper_accuracy_must_not_decrease=True),
                    real_inputs='RGB_and_measured8_only_auxiliary', main_policy_input='unchanged15D',
                    full_flow_evaluation=False, hardware_commands_sent=False)
    write_json(output/'protocol.json', protocol)
    policy, stats = load_baseline()
    original = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    real, real_index = prepare_real(policy)
    keep_features, checks, keep_proof = prepare_preservation(policy, stats)
    torch.save(dict(real=real, keep_features=keep_features, checks=checks), output/'cache.pt')
    write_json(output/'real_sample_index.json', real_index); write_json(output/'preservation_index.json', keep_proof)
    initial_adapter, initial_head = VisualResidual().state_dict(), AuxiliaryHead().state_dict()
    torch.save(dict(adapter=initial_adapter, head=initial_head), output/'initial_modules.pt')
    zero_adapter = VisualResidual(); zero_adapter.load_state_dict(initial_adapter)
    zero = assess_preservation(policy, zero_adapter, checks, stats)
    if zero['maximum_arm_change_rad'] != 0 or zero['maximum_gripper_change_m'] != 0:
        raise ValueError('Zero sidecar did not preserve original15D predictions')
    write_json(output/'zero_identity.json', zero)
    batches = np.random.default_rng(24).integers(0, len(real['train']['state']), size=(200,16))
    np.save(output/'batch_order.npy', batches, allow_pickle=False)
    files = [CHECKPOINT, REAL/'integrity.json', REAL/'adapter.py', REAL/'data/manifest.json', REAL/'data/statistics.json',
             WORKSPACE/'nero_act_command_increment_20261021/model.py',
             WORKSPACE/'docs/superpowers/specs/2026-10-24-nero-real-visual-bridge-design.md',
             WORKSPACE/'docs/superpowers/plans/2026-10-24-nero-real-visual-bridge.md',
             EVENT/'data/manifest.json', EVENT/'preservation/proof.json', EVENT/'preservation/current_camera.npz', output/'protocol.json',
             output/'real_sample_index.json', output/'preservation_index.json', output/'batch_order.npy',
             output/'cache.pt', output/'initial_modules.pt', *ROOT.glob('*.py')]
    files += [Path(r['path']) for r in json.loads((EVENT/'data/manifest.json').read_text())['episodes']]
    files += [Path(m.__file__) for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
              and str(m.__file__).startswith(str(GROUP)) and str(m.__file__).endswith('.py')]
    protected = {str(p.resolve()): digest(p) for p in files}
    results = {}
    for name in ('A','B'):
        results[name] = run_variant(name, policy, stats, real, keep_features, checks, batches,
                                    initial_head, initial_adapter, output)
    changed = [k for k,v in policy.state_dict().items() if not torch.equal(v, original[k])]
    if changed: raise ValueError('Frozen original parameter or buffer changed')
    for path, expected in protected.items():
        if digest(path) != expected: raise ValueError('Protected source or cache changed: '+path)
    decision = acceptance(results['A']['validation'], results['B']['validation'], results['B']['preservation'])
    report = dict(complete=True, protocol=protocol, variants=results, decision=decision,
                  real_train_queries=len(real['train']['state']), real_validation_queries=len(real['val']['state']),
                  rehearsal_queries=len(keep_features), check_queries=sum(len(g['current']) for g in checks),
                  zero_identity_verified=True, original_model_changed_tensors=changed,
                  source_sha256=protected, original_checkpoint_unchanged=digest(CHECKPOINT)==CHECKPOINT_SHA,
                  genuine_15d_real_labels_created=False, full_task_passed=False,
                  hardware_commands_sent=False, hardware_deployment_allowed=False)
    write_json(output/'results.json', report)
    print(json.dumps(dict(complete=True, validation_A=results['A']['validation'],
                         validation_B=results['B']['validation'], decision=decision)), flush=True)


if __name__ == '__main__': main()
