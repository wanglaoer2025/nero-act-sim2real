"""One1000-update offline recovery trial; imports no robot/CAN/camera interfaces."""
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

from arm_tuning import ROOT, INCREMENT, arm_prefix, configure_arm_tuning, masked_arm_loss
from pipeline import load_policy, validated_replay
from training.integrity import snapshot, require_unchanged
from training.approach_preservation import ApproachPreservation
from simulation.scene import build_scene
from simulation.visual_evaluation import ImageFeatures
from executor.model import make_policy_observation, physical_actions
from src.act.nero_dataset import preprocess_images
from lift_recovery import collect_lift_prefix


def unchanged_weights(policy):
    digest = hashlib.sha256()
    for name, value in policy.state_dict().items():
        if name.endswith(('model.action_head.weight', 'model.action_head.bias')):
            value = value[7:]
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def recovery_features(policy, stats, scene, checkpoint, source, output):
    episode = np.load(source, allow_pickle=False)
    model = build_scene(scene)
    data = mujoco.MjData(model)
    observer = ImageFeatures(scene, (180, 320))
    values = {k: [] for k in ['hidden', 'current', 'target', 'valid']}
    diagnostics = []
    with mujoco.Renderer(model, height=180, width=320) as renderer:
        for i, (qpos, qvel, stamp) in enumerate(zip(episode['qpos'], episode['qvel'], episode['time_s'])):
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.time = float(stamp)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera='calibrated_nero_camera')
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            rgb = renderer.render().copy()
            state = np.r_[qpos[:7], qpos[7]-qpos[8]].astype(np.float32)
            features, _ = observer(rgb, state)
            previous = (checkpoint['physics']['previous_target'][7] if i == 0 else episode['action'][i-1, 7])
            current = torch.from_numpy(make_policy_observation(state, stats, policy.config,
                                previous, geometry_features_m=features))[None].cuda()
            image = preprocess_images(rgb[None], [180, 320])[None].cuda()
            prediction, hidden = policy.encode(current, image)
            physical = physical_actions(prediction, stats, bounded_gripper=True)[:, :2, :7]
            decoded = arm_prefix(policy, hidden[:, :2], current)
            difference = float((decoded-physical).abs().max())
            if difference > 1e-6:
                raise ValueError('Cached arm decoder differs from actual ACT inference')
            diagnostics.append(difference)
            n = min(2, len(episode['action'])-i)
            target = np.zeros((2, 7), np.float32)
            target[:n] = episode['action'][i:i+n, :7]
            for key, value in dict(hidden=hidden[0, :2], current=current[0],
                    target=torch.as_tensor(target, device='cuda'),
                    valid=torch.arange(2, device='cuda') < n).items():
                values[key].append(value.detach().cpu())
    cache = {k: torch.stack(v) for k, v in values.items()}
    cache['runtime_decoder_maximum_difference_rad'] = max(diagnostics)
    torch.save(cache, output)
    return cache


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    output = ROOT/'training/pilot_A'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source = ROOT/'results/exact_lift_failure'
    scene = json.loads((source/'scene_configuration.json').read_text())
    checkpoint = json.loads((source/'runtime_checkpoints.json').read_text())['records'][-1]
    teacher = ROOT/'training_data/exact_lift.npz'
    recovery = collect_lift_prefix(scene, checkpoint, teacher)
    policy, stats = load_policy()
    frozen_before = unchanged_weights(policy)
    cache = recovery_features(policy, stats, scene, checkpoint, teacher, output/'recovery_features.pt')
    old_seal = json.loads((INCREMENT/'training/pilot_A/seal.json').read_text())
    require_unchanged(old_seal['cache_sha256'])
    old = torch.load(INCREMENT/'training/pilot_A/features.pt', map_location='cpu', weights_only=False)['train']
    keeper = ApproachPreservation(policy.config['approach_preservation'], stats, 'cuda')
    reference_hashes = validated_replay(policy.config['approach_preservation']['replay_path'])
    rehearsal_hidden = []
    for i in range(len(keeper.current)):
        _, h = policy.encode(keeper.current[i:i+1], keeper.image[i:i+1])
        rehearsal_hidden.append(h[0, :2].cpu())
    rehearsal = dict(hidden=torch.stack(rehearsal_hidden).cuda(), current=keeper.current,
                    target=physical_actions(keeper.target, stats, bounded_gripper=True)[:, :2, :7])
    with torch.no_grad():
        baseline = torch.cat([arm_prefix(policy, old['hidden'][i:i+256].cuda(),
                     old['current'][i:i+256].cuda()).cpu() for i in range(0, len(old['current']), 256)])
    source_files = list(ROOT.glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[ROOT/'plan.json']
    source_files += [source/'runtime_checkpoints.json', source/'scene_configuration.json', teacher, teacher.with_suffix('.json'),
                     INCREMENT/'training/pilot_A/policy_last.ckpt', INCREMENT/'training/pilot_A/integrity.json',
                     INCREMENT/'training/pilot_A/features.pt']
    hashes = dict(snapshot(source_files), **reference_hashes)
    caches = dict(snapshot([output/'recovery_features.pt']))
    data = {k: v.cuda() for k, v in cache.items() if torch.is_tensor(v)}
    parameters = configure_arm_tuning(policy)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.)
    rng = np.random.default_rng(42)
    order_recovery = rng.integers(0, len(data['current']), (1000, 32))
    order_old = rng.integers(0, len(old['current']), (1000, 32))
    np.savez(output/'batch_order.npz', recovery=order_recovery, rehearsal_old=order_old)
    traces = []
    started = time.monotonic()
    for step in range(1000):
        optimizer.zero_grad(set_to_none=True)
        indices = torch.as_tensor(order_recovery[step], device='cuda')
        learned = arm_prefix(policy, data['hidden'][indices], data['current'][indices])
        teacher_loss = masked_arm_loss(learned, data['target'][indices], data['valid'][indices])
        old_indices = torch.as_tensor(order_old[step])
        retained = arm_prefix(policy, old['hidden'][old_indices].cuda(), old['current'][old_indices].cuda())
        retention_loss = ((retained-baseline[old_indices].cuda())/.0001).square().mean()
        replay_indices = (torch.arange(8, device='cuda')+step*8) % 423
        preserved = arm_prefix(policy, rehearsal['hidden'][replay_indices], rehearsal['current'][replay_indices])
        approach_loss = ((preserved-rehearsal['target'][replay_indices])/.0001).square().mean()
        loss = teacher_loss+retention_loss+approach_loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite restricted-arm loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, .1)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite arm gradients')
        optimizer.step()
        record = dict(actual_updates=step+1, loss=float(loss.detach()), teacher_loss=float(teacher_loss.detach()),
                      retention_loss=float(retention_loss.detach()), approach_loss=float(approach_loss.detach()))
        traces.append(record)
        if step == 0 or (step+1) % 100 == 0:
            (output/'status.json').write_text(json.dumps(dict(stage='training', **record,
                        budget=1000, hardware_deployment_allowed=False), indent=2)+'\n')
            print(json.dumps(record), flush=True)
    if unchanged_weights(policy) != frozen_before:
        raise RuntimeError('Frozen ACT backbone, gripper or non-arm row changed')
    policy.eval()
    replay = keeper.assess(policy)
    with torch.no_grad():
        learned = arm_prefix(policy, data['hidden'], data['current'])
        valid = data['valid']
        errors = (learned-data['target']).abs()[valid]
    weights = policy.base.model.action_head
    model_path = output/'policy_last.ckpt'
    torch.save(dict(profile='increment_plus_arm_rows_v1', arm_weight=weights.weight[:7].detach().cpu(),
                    arm_bias=weights.bias[:7].detach().cpu(), actual_updates=1000), model_path)
    torch.save(optimizer.state_dict(), output/'optimizer_last.pt')
    (output/'optimizer_trace.json').write_text(json.dumps(traces, indent=2)+'\n')
    require_unchanged(hashes)
    require_unchanged(caches)
    proof = dict(complete=True, actual_updates=1000, source_sha256=hashes,
                 checkpoint_sha256=snapshot([model_path]), hardware_deployment_allowed=False,
                 frozen_non_arm_weights_before=frozen_before, frozen_non_arm_weights_after=unchanged_weights(policy))
    (output/'integrity.json').write_text(json.dumps(proof, indent=2)+'\n')
    summary = dict(complete=True, actual_updates=1000, elapsed_s=time.monotonic()-started, teacher=recovery,
                   runtime_decoder_maximum_difference_rad=cache['runtime_decoder_maximum_difference_rad'],
                   recovery_prefix_max_error_rad=float(errors.max()), recovery_prefix_rms_error_rad=float(errors.square().mean().sqrt()),
                   arm_reference=replay, hardware_deployment_allowed=False, full_task_passed=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'status.json').write_text(json.dumps(dict(stage='evaluation_pending', **summary), indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
