"""Frozen full-task ACT in physical simulation; no expert actions or hardware access."""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

from executor.contract import FullTaskContract
from simulation.scene import build_scene
from simulation.teacher import observe
from simulation.nero_domain_randomization import NeroServo, disturbance_torque
from simulation.outcome import OutcomeMonitor
from simulation.snapshot import capture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'policy_source'))
from executor.model import BoundedActPolicy, physical_actions
from src.act.nero_dataset import preprocess_images


def scenarios(seed_offset=0):
    base = dict(query_stride=10, servo_delay_steps=0, observation_age_s=.05,
                jitter=False, force=False)
    cases = [dict(base, name='nominal', seed=90000)] + [
        dict(base, name=f'independent_{i}', seed=90100+i, query_stride=11, jitter=True)
        for i in range(8)] + [
        dict(base, name='cadence_11', seed=90200, query_stride=11, observation_age_s=.17),
        dict(base, name='delay_100ms', seed=90201, query_stride=11, servo_delay_steps=2),
        dict(base, name='delay_500ms', seed=90202, query_stride=11, servo_delay_steps=10)] + [
        dict(base, name=f'force_{i}', seed=90300+i, query_stride=11, force=True) for i in range(2)]
    return [dict(case, seed=case['seed']+seed_offset) for case in cases]


def rollout(config, model_policy, stats, scenario, output):
    config = copy.deepcopy(config)
    rng = np.random.default_rng(scenario['seed'])
    if scenario['jitter']:
        for key in ['pick_xy_m', 'place_xy_m']:
            config[key] = (np.asarray(config[key])+rng.uniform(-.002, .002, 2)).tolist()
        config['initial_robot_state']['qpos_rad'] = (
            np.asarray(config['initial_robot_state']['qpos_rad'])+rng.uniform(-.002, .002, 7)).tolist()
    config['action_delay_steps'] = scenario['servo_delay_steps']
    if scenario['force']:
        config['arm_disturbance'] = dict(seed=scenario['seed'], max_torque_nm=.15,
                                       period_steps=47, duration_steps=3, start_step=90)
    model = build_scene(config)
    data = mujoco.MjData(model)
    data.qpos[:7] = config['initial_robot_state']['qpos_rad']
    width = config['initial_robot_state']['gripper_total_width_m']
    data.qpos[7:9] = [width/2, -width/2]
    mujoco.mj_forward(model, data)
    servo = NeroServo(model, data, config)
    guard = FullTaskContract(model_policy.config)
    monitor = OutcomeMonitor(config['pick_xy_m'], config['place_xy_m'], config['table_top_z_m'])
    previous = np.r_[data.qpos[:7], data.qpos[7]-data.qpos[8]]
    queries, query_steps, query_states, snapshots, states, targets, measures, errors = [], [], [], [], [], [], [], []
    reach_deadline = model_policy.config.get('reach_deadline_steps', 3200)
    if type(reach_deadline) is not int or not 3200 <= reach_deadline <= 6000:
        raise ValueError('Invalid declared task duration budget')
    render_data = mujoco.MjData(model)
    freeze_step, prefix, contract_rejections = None, None, 0
    started, failure = time.monotonic(), None
    try:
        with mujoco.Renderer(model, height=90, width=160) as renderer:
            for step in range(reach_deadline+101):
                if freeze_step is None and step >= reach_deadline:
                    break
                state = np.r_[data.qpos[:7], data.qpos[7]-data.qpos[8]]
                if freeze_step is None and step % scenario['query_stride'] == 0:
                    render_data.qpos[:], render_data.qvel[:] = data.qpos, data.qvel
                    mujoco.mj_forward(model, render_data)
                    renderer.update_scene(render_data, camera='calibrated_nero_camera')
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    image = renderer.render().copy()
                    observed = state.astype(np.float32)
                    if scenario['jitter']:
                        observed[:7] += rng.uniform(-.0002, .0002, 7)
                    with torch.inference_mode():
                        tensor = model_policy(
                            torch.from_numpy((observed-stats['qpos_mean'])/stats['qpos_std'])[None].to(model_policy.device),
                            preprocess_images(image[None], (90, 160))[None].to(model_policy.device))
                    raw = physical_actions(tensor[0], stats,
                                           bounded_gripper=model_policy.bounded_gripper).cpu().numpy()
                    queries.append(raw.copy())
                    query_steps.append(step)
                    query_states.append(state.copy())
                    snapshots.append(capture(model, data, servo, previous))
                    try:
                        prefix = guard.validate_prefix(raw, dict(state=state, age_s=scenario['observation_age_s']), previous)
                    except ValueError:
                        contract_rejections += 1
                        raise
                index = step % scenario['query_stride']
                target = prefix[index].copy() if freeze_step is None and index < 10 else previous.copy()
                states.append(state.copy())
                targets.append(target.copy())
                data.qfrc_applied[:7] = disturbance_torque(config, step)
                servo.command(target[:7], target[7])
                bad = False
                for _ in range(25):
                    servo.apply()
                    mujoco.mj_step(model, data)
                    if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():
                        raise RuntimeError('Unstable physical rollout')
                    bad |= observe(model, data, config)['unexpected_contact']
                mujoco.mj_forward(model, data)
                measured = observe(model, data, config)
                measured['unexpected_contact'] |= bad
                monitor.update(measured)
                measures.append(measured)
                previous = target.copy()
                if bad:
                    raise RuntimeError('Unexpected robot/environment/cup contact')
                # Task completion only stops future queries; no expert target is generated.
                if freeze_step is None and monitor.carried and monitor.stable_released >= 1:
                    freeze_step = step
                if freeze_step is not None and step-freeze_step >= 100:
                    break
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        errors.append(failure)
    result = dict(monitor.result(), scenario=scenario, frames=len(states), queries=len(queries),
                  error=failure, teacher_free=True, real_hardware_tested=False,
                  post_completion_freeze_step=freeze_step, runtime_s=time.monotonic()-started)
    result['execution_contract_failures'] += contract_rejections
    result['success'] &= failure is None
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'rollout.npz', states=np.asarray(states), sent_targets=np.asarray(targets),
                        raw_chunks=np.asarray(queries), query_steps=query_steps,
                        feedback_at_query=np.asarray(query_states))
    (output/'measurements.json').write_text(json.dumps(measures))
    (output/'physical_snapshots.json').write_text(json.dumps(dict(query_steps=query_steps, snapshots=snapshots)))
    (output/'scene_configuration.json').write_text(json.dumps(config, indent=2))
    (output/'result.json').write_text(json.dumps(result, indent=2))
    return result


def evaluation_scene(config):
    return copy.deepcopy(config['scene_config']) if 'scene_config' in config else json.loads((ROOT/'scene/config.json').read_text())


def evaluation_sources():
    names = ['executor/model.py', 'executor/contract.py', 'simulation/evaluate.py',
        'simulation/scene.py', 'simulation/teacher.py', 'simulation/outcome.py', 'simulation/snapshot.py',
        'simulation/nero_domain_randomization.py', 'simulation/nero_scene_camera.py',
        'simulation/scene_template.xml', 'scene/config.json']
    return sorted([ROOT/name for name in names]+list((ROOT/'policy_source').rglob('*.py'))+
                  [p for p in (ROOT/'simulation/assets').rglob('*') if p.is_file()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--seed-offset', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    bundle = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = dict(bundle['config'], device='cuda', pretrained_backbone=False)
    FullTaskContract(config)
    with (args.checkpoint.parent/'dataset_stats.pkl').open('rb') as f:
        stats = pickle.load(f)
    torch.set_num_threads(2)
    policy = BoundedActPolicy(config).cuda()
    policy.load_state_dict(bundle['model'], strict=True)
    policy.eval()
    policy.config, policy.device = config, torch.device('cuda')
    scene_config = evaluation_scene(config)
    cases = scenarios(args.seed_offset)[:1] if args.dev else scenarios(args.seed_offset)
    source_paths = evaluation_sources()
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol = dict(checkpoint_sha256=checkpoint_hash, selected_epoch=bundle['epoch'],
                    task_scope=config['task_scope'], source_sha256=source_hashes,
                    scenarios=cases, teacher_free=True, real_hardware_tested=False,
                    reach_deadline_steps=config.get('reach_deadline_steps', 3200), post_completion_hold_steps=100,
                    observation_policy='fresh rendered RGB and measured8D state per query',
                    hardware_deployment_allowed=False)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    results = []
    for case in cases:
        result = rollout(scene_config, policy, stats, case, args.output/case['name'])
        results.append(result)
        summary = dict(protocol, results=results, complete=len(results) == len(cases),
                       passed=sum(r['success'] for r in results), total=len(cases))
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != 'last_observation'}), flush=True)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during evaluation; results are not frozen-model evidence')
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != value for name, value in source_hashes.items()):
        raise RuntimeError('Evaluation source changed during the frozen protocol')


if __name__ == '__main__':
    main()
