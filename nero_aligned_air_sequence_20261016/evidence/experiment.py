"""Simulation-only aligned ACT air-motion dataset and closed-loop evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/home/wq/robot_learning')
sys.path.insert(0, str(PROJECT/'src'))
from nero_domain_randomization import NeroServo
from nero_scene_camera import add_calibrated_camera

Q0 = np.array([.18481291449367956, -.12836896648418294, .10992083629060287,
                1.5715293650807343, -.022689280275926284, .2777516971623776, .13554226970987962])
W0 = .060118
DELTA = np.array([.04, .012, .025, -.008, .008, -.020, .015])
PERIOD = .05
HORIZON = 260


def teacher(step):
    # A single non-contact sweep; joint1 progress identifies phase without a timer input.
    phase = np.clip((step*PERIOD-.5)/10., 0, 1)
    progress = .5-.5*np.cos(np.pi*phase)
    return np.r_[Q0+DELTA*progress, W0+.008*np.sin(np.pi*progress)**2]


def model_for(seed):
    rng = np.random.default_rng(seed)
    config = json.loads(Path('/home/wq/sim2real/nero_act_full_runtime_20261016/simulation_initial_state_aligned.json').read_text())
    tree = ET.parse(PROJECT/'assets/nero_mujoco/nero_gripper.xml')
    root = tree.getroot(); world = root.find('worldbody')
    contact = ET.SubElement(root, 'contact')
    for a, b in [('base_link', 'link1'), ('link5', 'link7'), ('link5', 'flange_link')]:
        ET.SubElement(contact, 'exclude', body1=a, body2=b)
    ET.SubElement(world, 'light', pos='-0.4 0.3 1.5', dir='0 0 -1', diffuse='0.8 0.8 0.8')
    ET.SubElement(world, 'geom', name='table', type='box', pos='-0.4 -0.15 -0.035', size='.65 .65 .025', rgba='.9 .9 .9 1')
    add_calibrated_camera(world, config)
    for geom in root.iter('geom'):
        if geom.get('type') == 'mesh':
            color = np.clip(rng.uniform(.14, .3, 3), 0, 1)
            geom.set('rgba', ' '.join(map(str, [*color, 1.])))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    config['action_delay_steps'] = int(rng.integers(0, 9))
    gain = rng.uniform(.85, 1.15)
    config['arm_kp'] = (np.array(config['arm_kp'])*gain).tolist()
    config['arm_kd'] = (np.array(config['arm_kd'])*np.sqrt(gain)).tolist()
    config['finger_force_limit_n'] = 1.
    data = mujoco.MjData(model)
    data.qpos[:7] = Q0+rng.uniform(-.0008, .0008, 7)
    data.qpos[7:9] = [W0/2, -W0/2]
    mujoco.mj_forward(model, data)
    return model, data, config


def state(data):
    return np.r_[data.qpos[:7], data.qpos[7]-data.qpos[8]]


def outcome(states, commands):
    states, commands = np.asarray(states), np.asarray(commands)
    end_error = float(np.max(abs(states[-1, :7]-(Q0+DELTA))))
    progress = float(states[:, 0].max()-Q0[0])
    grip_span = float(np.ptp(states[:, 7]))
    envelope = bool(np.all(np.abs(states[:, :7]-Q0) < .06) and np.all((states[:, 7] >= .052) & (states[:, 7] <= .078)))
    return {'endpoint_max_error_rad': end_error, 'j1_progress_rad': progress,
            'gripper_span_m': grip_span, 'end_gripper_error_m': float(abs(states[-1, 7]-W0)),
            'envelope_passed': envelope,
            'success': bool(envelope and end_error < .006 and progress > .025 and grip_span > .003 and abs(states[-1, 7]-W0) < .003)}


def collect(count):
    directory = ROOT/'data'; directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for episode in range(count):
        model, data, config = model_for(1000+episode)
        servo = NeroServo(model, data, config)
        states, images, actions = [], [], []
        with mujoco.Renderer(model, height=90, width=160) as renderer:
            for step in range(HORIZON):
                states.append(state(data)); target = teacher(step)
                renderer.update_scene(data, camera='calibrated_nero_camera')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                images.append(renderer.render().copy()); actions.append(target)
                servo.command(target[:7], target[7])
                for _ in range(25):
                    servo.apply(); mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
                    raise RuntimeError('MuJoCo numerical failure')
        result = outcome(states, actions)
        if not result['success']:
            raise RuntimeError(f'teacher failed: {result}')
        path = directory/f'episode_{episode}.hdf5'
        with h5py.File(path, 'w') as f:
            values = np.asarray(states, dtype=np.float32)
            f.create_dataset('observations/qpos', data=values[:, :7])
            f.create_dataset('observations/gripper_state', data=values[:, 7:8])
            f.create_dataset('observations/images/camera_0', data=np.asarray(images), compression='gzip', compression_opts=1)
            f.create_dataset('action', data=np.asarray(actions, dtype=np.float32))
            f.create_dataset('timestamp', data=np.arange(HORIZON)*PERIOD)
            f.attrs['scene_config'] = json.dumps(config)
            f.attrs['task'] = 'aligned_noncontact_air_motion; not cup grasp'
        entries.append({'path': str(path), 'status': 'valid', 'teacher_result': result,
                        'num_steps': HORIZON, 'duration_s': HORIZON*PERIOD, 'actual_hz': 20})
        print(json.dumps({'episode': episode, 'result': result}), flush=True)
    (directory/'manifest.json').write_text(json.dumps({'episodes': entries}, indent=2))


def evaluate(checkpoint, seeds, real_image=False, query_stride=10, noise=False, startup_hold_steps=0, inference_offset_steps=0):
    import pickle
    import torch
    sys.path.insert(0, '/home/wq/sim2real/nero_act_source-offline-ready')
    from src.act.nero_policy import NeroActPolicy
    from src.act.nero_dataset import preprocess_images
    torch.set_num_threads(2)
    bundle = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = bundle['config']
    policy = NeroActPolicy(dict(config, device='cpu', pretrained_backbone=False))
    policy.load_state_dict(bundle['model'], strict=True); policy.eval()
    with (checkpoint.parent/'dataset_stats.pkl').open('rb') as f:
        stats = pickle.load(f)
    if real_image:
        with np.load('/home/wq/sim2real/nero_act_formal_transfer_input_20261015_233329/current_capture/observations.npz') as z:
            recorded_image = z['rgb'][0]
    results = []
    for seed in seeds:
        model, data, physical = model_for(seed)
        rng = np.random.default_rng(seed+990000)
        if noise:
            data.qpos[:7] += rng.uniform(-.002, .002, 7)
            physical['action_delay_steps'] = 10
            mujoco.mj_forward(model, data)
        servo = NeroServo(model, data, physical)
        initial_hold = state(data).copy()
        states, commands, raw_predictions = [], [], []
        previous = state(data); chunk = None; failure = None
        with mujoco.Renderer(model, height=90, width=160) as renderer:
            for step in range(HORIZON+80):
                feedback = state(data); states.append(feedback.copy())
                if step % query_stride == 0:
                    renderer.update_scene(data, camera='calibrated_nero_camera')
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    image = recorded_image if real_image else renderer.render()
                    tensor = preprocess_images(image[None], config['image_size'])[None]
                    observed = feedback + (rng.uniform(-.0002, .0002, 8) if noise else 0)
                    with torch.inference_mode():
                        raw = policy(torch.from_numpy(((observed-stats['qpos_mean'])/stats['qpos_std']).astype(np.float32))[None], tensor)[0].numpy()
                    chunk = raw*stats['action_std']+stats['action_mean']
                    raw_predictions.append(chunk.copy())
                # Extra interval represents ROS per-block service/submission overhead.
                requested = chunk[step % query_stride+inference_offset_steps].copy() if step % query_stride < 10 else previous.copy()
                if not np.isfinite(requested).all() or np.max(abs(requested[:7]-feedback[:7])) > .03 or abs(requested[7]-feedback[7]) > .01:
                    failure = 'discontinuous or invalid policy output'; break
                command = requested.copy()
                command[:7] = np.clip(command[:7], previous[:7]-.0075, previous[:7]+.0075)
                command[7] = np.clip(command[7], max(.052, previous[7]-.0005), min(.078, previous[7]+.0005))
                if np.any(command[:7] < model.jnt_range[:7, 0]) or np.any(command[:7] > model.jnt_range[:7, 1]):
                    failure = 'physical joint range'; break
                commands.append(command.copy()); previous = command
                servo.command(command[:7], command[7])
                if step < startup_hold_steps:
                    servo.active = initial_hold.copy()
                for _ in range(25):
                    servo.apply(); mujoco.mj_step(model, data)
        result = outcome(states, commands)
        result.update(seed=seed, static_real_image=real_image, failure=failure,
                      query_stride=query_stride, noise=noise, startup_hold_steps=startup_hold_steps,
                      inference_offset_steps=inference_offset_steps)
        result['success'] &= failure is None
        results.append(result)
        destination = ROOT/'evaluations'
        destination.mkdir(exist_ok=True)
        np.savez_compressed(destination/f'{checkpoint.stem}_{seed}_{int(real_image)}_{query_stride}_{int(noise)}_{startup_hold_steps}_{inference_offset_steps}.npz', state=states, command=commands, predictions=raw_predictions)
        print(json.dumps(result), flush=True)
    report = {'checkpoint': str(checkpoint), 'epoch': bundle['epoch'], 'results': results,
              'successes': sum(r['success'] for r in results), 'cases': len(results),
              'real_hardware_tested': False, 'scope': 'simulated closed loop; static recorded-real-image cases test appearance only'}
    (ROOT/'evaluations'/f'{checkpoint.stem}_{min(seeds)}_{"realimage" if real_image else "simimage"}_{query_stride}_{int(noise)}_{startup_hold_steps}_{inference_offset_steps}.json').write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['collect', 'evaluate'])
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--real-image', action='store_true')
    parser.add_argument('--seed-start', type=int, default=8000)
    parser.add_argument('--query-stride', type=int, default=10)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--startup-hold-steps', type=int, default=0)
    parser.add_argument('--inference-offset-steps', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'collect':
        collect(args.count)
    else:
        evaluate(args.checkpoint, range(args.seed_start, args.seed_start+args.count), args.real_image,
                 args.query_stride, args.noise, args.startup_hold_steps, args.inference_offset_steps)


if __name__ == '__main__':
    main()
