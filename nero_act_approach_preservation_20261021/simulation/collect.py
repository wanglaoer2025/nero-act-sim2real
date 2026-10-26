"""Generate complete physical simulation demonstrations. Never connects to hardware."""
import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from simulation.teacher import run_teacher
from simulation.scene import build_scene
from executor.contract import FullTaskContract
from simulation.nero_domain_randomization import NeroServo, disturbance_torque
from simulation.snapshot import restore

ROOT = Path(__file__).resolve().parents[1]


def save_records(path, rows, config, result, *, preserve_action_precision=False):
    q = np.asarray([r['qpos'] for r in rows])
    v = np.asarray([r['qvel'] for r in rows])
    actions = np.asarray([r['action'] for r in rows], np.float64 if preserve_action_precision else np.float32)
    stamps = np.asarray([r['timestamp'] for r in rows])
    if (not len(rows) or q.ndim != 2 or q.shape[1] != 16
            or actions.shape != (len(rows), 8) or v.shape != (len(rows), 15)
            or not np.isfinite(np.r_[q.ravel(), v.ravel(), actions.ravel(), stamps]).all()
            or np.any(np.diff(stamps) <= 0)):
        raise ValueError('Invalid same-cycle physical records')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'x') as f:
        task = result.get('task', 'complete_pick_carry_place_retreat')
        f.attrs.update(sim=True, synthetic=True, camera_names='["camera_0"]', camera_name='camera_0',
                       action_dim=8, arm_qpos_dim=7, gripper_state_dim=1, control_hz=20,
                       hardware_deployment_allowed=False, task=task,
                       full_task_verified=bool(result.get('full_task_verified', task == 'complete_pick_carry_place_retreat')),
                       teacher_success=bool(result['success']), scene_config=json.dumps(config),
                       camera_refresh_hz=2, same_cycle_observation_before_action=True)
        f['action'], f['timestamp'] = actions, stamps
        obs = f.create_group('observations')
        obs['qpos'] = q[:, :7].astype(np.float32)
        obs['gripper_state'] = (q[:, 7]-q[:, 8]).astype(np.float32)[:, None]
        obs['object_xyz'] = np.asarray([r['cup_xyz'] for r in rows])
        obs.create_dataset('images/camera_0', data=np.asarray([r['rgb'] for r in rows]),
                           compression='lzf', chunks=(1, 90, 160, 3))
        sim = f.create_group('simulation')
        sim['qpos'], sim['qvel'] = q, v
        sim['stage'] = np.asarray([r['stage'].encode() for r in rows])
        sim['external_arm_torque_nm'] = np.asarray([r['torque'] for r in rows])


def collect_episode(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, renderer, render_data, image = [], None, None, None
    def capture(model, data, action, stage):
        nonlocal renderer, render_data, image
        if renderer is None:
            renderer = mujoco.Renderer(model, height=90, width=160)
            render_data = mujoco.MjData(model)
        if len(rows) % 10 == 0:
            render_data.qpos[:] = data.qpos
            render_data.qvel[:] = data.qvel
            mujoco.mj_forward(model, render_data)
            renderer.update_scene(render_data, camera='calibrated_nero_camera')
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            image = renderer.render().copy()
        cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cup')
        rows.append(dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), action=action.copy(),
                         timestamp=float(data.time), stage=stage, rgb=image.copy(),
                         cup_xyz=data.xpos[cup].copy(), torque=disturbance_torque(config, len(rows))))
    try:
        result = run_teacher(config, capture)
    finally:
        if renderer is not None:
            renderer.close()
    trace = result.pop('trace')
    output.with_suffix('.result.json').write_text(json.dumps(result, indent=2))
    if not result['success']:
        output.with_suffix('.failed_trace.json').write_text(json.dumps(trace))
        raise RuntimeError(f'Physical expert rejected: {result["error"]}; {output.with_suffix(".result.json")}')
    save_records(output, rows, config, result)
    result['prefix_contract_audit'] = verify_action_contract(output)
    return result


def verify_action_contract(path):
    with h5py.File(path, 'r') as f:
        actions = f['action'][:]
        states = np.c_[f['observations/qpos'][:], f['observations/gripper_state'][:]]
        config = json.loads(f.attrs.get('scene_config', '{}'))
    initial_history = config.get('initial_simulation_snapshot', {}).get('previous_target', states[0])
    guard = FullTaskContract(dict(task_scope='complete_pick_carry_place_retreat',
                                  chunk_size=20, state_dim=8, action_dim=8))
    windows = 0
    for index in range(0, len(actions), 10):
        raw = actions[index:index+20]
        raw = np.pad(raw, ((0, 20-len(raw)), (0, 0)), mode='edge')
        protected = guard.validate_prefix(raw, dict(state=states[index], age_s=.05),
                                         initial_history if index == 0 else actions[index-1])
        if np.max(abs(protected-raw[:10])) > 1e-6:
            raise ValueError('Expert targets would be materially changed by the execution contract')
        windows += 1
    if not windows:
        raise ValueError('No actual expert action windows to verify')
    return dict(passed=True, windows=windows)


def verify_replay(path):
    with h5py.File(path, 'r') as f:
        config = json.loads(f.attrs['scene_config'])
        states, velocities, actions = f['simulation/qpos'][:], f['simulation/qvel'][:], f['action'][:]
        torques = f['simulation/external_arm_torque_nm'][:]
    model = build_scene(config)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = states[0], velocities[0]
    mujoco.mj_forward(model, data)
    servo = NeroServo(model, data, config)
    if 'initial_simulation_snapshot' in config:
        restore(model, data, servo, config['initial_simulation_snapshot'])
        if not np.array_equal(data.qpos, states[0]) or not np.array_equal(data.qvel, velocities[0]):
            raise ValueError('Saved correction starts at a different physical snapshot')
    error = 0.
    for state, action, torque in zip(states, actions, torques):
        error = max(error, float(np.max(abs(data.qpos-state))))
        data.qfrc_applied[:7] = torque
        servo.command(action[:7], action[7])
        for _ in range(25):
            servo.apply()
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    if error > 1e-7:
        raise RuntimeError(f'Physical replay diverged: {error}')
    return dict(frames=len(states), maximum_qpos_error=error, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.count <= 12:
        parser.error('count must be in1..12')
    base = json.loads((ROOT/'scene/config.json').read_text())
    manifest = ROOT/'training_data/manifest.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    entries, failed = [], []
    for index in range(args.count):
        config = copy.deepcopy(base)
        rng = np.random.default_rng(71000+index)
        if index:
            for key in ['pick_xy_m', 'place_xy_m']:
                config[key] = (np.asarray(config[key])+rng.uniform(-.003, .003, 2)).tolist()
            config['initial_robot_state']['qpos_rad'] = (
                np.asarray(config['initial_robot_state']['qpos_rad'])+rng.uniform(-.002, .002, 7)).tolist()
        config['appearance']['light_scale'] = .88+.01*index
        if index in [3, 5, 7, 9, 11]:
            config['arm_disturbance'] = dict(seed=72000+index, max_torque_nm=.15,
                                           period_steps=47, duration_steps=3, start_step=90)
        output = ROOT/f'training_data/episode_{index}.hdf5'
        try:
            result = collect_episode(config, output)
            replay = verify_replay(output)
            entries.append(dict(path=str(output), status='valid', seed=71000+index,
                                frames=result['frames'], teacher_success=True, replay=replay))
            print(json.dumps(dict(episode=index, frames=result['frames'], success=True, replay=replay)), flush=True)
        except Exception as exc:
            failed.append(dict(episode=index, error=str(exc)))
            print(json.dumps(failed[-1]), flush=True)
        manifest.write_text(json.dumps(dict(episodes=entries, rejected=failed), indent=2))
        if failed:
            raise RuntimeError('Collection failed; evidence retained, no training started')
    (ROOT/'training_data/split.json').write_text(json.dumps(dict(train=list(range(8)), val=[8, 9], diagnostic=[10, 11]), indent=2))


if __name__ == '__main__':
    main()
