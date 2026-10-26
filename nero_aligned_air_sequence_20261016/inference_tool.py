"""Capture RGB or run ACT inference only. No robot SDK, ROS or CAN connection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


def capture(output):
    import cv2
    import pyzed.sl as sl
    camera = sl.Camera()
    try:
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.NONE
        status = camera.open(init)
        if status > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f'ZED open failed: {status}')
        image = sl.Mat()
        runtime = sl.RuntimeParameters()
        for _ in range(30):
            if camera.grab(runtime) <= sl.ERROR_CODE.SUCCESS:
                status = camera.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, sl.Resolution(640, 360))
                if status <= sl.ERROR_CODE.SUCCESS:
                    break
        else:
            raise RuntimeError('ZED frame unavailable')
        received = time.monotonic()
        frame = np.asarray(image.get_data()).copy()[::-1, ::-1]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB if frame.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
        np.save(output/'rgb.npy', rgb)
        cv2.imwrite(str(output/'camera_0_upright.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        (output/'camera.json').write_text(json.dumps({'host_monotonic_s': received,
            'upright_rgb': True, 'rotation_applied_once_degrees': 180,
            'motion_sent': False, 'hardware_synchronized': False}, indent=2))
    finally:
        camera.close()


def infer(output):
    import pickle
    import torch
    sys.path.insert(0, str(ROOT/'policy_source'))
    from src.act.nero_policy import NeroActPolicy
    from src.act.nero_dataset import preprocess_images
    torch.set_num_threads(1)
    bundle = torch.load(ROOT/'model/policy.ckpt', map_location='cpu', weights_only=False)
    config = bundle['config']
    if config['state_dim'] != 8 or config['action_dim'] != 8 or config['camera_names'] != ['camera_0']:
        raise ValueError('This response probe requires the bundled 8-D single-camera simulation ACT')
    state = np.asarray(json.loads((output/'input_state.json').read_text())['state'], dtype=np.float32)
    image = np.load(output/'rgb.npy', allow_pickle=False)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError('invalid state')
    if image.shape != (360, 640, 3) or image.dtype != np.uint8:
        raise ValueError('invalid upright RGB')
    with (ROOT/'model/dataset_stats.pkl').open('rb') as f:
        stats = pickle.load(f)
    for key in ['qpos_mean', 'qpos_std', 'action_mean', 'action_std']:
        values = np.asarray(stats[key], dtype=np.float32)
        if values.shape != (8,) or not np.isfinite(values).all() or (key.endswith('std') and np.any(values <= 0)):
            raise ValueError(f'invalid statistics: {key}')
        stats[key] = values
    policy = NeroActPolicy(dict(config, device='cpu', pretrained_backbone=False))
    policy.load_state_dict(bundle['model'], strict=True)
    policy.eval()
    before = time.monotonic()
    with torch.inference_mode():
        raw = policy(torch.from_numpy((state-stats['qpos_mean'])/stats['qpos_std'])[None],
                     preprocess_images(image[None], config['image_size'])[None])[0].numpy()
    raw = raw*stats['action_std']+stats['action_mean']
    if raw.shape != (100, 8) or not np.isfinite(raw).all():
        raise ValueError('invalid ACT predictions')
    np.save(output/'raw_act_chunk.npy', raw)
    bounds = json.loads((ROOT/'model/safety_summary.json').read_text())
    lower = np.asarray(bounds['train_qpos_min'])-.15
    upper = np.asarray(bounds['train_qpos_max'])+.15
    outside = np.flatnonzero((state[:7] < lower) | (state[:7] > upper))
    result = {'model': 'aligned noncontact air ACT 20261016', 'epoch': bundle.get('epoch'),
              'checkpoint_sha256': hashlib.sha256((ROOT/'model/policy.ckpt').read_bytes()).hexdigest(),
              'checkpoint_hardware_deployment_allowed': bundle.get('hardware_deployment_allowed'),
              'state': state.tolist(), 'raw_first10': raw[:10].tolist(),
              'outside_demonstration_joints': [f'joint{i+1}' for i in outside],
              'forward_ms': (time.monotonic()-before)*1000,
              'motion_sent': False, 'raw_policy_execution_approved': False,
              'scope': 'model inference only; raw targets must not be directly executed'}
    (output/'prediction.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture', 'infer', 'check'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'check':
        import cv2
        import torch
        import pyzed.sl
        print(json.dumps({'environment_imports_ok': True, 'motion_sent': False}))
        return
    if args.output is None:
        parser.error('--output required')
    if args.mode == 'capture':
        capture(args.output)
    else:
        print(json.dumps(infer(args.output), indent=2))


if __name__ == '__main__':
    main()
