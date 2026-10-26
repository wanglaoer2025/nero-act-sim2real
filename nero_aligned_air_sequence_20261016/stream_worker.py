"""Persistent ACT/ZED inference process; no robot commands or CAN access."""
import contextlib
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from trial_contract import air_trial_allowed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'policy_source'))
from src.act.nero_policy import NeroActPolicy
from src.act.nero_dataset import preprocess_images


def main():
    directory = Path(sys.argv[1])
    output = Path(sys.argv[2])
    frames = output/'camera_frames'
    frames.mkdir(exist_ok=False)
    frame_number = 0
    torch.set_num_threads(1)
    bundle = torch.load(directory/'policy.ckpt', map_location='cpu', weights_only=False)
    config = bundle['config']
    if config['state_dim'] != 8 or config['action_dim'] != 8 or config['chunk_size'] != 100 or config['camera_names'] != ['camera_0']:
        raise ValueError('Full runner currently supports explicitly matched single-camera 8-D ACT only')
    if config.get('action_representation', 'absolute_joint_position') != 'absolute_joint_position' or config.get('coordinate_conditioning') or config.get('previous_gripper_conditioning'):
        raise ValueError('Full runner requires absolute arm targets and the explicit feedback-only state profile')
    if config.get('task_scope') != 'aligned_noncontact_air_sequence':
        raise ValueError('checkpoint was not trained for this bounded air-motion task')
    with (directory/'dataset_stats.pkl').open('rb') as f:
        stats = pickle.load(f)
    for name in ['qpos_mean', 'qpos_std', 'action_mean', 'action_std']:
        stats[name] = np.asarray(stats[name], dtype=np.float32)
        if stats[name].shape != (8,) or not np.isfinite(stats[name]).all() or (name.endswith('std') and np.any(stats[name] <= 0)):
            raise ValueError('invalid paired normalization')
    with contextlib.redirect_stdout(sys.stderr):
        model = NeroActPolicy(dict(config, device='cpu', pretrained_backbone=False))
    model.load_state_dict(bundle['model'], strict=True); model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 8), torch.zeros(1, 1, 3, *config['image_size']))
    trial_allowed = air_trial_allowed(bundle)
    print(json.dumps({'type': 'ready', 'hardware_deployment_allowed': bundle.get('hardware_deployment_allowed') is True,
        'experimental_air_trial_allowed': trial_allowed, 'trial_profile': bundle.get('trial_profile'),
        'checkpoint_sha256': hashlib.sha256((directory/'policy.ckpt').read_bytes()).hexdigest()}), flush=True)
    camera = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('stop'):
                break
            # Refuse a full-control query for the supplied shadow-only artifact.
            if not trial_allowed:
                raise RuntimeError('checkpoint has no permission for bounded air-sequence trial')
            if camera is None:
                import pyzed.sl as sl
                camera = sl.Camera()
                init = sl.InitParameters()
                init.camera_resolution = sl.RESOLUTION.HD720; init.camera_fps = 30
                init.depth_mode = sl.DEPTH_MODE.NONE
                if camera.open(init) > sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError('ZED open failed')
                mat = sl.Mat(); runtime = sl.RuntimeParameters()
            if camera.grab(runtime) > sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('camera frame unavailable')
            if camera.retrieve_image(mat, sl.VIEW.LEFT, sl.MEM.CPU, sl.Resolution(640, 360)) > sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('camera image unavailable')
            camera_received = time.monotonic()
            frame = np.asarray(mat.get_data()).copy()[::-1, ::-1]
            image = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB if frame.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
            frame_file = frames/f'{frame_number:04d}.png'
            if not cv2.imwrite(str(frame_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError('could not save actual policy camera frame')
            frame_number += 1
            state = np.asarray(request['state'], dtype=np.float32)
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError('invalid live feedback')
            started = time.monotonic()
            with torch.inference_mode():
                chunk = model(torch.from_numpy((state-stats['qpos_mean'])/stats['qpos_std'])[None],
                              preprocess_images(image[None], config['image_size'])[None])[0].numpy()
            chunk = chunk*stats['action_std']+stats['action_mean']
            if chunk.shape != (100, 8) or not np.isfinite(chunk).all():
                raise ValueError('invalid inference result')
            print(json.dumps({'type': 'prediction', 'chunk': chunk.tolist(),
                'details': {'observation_wall_s': request['observed_monotonic_s'],
                            'camera_received_monotonic_s': camera_received,
                            'feedback_state': request['state'],
                            'camera_frame': str(frame_file.relative_to(output)),
                            'inference_ms': (time.monotonic()-started)*1000}}), flush=True)
    finally:
        if camera is not None:
            camera.close()


if __name__ == '__main__':
    main()
