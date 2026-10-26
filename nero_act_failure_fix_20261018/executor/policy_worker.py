"""Sealed full-task ACT inference only; no ROS, robot SDK or CAN imports."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

from executor.model import BoundedActPolicy, physical_actions, verify_paired_readout_statistics,make_policy_observation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'policy_source'))
from src.act.nero_dataset import preprocess_images


from executor.bundle import verify_model_bundle


class FrozenAct:
    def __init__(self, directory, *, device='cpu'):
        directory = Path(directory)
        self.metadata = verify_model_bundle(directory)
        bundle = torch.load(directory/'policy.ckpt', map_location='cpu', weights_only=False)
        self.config = dict(bundle['config'], device=device, pretrained_backbone=False)
        if not self.config.get('bounded_gripper_output'):
            raise ValueError('Complete-task bundle needs the trained bounded gripper head')
        with (directory/'dataset_stats.pkl').open('rb') as f:
            self.stats = pickle.load(f)
        verify_paired_readout_statistics(self.config, self.stats)
        for name in ['qpos_mean', 'qpos_std', 'action_mean', 'action_std']:
            value = np.asarray(self.stats[name], np.float32)
            if value.shape != (8,) or not np.isfinite(value).all() or (name.endswith('std') and np.any(value <= 0)):
                raise ValueError('Invalid paired normalization')
        if (not np.isclose(self.config['gripper_action_mean'], self.stats['action_mean'][7])
                or not np.isclose(self.config['gripper_action_std'], self.stats['action_std'][7])):
            raise ValueError('Gripper head does not match normalization')
        torch.set_num_threads(1)
        self.model = BoundedActPolicy(self.config).to(device)
        self.model.load_state_dict(bundle['model'], strict=True)
        self.model.eval()
        self.device = device
        with torch.inference_mode():
            self.model(torch.zeros(1, self.config.get('policy_input_dim',self.config['state_dim']), device=device), torch.zeros(1, 1, 3, 90, 160, device=device))

    def predict(self, rgb, state,previous_gripper_target_m=None):
        rgb, state = np.asarray(rgb), np.asarray(state, np.float32)
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or state.shape != (8,) or not np.isfinite(state).all()):
            raise ValueError('Expected upright RGB uint8 and finite measured state[8]')
        with torch.inference_mode():
            normalized = self.model(
                torch.from_numpy(make_policy_observation(state,self.stats,self.config,previous_gripper_target_m))[None].to(self.device),
                preprocess_images(rgb[None], (90, 160))[None].to(self.device))
            raw = physical_actions(normalized[0], self.stats, bounded_gripper=True).cpu().numpy()
        if raw.shape != (20, 8) or not np.isfinite(raw).all():
            raise ValueError('Invalid complete-task prediction')
        return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    if args.stream == (args.capture is not None):
        parser.error('Choose exactly one of --capture and --stream')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = None
    if args.stream:
        protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    policy = FrozenAct(args.model)
    if not args.stream:
        obs = np.load(args.capture/'observations.npz', allow_pickle=False)
        records, chunks = [], []
        for index in sorted(set([0, len(obs['state'])//2, len(obs['state'])-1])):
            started = time.monotonic()
            previous=(float(obs['previous_gripper_target_m'][index]) if 'previous_gripper_target_m' in obs else None)
            raw = policy.predict(obs['rgb'][index], obs['state'][index],previous)
            chunks.append(raw)
            records.append(dict(index=index, state=obs['state'][index].tolist(), inference_ms=(time.monotonic()-started)*1000))
        np.save(args.output/'raw_chunks.npy', np.asarray(chunks))
        (args.output/'report.json').write_text(json.dumps(dict(mode='saved_real_observation_shadow',
            checkpoint_sha256=policy.metadata['checkpoint_sha256'], predictions=records,
            motion_sent=False, gripper_sent=False, live_execution_validated=False), indent=2))
        return
    import cv2
    import pyzed.sl as sl
    camera = sl.Camera()
    try:
        init = sl.InitParameters()
        init.camera_resolution, init.camera_fps, init.depth_mode = sl.RESOLUTION.HD720, 30, sl.DEPTH_MODE.NONE
        if camera.open(init) > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('Cannot open live ZED')
        mat, runtime = sl.Mat(), sl.RuntimeParameters()
        print(json.dumps(dict(type='ready', checkpoint_sha256=policy.metadata['checkpoint_sha256'],
                              action_chunk_shape=[20, 8], hardware_commands_possible=False)), file=protocol, flush=True)
        for index, line in enumerate(sys.stdin):
            request = json.loads(line)
            if request.get('stop'):
                break
            if camera.grab(runtime) > sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('Fresh camera frame unavailable')
            if camera.retrieve_image(mat, sl.VIEW.LEFT, sl.MEM.CPU, sl.Resolution(640, 360)) > sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('Camera retrieve failed')
            stamp = time.monotonic()
            frame = np.asarray(mat.get_data()).copy()[::-1, ::-1]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB if frame.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
            state = request['state']
            started = time.monotonic()
            previous=request.get('previous_gripper_target_m')
            raw = policy.predict(rgb, state,previous)
            image_path = args.output/f'camera_{index:05d}.png'
            if not cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Cannot preserve actual inference image')
            print(json.dumps(dict(type='prediction', chunk=raw.tolist(), details=dict(
                observed_monotonic_s=request['observed_monotonic_s'], camera_received_monotonic_s=stamp,
                inference_ms=(time.monotonic()-started)*1000, camera_frame=str(image_path),
                feedback_state=state,previous_gripper_target_m=previous))), file=protocol, flush=True)
    finally:
        camera.close()
        protocol.close()


if __name__ == '__main__':
    main()
