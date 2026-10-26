"""Offline RGB8 future-feedback view; no hardware imports or command semantics."""
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch
from torch.nn import functional as F


PROFILE = 'real_rgb8_future_feedback_v1'
SEMANTICS = 'future_measured_qpos_plus_recorded_gripper_intent_v1'
MISSING = ['paired_live_visual_geometry', 'confirmed_previous_gripper_command',
           'submitted_arm_commands', 'device_exposure_timestamps',
           'raw_depth_xyz_confidence', 'verified_calibration_binding']


def file_snapshot(path):
    st = Path(path).stat()
    return {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def _numeric_hash(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _source_data(source):
    snapshot = file_snapshot(source)
    with h5py.File(source, 'r') as f:
        if (f.attrs.get('sim', True) != False
                or f.attrs.get('demonstration_mode') != 'yolo_teacher_policy'
                or f.attrs.get('action_semantics') != 'future_measured_qpos_plus_gripper_target'
                or f.attrs.get('image_format') != 'RGB'):
            raise ValueError('Explicit real teacher feedback-label/RGB source required')
        required = ['observations/qpos', 'action', 'timestamp', 'observations/gripper_state']
        if any(k not in f for k in required) or 'observations/images' not in f:
            raise ValueError('Source is missing recorded arrays')
        q, action, time, grip = [f[k][:] for k in required]
        n = len(time)
        if (not n or q.shape != (n, 7) or action.shape != (n, 8)
                or grip.shape != (n, 1) or time.shape != (n,)
                or not all(np.issubdtype(x.dtype, np.floating) and np.isfinite(x).all()
                           for x in (q, action, time, grip))
                or np.any(np.diff(time) <= 0)):
            raise ValueError('Source arrays must be aligned, finite and time-ordered')
        lead = f.attrs.get('action_lead_steps')
        if (isinstance(lead, (bool, np.bool_)) or not isinstance(lead, (int, np.integer))
                or not 0 <= lead < n
                or not np.array_equal(action[:, :7], q[np.minimum(np.arange(n) + lead, n - 1)])):
            raise ValueError('Source feedback labels do not match their declared lead')
        cameras = {name: ds.shape for name, ds in f['observations/images'].items()}
        if not cameras or any(len(shape) != 4 or shape[0] != n or shape[-1] != 3
                              or min(shape) <= 0 or f['observations/images/' + name].dtype != np.uint8
                              for name, shape in cameras.items()):
            raise ValueError('Complete RGB uint8 camera streams required')
        original = {k: (v.item() if isinstance(v, np.generic) else v)
                    for k, v in f.attrs.items()}
    if file_snapshot(source) != snapshot:
        raise ValueError('Source changed while being read')
    return q, action, time, grip, cameras, original, snapshot


def convert_episode(source, destination, lead_steps=2, split='reserve'):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if split not in ('train', 'val', 'reserve'):
        raise ValueError('Explicit train/val/reserve split required')
    if isinstance(lead_steps, bool) or not isinstance(lead_steps, int) or lead_steps < 1:
        raise ValueError('Positive integer future-feedback lead required')
    q, old_action, time, grip, cameras, original, snapshot = _source_data(source)
    n = len(time) - lead_steps
    if n <= 0:
        raise ValueError('Source is shorter than the requested future lead')
    actions = np.c_[q[lead_steps:], old_action[:n, 7]].astype(np.float32)
    source_id = source.parent.name + '/' + source.name
    metadata = dict(profile=PROFILE, action_semantics=SEMANTICS, sim=False,
                    image_format='RGB', action_dim=8, state_dim=8, policy_input_dim=8,
                    split=split, source_id=source_id, source_path=str(source),
                    source_size=snapshot['size'], source_mtime_ns=snapshot['mtime_ns'],
                    source_numeric_sha256=_numeric_hash((q, old_action, time, grip)),
                    original_action_lead_steps=int(original['action_lead_steps']),
                    action_lead_steps=lead_steps, gripper_action_lead_steps=0,
                    camera_names=json.dumps(list(cameras)),
                    time_profile='host_recorded_not_exposure_synchronized',
                    task_success='not_independently_reviewed',
                    current_visual15_compatible=False, deployment_authorized=False,
                    missing_features=json.dumps(MISSING),
                    original_metadata=json.dumps(original, ensure_ascii=False, default=str))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete file; hard-link publication refuses concurrent overwrite.
    with tempfile.TemporaryDirectory(prefix='.convert-', dir=destination.parent) as tmp:
        staged = Path(tmp) / 'episode.hdf5'
        with h5py.File(staged, 'w') as f:
            f.attrs.update(metadata)
            f['action'] = actions
            f['timestamp'] = time[:n]
            f['joint_target_feedback_timestamp'] = time[lead_steps:]
            f['gripper_intent_timestamp'] = time[:n]
            f['observations/qpos'] = q[:n].astype(np.float32)
            f['observations/gripper_state'] = grip[:n].astype(np.float32)
            f['full_chunk_query_valid'] = np.arange(n) < n - 19
            images = f.create_group('observations/images')
            for name, shape in cameras.items():
                layout = h5py.VirtualLayout(shape=(n, *shape[1:]), dtype=np.uint8)
                layout[:] = h5py.VirtualSource(str(source), 'observations/images/' + name, shape=shape)[:n]
                images.create_virtual_dataset(name, layout)
        if file_snapshot(source) != snapshot:
            raise ValueError('Source changed during conversion')
        os.link(staged, destination)
    return dict(path=str(destination), source_path=str(source), source_id=source_id,
                split=split, frames=n, full_chunk_queries=max(0, n - 19),
                lead_steps=lead_steps, source_snapshot=snapshot)


def verify_source(path, *, numeric=False):
    with h5py.File(path, 'r') as f:
        if f.attrs.get('profile') != PROFILE or f.attrs.get('action_semantics') != SEMANTICS:
            raise ValueError('Explicit offline RGB8 future-feedback profile required')
        source = Path(f.attrs['source_path'])
        expected = {'size': int(f.attrs['source_size']), 'mtime_ns': int(f.attrs['source_mtime_ns'])}
        digest = str(f.attrs['source_numeric_sha256'])
    if not source.is_file():
        raise FileNotFoundError('Source file for virtual images is missing: ' + str(source))
    if file_snapshot(source) != expected:
        raise ValueError('Source file changed after conversion: ' + str(source))
    if numeric:
        q, action, time, grip, _, _, _ = _source_data(source)
        if _numeric_hash((q, action, time, grip)) != digest:
            raise ValueError('Source numeric fingerprint mismatch')
    return source


def _statistics_arrays(stats):
    arrays = [np.asarray(stats[k], np.float32) for k in
              ('qpos_mean', 'qpos_std', 'action_mean', 'action_std')]
    if (any(x.shape != (8,) or not np.isfinite(x).all() for x in arrays)
            or np.any(arrays[1] <= 0) or np.any(arrays[3] <= 0)):
        raise ValueError('Paired finite8D statistics with positive scales required')
    return arrays


def fit_statistics(train_paths):
    if not train_paths:
        raise ValueError('Nonempty explicit train files required')
    states, actions, seen = [], [], set()
    for path in train_paths:
        source = verify_source(path, numeric=True)
        if source in seen:
            raise ValueError('Duplicate source in train statistics')
        seen.add(source)
        with h5py.File(path, 'r') as f:
            if f.attrs['split'] != 'train':
                raise ValueError('Statistics may use train files only')
            states.append(np.c_[f['observations/qpos'][:], f['observations/gripper_state'][:]])
            actions.append(f['action'][:])
    stats = {}
    for name, values in [('qpos', states), ('action', actions)]:
        values = np.concatenate(values).astype(np.float64)
        stats[name + '_mean'] = values.mean(0).astype(np.float32)
        stats[name + '_std'] = np.maximum(values.std(0), 1e-2).astype(np.float32)
    _statistics_arrays(stats)
    return stats


def read_window(path, start, stats, image_size=(180, 320), camera_names=('camera_0',)):
    verify_source(path)
    mean, std, action_mean, action_std = _statistics_arrays(stats)
    if (len(image_size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in image_size)
            or not camera_names or len(set(camera_names)) != len(camera_names)):
        raise ValueError('Positive image size and unique cameras required')
    with h5py.File(path, 'r') as f:
        n = len(f['action'])
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or not 0 <= start < n:
            raise IndexError('Observation index outside converted episode')
        if any('observations/images/' + name not in f for name in camera_names):
            raise ValueError('Requested RGB camera is missing')
        images = np.stack([f['observations/images/' + name][start] for name in camera_names])
        state = np.r_[f['observations/qpos'][start], f['observations/gripper_state'][start]]
        targets = f['action'][start:start + 20]
    verify_source(path)
    padded = np.zeros((20, 8), np.float32)
    padded[:len(targets)] = (targets - action_mean) / action_std
    mask = np.arange(20) >= len(targets)
    image = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
    image = F.interpolate(image, size=tuple(image_size), mode='bilinear', align_corners=False)
    return image, torch.from_numpy((state - mean) / std), torch.from_numpy(padded), torch.from_numpy(mask), int(start)


class FutureStateDataset(torch.utils.data.Dataset):
    """ACT's image/state/action/padding/start interface, without virtual fill fallback."""

    def __init__(self, paths, stats, *, policy_input_dim=8, stride=4,
                 image_size=(180, 320), camera_names=('camera_0',)):
        if policy_input_dim != 8:
            raise ValueError('This RGB8 view cannot supply current visual15 submitted-target observations')
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError('Positive integer query stride required')
        _statistics_arrays(stats)
        self.paths = [Path(p) for p in paths]
        self.stats, self.stride = stats, stride
        self.image_size, self.camera_names = image_size, camera_names
        self.ends, splits, sources, total = [], set(), set(), 0
        for path in self.paths:
            source = verify_source(path, numeric=True)
            if source in sources:
                raise ValueError('Duplicate source in dataset')
            sources.add(source)
            with h5py.File(path, 'r') as f:
                splits.add(str(f.attrs['split']))
                queries = max(0, len(f['action']) - 19)
            total += len(range(0, queries, stride))
            self.ends.append(total)
        if len(splits) != 1 or not total:
            raise ValueError('Nonempty single-split dataset required')

    def __len__(self):
        return self.ends[-1]

    def sample_identity(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = bisect_right(self.ends, index)
        offset = self.ends[file_index - 1] if file_index else 0
        return self.paths[file_index], (index - offset) * self.stride

    def __getitem__(self, index):
        path, start = self.sample_identity(index)
        return read_window(path, start, self.stats, self.image_size, self.camera_names)
