"""Read-only portable artifact verification; no ROS, model, camera or CAN access."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, check_capture=False):
    manifest = json.loads((root / 'package_sha256.json').read_text())
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or unsafe file: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    request = json.loads((root / 'simulation_request.json').read_text())
    info = json.loads((root / 'package_info.json').read_text())
    real = json.loads((root / 'hardware_evidence/act_300_passed/report.json').read_text())
    if not (real['success'] and real['completed_step_count'] == 300 and real['step_target_completed']):
        raise ValueError('300-step evidence does not match the handoff')
    if info['grasp_policy_accepted'] or info['fresh_cup_task_weights_included']:
        raise ValueError('Unexpected claim of a new accepted grasp policy')
    result = {'file_integrity_passed': True, 'verified_files': len(manifest),
              'capture_arrays_checked': False, 'hardware_accessed': False,
              'grasp_policy_accepted': False}
    if check_capture:
        import numpy as np
        with np.load(root / 'scene/current_capture/observations.npz', allow_pickle=False) as data:
            count = info['captured_samples']
            if data['rgb'].shape != (count, 360, 640, 3) or data['rgb'].dtype != np.uint8:
                raise ValueError('Unexpected real RGB shape/dtype')
            if data['state'].shape != (count, 8) or not np.isfinite(data['state']).all():
                raise ValueError('Unexpected real joint/gripper states')
            expected_names = [f'joint{i}' for i in range(1, 8)] + ['gripper']
            if data['joint_names'].tolist() != expected_names:
                raise ValueError('Unexpected joint order')
            if not np.allclose(data['state'][0], request['current_scene']['first_paired_state'], atol=1e-8, rtol=0):
                raise ValueError('Scene request is not paired with the first real observation')
            if not np.all(np.diff(data['sample_monotonic_s']) > 0):
                raise ValueError('Non-increasing observation times')
            ages = data['joint_receipt_age_s']
            if not np.isfinite(ages).all() or np.any(ages < 0) or np.any(ages > .2):
                raise ValueError('Recorded host feedback receipt age outside capture gate')
            result.update(capture_arrays_checked=True, real_samples=count,
                          rgb_shape=list(data['rgb'].shape), state_shape=list(data['state'].shape))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-capture', action='store_true', help='also validate the real arrays using NumPy')
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parent, args.check_capture)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
