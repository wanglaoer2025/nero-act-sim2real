"""Read-only on-site ROS stack diagnostics; no launch/enable/motion command.

Captures selected driver/gate/launch/controller model source files so the locally
modified bridge can be reviewed before changing its activation sequence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workspace', type=Path, default=Path('/home/user/projects/nero_ros2_ws'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    commands = [
        ['ros2', 'node', 'list'], ['ros2', 'service', 'list', '-t'],
        ['ros2', 'action', 'list', '-t'], ['ros2', 'topic', 'list', '-t'],
        ['ros2', 'param', 'dump', '/agx_arm_ctrl_single_node'],
        ['ros2', 'param', 'dump', '/control_gate'],
        ['ros2', 'param', 'dump', '/move_group'],
        ['ros2', 'control', 'list_controllers'],
        ['ros2', 'control', 'list_hardware_components'],
        ['ros2', 'topic', 'info', '/feedback/joint_states', '-v'],
        ['ros2', 'topic', 'info', '/control/joint_states', '-v'],
        ['ros2', 'topic', 'info', '/control/gated_joint_states', '-v'],
    ]
    results = []
    for i, command in enumerate(commands):
        print('Reading:', ' '.join(command), flush=True)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=8)
            text = result.stdout+'\n'+result.stderr
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            text, code = str(exc), None
        name = f'{i:02d}_'+command[1]+'.txt'
        (args.output/name).write_text(text)
        results.append(dict(command=command, exit_code=code, file=name))
    selected = []
    source_root = args.workspace/'src/agx_arm_ros/src'
    packages = ['agx_arm_ctrl', 'agx_arm_moveit', 'agx_arm_bringup']
    copied = 0
    # Only source/config extensions in known ROS packages, not environment files,
    # model weights, credentials, arbitrary symlink targets or the Python SDK.
    for package in packages:
        root = source_root/package
        if not root.is_dir(): continue
        for path in sorted(root.rglob('*')):
            if path.is_symlink() or not path.is_file(): continue
            relative = path.relative_to(root)
            if any(p.startswith('.') or p in ('__pycache__', 'test', 'tests') for p in relative.parts): continue
            if path.suffix not in ('.py', '.yaml', '.srdf', '.urdf', '.xacro', '.xml'): continue
            if path.stat().st_size > 200_000: continue
            data = path.read_bytes()
            if copied+len(data) > 3_000_000: break
            target = args.output/'ros_source'/package/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            selected.append(dict(file=str(target.relative_to(args.output)), bytes=len(data),
                                 sha256=hashlib.sha256(data).hexdigest()))
            copied += len(data)
    (args.output/'manifest.json').write_text(json.dumps(dict(
        read_only=True, launch_or_motion_sent=False, workspace=str(args.workspace),
        commands=results, source_files=selected), indent=2))
    print('Saved:', args.output, flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
