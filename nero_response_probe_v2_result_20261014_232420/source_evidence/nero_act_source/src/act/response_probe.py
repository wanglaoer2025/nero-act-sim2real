"""Small ramp/hold/return probe with fresh SDK feedback and physical tracking checks.

No policy, camera or gripper commands. Default mode is completely offline.
"""
import argparse
import json
import math
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

STATUS_FIELDS = ('ctrl_mode', 'arm_status', 'mode_feedback', 'teach_status',
                 'motion_status', 'trajectory_num', 'err_code')
AMPLITUDE = .005
TOLERANCE = .001
PERIOD = .05


class FreshReader:
    """Require SDK timestamp advancement, without assuming its clock domain."""
    def __init__(self, robot, now=time.monotonic, sleep=time.sleep):
        self.robot, self.now, self.sleep = robot, now, sleep
        self.previous = {'joint': None, 'status': None}

    def read(self):
        deadline = self.now() + .15
        pending = {}
        while self.now() < deadline:
            for name, getter in (('joint', self.robot.get_joint_angles),
                                 ('status', self.robot.get_arm_status)):
                if name in pending:
                    continue
                value = getter()
                if value is None:
                    continue
                stamp = getattr(value, 'timestamp', None)
                if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                    raise ValueError(f'{name}: missing/nonfinite SDK timestamp')
                previous = self.previous[name]
                if previous is not None and stamp <= previous:
                    continue
                if name == 'joint':
                    q = np.asarray(value.msg, dtype=float).copy()
                    if q.shape != (7,) or not np.isfinite(q).all():
                        raise ValueError('invalid joint feedback')
                    payload = q.tolist()
                else:
                    payload = {key: int(getattr(value.msg, key)) for key in STATUS_FIELDS}
                    if payload['arm_status'] != 0 or payload['err_code'] != 0:
                        raise RuntimeError(f'controller fault: {payload}')
                    if payload['ctrl_mode'] != 1 or payload['mode_feedback'] != 1:
                        raise RuntimeError(f'controller is not CAN/MOVE_J: {payload}')
                pending[name] = (stamp, payload, self.now())
            if len(pending) == 2:
                for name in pending:
                    self.previous[name] = pending[name][0]
                return {'q': pending['joint'][1], 'joint_sdk_timestamp': pending['joint'][0],
                        'status_sdk_timestamp': pending['status'][0], 'status': pending['status'][1],
                        'host_monotonic_s': self.now(),
                        'pair_host_gap_s': abs(pending['joint'][2] - pending['status'][2])}
            self.sleep(.001)
        raise TimeoutError('joint/status SDK timestamp did not advance within 150 ms')


class PassiveCAN:
    """Receive raw frames including other local sockets' loopback; never send."""
    def __init__(self, path):
        self.path = path
        self.error = None
        self.count = 0
        self.done = threading.Event()

    def __enter__(self):
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            self.sock.bind(('can0',))
            self.sock.settimeout(.1)
            self.file = self.path.open('x')
        except BaseException:
            self.sock.close()
            raise
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()
        return self

    def receive(self):
        try:
            while not self.done.is_set():
                try:
                    data, _, flags, _ = self.sock.recvmsg(16)
                except socket.timeout:
                    continue
                can_id, dlc, payload = struct.unpack('=IB3x8s', data)
                self.file.write(json.dumps({'host_monotonic_s': time.monotonic(),
                    'can_id_with_flags': can_id, 'dlc': dlc, 'data_hex': payload[:dlc].hex(),
                    'local_tx_loopback': bool(flags & socket.MSG_DONTROUTE)}) + '\n')
                self.count += 1
        except BaseException as exc:
            self.error = repr(exc)

    def __exit__(self, *unused):
        self.done.set()
        self.thread.join(timeout=2)
        self.sock.close()
        self.file.close()


def tracking_result(rows, q0, joint):
    endpoint = np.asarray(q0, dtype=float).copy()
    endpoint[joint] += AMPLITUDE
    holds = [row for row in rows if row['phase'] == 'hold']
    returns = [row for row in rows if row['phase'] == 'return']
    late = holds[-5:]
    progress = max((row['feedback']['q'][joint] - q0[joint] for row in holds), default=0.)
    outbound = len(late) == 5 and progress >= .004 and all(
        np.max(np.abs(np.asarray(row['feedback']['q']) - endpoint)) <= TOLERANCE for row in late)
    returned = len(returns) >= 5 and all(
        np.max(np.abs(np.asarray(row['feedback']['q']) - q0)) <= TOLERANCE for row in returns[-5:])
    return {'outbound_tracking_passed': bool(outbound), 'return_tracking_passed': bool(returned),
            'positive_progress_rad': float(progress), 'passed': bool(outbound and returned)}


def run_probe(wrapper, reader, rows, joint, limits, observer=None,
              now=time.monotonic, sleep=time.sleep):
    # Discard the first sample: require an observed timestamp update before motion.
    reader.read()
    baseline = reader.read()
    q0 = np.asarray(baseline['q'], dtype=float)
    endpoint = q0.copy()
    endpoint[joint] += AMPLITUDE
    if np.any(q0 < limits[:, 0]) or np.any(q0 > limits[:, 1]) or np.any(endpoint > limits[:, 1]):
        raise ValueError('current pose or endpoint outside joint limits')
    rows.append({'phase': 'baseline', 'command': None, 'feedback': baseline})
    origin = now()
    for phase, steps in (('ramp', 20), ('hold', 40), ('return', 60)):
        if phase == 'return' and not tracking_result(rows, q0, joint)['outbound_tracking_passed']:
            raise RuntimeError('outbound tracking not demonstrated; no automatic return target')
        for step in range(steps):
            if observer is not None and observer.error:
                raise RuntimeError(f'passive CAN capture failed: {observer.error}')
            start = now()
            if phase == 'ramp':
                fraction = (step + 1) / steps
                target = q0 + (endpoint - q0) * (.5 - .5 * math.cos(math.pi * fraction))
            elif phase == 'hold':
                target = endpoint.copy()
            else:
                fraction = min((step + 1) / 20, 1.)
                target = endpoint + (q0 - endpoint) * (.5 - .5 * math.cos(math.pi * fraction))
            feedback = reader.read()
            actual = np.asarray(feedback['q'])
            if np.max(np.abs(actual - q0)) > .015 or np.max(np.abs(target - actual)) > .01:
                raise RuntimeError('unexpected joint excursion or target lead')
            row = {'phase': phase, 'step': step, 'elapsed_s': now() - origin,
                   'command': target.tolist(), 'feedback': feedback, 'command_sent': False}
            rows.append(row)
            command_start = now()
            wrapper.send_arm_qpos(target)
            row['command_sent'] = True
            row['send_call_s'] = now() - command_start
            # Never burst commands to catch up after a slow read/send.
            sleep(max(0., PERIOD - (now() - start)))
    result = tracking_result(rows, q0, joint)
    if not result['passed']:
        raise RuntimeError(f'tracking acceptance failed: {result}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--joint-index', type=int, choices=range(7), default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = {'joint_index': args.joint_index, 'amplitude_rad': AMPLITUDE, 'speed_percent': 5,
            'nominal_hz': 20, 'ramp_s': 1, 'hold_s': 2, 'return_ramp_and_observation_s': 3,
            'tracking_tolerance_rad': TOLERANCE, 'motion_sent': False,
            'policy_used': False, 'freshness_scope': 'SDK timestamp advancement; not device age'}
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return 0
    if not Path('/sys/class/net/can0').exists():
        parser.error('can0 absent')
    from src.utils.arm import confirm_action
    if not confirm_action('关节响应复测',
        '确保SDK独占CAN、ROS驱动已停止、空夹爪悬空；最多6秒目标发送。'
        '异常保持当前位置，严重故障/中断电子急停；速度5%不自动恢复。', 'RESPONSE'):
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(plan, passed=False, error=None, motion_sent=False)
    rows = []
    wrapper = None
    observer = None
    observer_started = False
    from src.act.robot_wrapper import NeroRobotWrapper
    try:
        summary = json.loads((Path(__file__).parents[3] / 'joint_limits.json').read_text())
        limits = np.asarray(summary, dtype=float)
        if limits.shape != (7, 2) or not np.isfinite(limits).all():
            raise ValueError('invalid joint limits')
        observer = PassiveCAN(args.output / 'can_frames.jsonl')
        observer.__enter__()
        observer_started = True
        wrapper = NeroRobotWrapper()
        wrapper.connect()
        wrapper.set_speed_percent(5)
        result = run_probe(wrapper, FreshReader(wrapper.robot), rows, args.joint_index, limits, observer)
        report.update(result)
        if observer.error or observer.count == 0:
            raise RuntimeError(f'CAN evidence absent/incomplete: {observer.error}')
    except KeyboardInterrupt:
        report['error'] = 'operator interruption'
        report['passed'] = False
        if wrapper is not None and wrapper.connected:
            try:
                wrapper.emergency_stop()
            except Exception as stop_error:
                report['stop_error'] = repr(stop_error)
    except Exception as exc:
        report['error'] = repr(exc)
        report['passed'] = False
        if wrapper is not None and wrapper.connected:
            try:
                if isinstance(exc, (TimeoutError, ValueError)) or 'controller fault' in str(exc) or 'excursion' in str(exc):
                    wrapper.emergency_stop()
                    report['stop_action'] = 'electronic_emergency_stop'
                else:
                    hold_reader = FreshReader(wrapper.robot)
                    hold_reader.read()
                    hold = hold_reader.read()
                    wrapper.send_arm_qpos(hold['q'])
                    report['stop_action'] = 'hold_current_position'
                    report['stop_feedback'] = hold
            except Exception as stop_error:
                report['stop_error'] = repr(stop_error)
                try:
                    wrapper.emergency_stop()
                except Exception as emergency_error:
                    report['emergency_stop_error'] = repr(emergency_error)
    finally:
        if wrapper is not None and wrapper.connected:
            try:
                wrapper.disconnect()
            except Exception as disconnect_error:
                report['disconnect_error'] = repr(disconnect_error)
                report['passed'] = False
        if observer_started:
            try:
                observer.__exit__()
            except Exception as capture_error:
                report['capture_close_error'] = repr(capture_error)
                report['passed'] = False
            if observer.error:
                report['capture_error'] = observer.error
                report['passed'] = False
        report['motion_sent'] = any(row.get('command_sent') for row in rows)
        report['can_frames'] = 0 if observer is None else observer.count
        times = [row['feedback']['host_monotonic_s'] for row in rows if row.get('command_sent')]
        intervals = np.diff(times)
        report['observed_loop_hz'] = float(1 / np.mean(intervals)) if len(intervals) else None
        report['loop_period_p95_s'] = float(np.percentile(intervals, 95)) if len(intervals) else None
        report['ready_for_v9_execution'] = False
        (args.output / 'trace.json').write_text(json.dumps(rows, indent=2) + '\n')
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
