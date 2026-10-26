"""Real feedback timing only: no heartbeat publisher, action client or SDK sends."""
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

from executor.health_trace import HealthTrace
from executor.runtime import SceneObserver, json_write
from executor.ros_execution import session_class

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/readonly_health_soak'
OUT.mkdir(exist_ok=False)
tx = lambda: int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
before = tx()
report = dict(mode='receive_only_health_timing', motion_sent=False, heartbeat_published=False,
              duration_requested_s=210, success=False, freshness_limit_s=.15)
health = node = None
records, feedback, gc_events = HealthTrace(), [], []
active_gc = {}
def gc_callback(phase, info):
    generation = info['generation']
    if phase == 'start':
        active_gc[generation] = time.monotonic()
    elif generation in active_gc:
        start = active_gc.pop(generation)
        gc_events.append(dict(generation=generation, start=start, duration_s=time.monotonic()-start))

try:
    # Match the heavy replay imports/perception initialization using a saved image.
    scene = json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text())
    sensor = SceneObserver(scene, OUT)
    saved = ROOT.parent/'nero_act_replay_current_cup_retest_20261019/evidence/retry_readonly_02/camera/camera_00008.png'
    sensor.measure(dict(camera_frame=str(saved), camera_received_monotonic_s=time.monotonic()))
    session_class()  # Import the real session classes; never construct one.
    sys.path.insert(0, '/home/user/wq/nero_full_task_20261016')
    from task_replay import observer
    health = observer()
    import rclpy
    from sensor_msgs.msg import JointState
    rclpy.init()
    node = rclpy.create_node('act_receive_only_health_timing')
    names = [f'joint{i}' for i in range(1, 8)]+['gripper']
    latest = {}
    def receive(msg):
        index = {name: i for i, name in enumerate(msg.name)}
        if all(name in index for name in names):
            stamp = msg.header.stamp
            row = dict(monotonic_s=time.monotonic(), ros_stamp_s=stamp.sec+stamp.nanosec*1e-9,
                       position=[msg.position[index[name]] for name in names])
            feedback.append(row)
            latest.update(row)
    sub = node.create_subscription(JointState, '/feedback/joint_states', receive, 10)
    deadline = time.monotonic()+5
    while not latest and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.005)
    if not latest or node.count_publishers('/control/gated_joint_states'):
        raise RuntimeError('Fresh receive-only route required')
    initial = np.asarray(health.read_state()['state']).copy()
    gc.collect()
    gc.callbacks.append(gc_callback)
    started = last = time.monotonic()
    gaps, read_times, log_times = [], [], []
    last_progress = started
    while time.monotonic()-started < 210:
        rclpy.spin_once(node, timeout_sec=.002)
        now = time.monotonic()
        if now-last < .01:
            continue
        read_start = time.monotonic()
        measured = health.read_health(require_enabled=True)
        read_times.append(time.monotonic()-read_start)
        actual = np.asarray(latest['position'])
        sdk = np.asarray(measured['state'])
        if time.monotonic()-latest['monotonic_s'] > .15 or np.max(abs(sdk-actual)) > .003:
            raise RuntimeError('Physical ROS/SDK feedback stale or mismatched')
        if abs(measured['gripper_details']['force_n']) > 1.25:
            raise RuntimeError('Unexpected force in receive-only test')
        log_start = time.monotonic()
        records.append(measured)
        log_times.append(time.monotonic()-log_start)
        completed = time.monotonic()
        gaps.append(completed-last)
        last = completed
        if completed-last_progress >= 30:
            print(json.dumps(dict(elapsed_s=completed-started, samples=len(records),
                                  max_health_gap_s=max(gaps))), flush=True)
            last_progress = completed
    final = np.asarray(health.read_state()['state'])
    report.update(elapsed_s=time.monotonic()-started, samples=len(records), feedback_samples=len(feedback),
                  maximum_health_gap_s=max(gaps), maximum_health_read_s=max(read_times),
                  maximum_log_snapshot_s=max(log_times), gc_events=gc_events,
                  maximum_state_change=float(abs(final-initial).max()),
                  no_health_gap_over_150ms=bool(max(gaps) <= .15))
    report['success'] = bool(max(gaps) <= .15 and max(abs(final-initial)) <= .001)
except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
finally:
    if gc_callback in gc.callbacks:
        gc.callbacks.remove(gc_callback)
    if health is not None:
        health.disconnect()
    if node is not None:
        node.destroy_node()
        rclpy.try_shutdown()
    report.update(can_tx_before=before, can_tx_after=tx())
    report['success'] = report['success'] and report['can_tx_after'] == before
    json_write(OUT/'independent_health.json', records)
    json_write(OUT/'report.json', report)
    print(json.dumps(report), flush=True)
