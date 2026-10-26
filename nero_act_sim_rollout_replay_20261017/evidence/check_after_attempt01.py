"""Receive-only SDK/ROS feedback provenance check; no control clients or publishers."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('ROS_DOMAIN_ID') != '42':
    raise RuntimeError('Read-only provenance check requires the known ROS domain42')
sys.path.insert(0, '/home/user/wq/nero_full_task_20261016')
from task_replay import observer
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

output = ROOT/'evidence/after_attempt01_feedback.json'
if output.exists():raise FileExistsError(output)
tx_path = Path('/sys/class/net/can0/statistics/tx_packets')
tx_before = int(tx_path.read_text())
report = dict(mode='readonly_feedback_provenance', wall_s=time.time(),
              motion_sent=False, gripper_sent=False, enable_sent=False)
health = node = None
try:
    health = observer()
    rclpy.init(args=[]);node = rclpy.create_node('act_feedback_provenance_receive_only')
    latest, samples = {}, []
    names = [f'joint{i}' for i in range(1, 8)]+['gripper']
    def callback(topic):
        def receive(message):
            mapping = dict(zip(message.name, message.position))
            if len(mapping) != len(message.name):raise ValueError('Duplicate joint names')
            latest[topic] = dict(q=[float(mapping[name]) for name in names],
                received_monotonic_s=time.monotonic(),
                ros_stamp_s=message.header.stamp.sec+message.header.stamp.nanosec*1e-9)
        return receive
    subscriptions = [node.create_subscription(JointState, topic, callback(topic), qos_profile_sensor_data)
                     for topic in ['/feedback/joint_states', '/control/joint_states']]
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:rclpy.spin_once(node, timeout_sec=.02)
    for _ in range(10):
        until = time.monotonic()+.1
        while time.monotonic() < until:rclpy.spin_once(node, timeout_sec=.005)
        sdk = health.read_health(require_enabled=True)
        feedback = latest['/feedback/joint_states']
        age = time.monotonic()-feedback['received_monotonic_s']
        error = abs(np.array(sdk['state'])-np.array(feedback['q']))
        if age > .15 or max(error[:7]) > .003 or error[7] > .002:
            raise RuntimeError('Current receive-only SDK and ROS feedback disagree or are stale')
        samples.append(dict(sdk=np.asarray(sdk['state']).tolist(), ros=feedback['q'], feedback_age_s=age,
                            arm_max_error_rad=float(max(error[:7])), width_error_m=float(error[7])))
    def endpoints(topic):
        return [dict(node_name=info.node_name, node_namespace=info.node_namespace,
                     gid=list(info.endpoint_gid)) for info in node.get_publishers_info_by_topic(topic)]
    report.update(samples=samples, latest=latest,
        feedback_publishers=endpoints('/feedback/joint_states'),
        virtual_publishers=endpoints('/control/joint_states'),
        gated_publishers=node.count_publishers('/control/gated_joint_states'),
        gated_subscribers=node.count_subscribers('/control/gated_joint_states'))
    if len(report['feedback_publishers']) != 1 or report['gated_publishers'] != 0:
        raise RuntimeError('Feedback/control ownership differs from the known read-only inventory')
    report['success'] = True
except BaseException as exc:
    report.update(success=False, error=f'{type(exc).__name__}: {exc}')
finally:
    if health is not None:health.disconnect()
    if node is not None:node.destroy_node();rclpy.try_shutdown()
    report.update(can_tx_before=tx_before, can_tx_after=int(tx_path.read_text()))
    if report['can_tx_before'] != report['can_tx_after']:
        report.update(success=False, tx_changed_during_readonly_check=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
if not report['success']:raise SystemExit(1)
