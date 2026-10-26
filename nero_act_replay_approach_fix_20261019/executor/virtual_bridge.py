"""Synchronize isolated mock controllers to receive-only actual feedback; no physical trajectory."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

NAMES = [f'joint{i}' for i in range(1, 8)]+['gripper']


def require_virtual_isolation(inventory):
    if (not inventory['hardware_plugins']
            or any(p != 'mock_components/GenericSystem' for p in inventory['hardware_plugins'])
            or any(inventory[k] != 0 for k in ['driver_nodes', 'gate_nodes', 'gated_publishers',
                                              'gated_subscribers', 'raw_other_subscribers'])):
        raise PermissionError('Virtual synchronization requires mock-only hardware and no physical driver/control route')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inspect', action='store_true')
    mode.add_argument('--synchronize-virtual', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.inspect and not args.synchronize_virtual:
        print(json.dumps(dict(mode='offline_description',hardware_connected=False,physical_motion=False)))
        return
    if os.environ.get('ROS_DOMAIN_ID') != '42' or args.output is None or args.output.exists():
        parser.error('ROS_DOMAIN_ID=42 and a new output file are required')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from controller_manager_msgs.srv import ListHardwareComponents
    rclpy.init(args=[]);node = rclpy.create_node('act_virtual_bridge_initialization')
    latest, report, handles = {}, dict(mode='virtual_mock_preparation', physical_motion=False), []
    health = None;target = None
    tx = lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    before = tx()
    def receive(message):
        if len(set(message.name)) != len(message.name) or len(message.name) != len(message.position):
            raise ValueError('Invalid mock joint state')
        mapping = dict(zip(message.name, message.position))
        q = np.array([mapping[n] for n in NAMES])
        if not np.isfinite(q).all():raise ValueError('Nonfinite mock joint state')
        latest.update(state=q, stamp=time.monotonic())
    subscription = node.create_subscription(JointState, '/control/joint_states', receive, qos_profile_sensor_data)
    def step():
        rclpy.spin_once(node, timeout_sec=.005)
        if health is not None and target is not None:
            if (node.count_publishers('/control/gated_joint_states')
                    or node.count_subscribers('/control/gated_joint_states')
                    or node.count_subscribers('/control/joint_states') != 1):
                raise RuntimeError('A physical control route appeared during virtual preparation')
            actual = np.asarray(health.read_health(require_enabled=True)['state'])
            if (tx() != before or np.max(abs(actual[:7]-target[:7])) > .0007
                    or abs(actual[7]-target[7]) > .0005):
                raise RuntimeError('Actual pose/CAN TX changed during virtual-only preparation')
    def wait(future, timeout=5):
        end = time.monotonic()+timeout
        while not future.done():
            if time.monotonic() >= end:raise TimeoutError('Mock-controller response timeout')
            step()
        if future.exception():raise future.exception()
        return future.result()
    try:
        client = node.create_client(ListHardwareComponents, '/controller_manager/list_hardware_components')
        if not client.wait_for_service(timeout_sec=5):raise RuntimeError('Hardware inventory service unavailable')
        component = wait(client.call_async(ListHardwareComponents.Request())).component
        end = time.monotonic()+1
        while time.monotonic() < end:step()
        def inventory():
            nodes = [name for name, _ in node.get_node_names_and_namespaces()]
            return dict(hardware_plugins=[item.plugin_name for item in component],
                driver_nodes=nodes.count('agx_arm_ctrl_single_node'), gate_nodes=nodes.count('control_gate'),
                gated_publishers=node.count_publishers('/control/gated_joint_states'),
                gated_subscribers=node.count_subscribers('/control/gated_joint_states'),
                raw_other_subscribers=node.count_subscribers('/control/joint_states')-1)
        inv = inventory()
        report['inventory'] = inv
        if args.inspect:
            report['success'] = True;report['control_actions_submitted'] = False
            return
        require_virtual_isolation(inventory())
        if 'state' not in latest or time.monotonic()-latest['stamp'] > .2:
            raise RuntimeError('No fresh mock bridge state')
        sys.path.insert(0, '/home/user/wq/nero_full_task_20261016')
        from task_replay import observer
        health = observer();target = np.asarray(health.read_health(require_enabled=True)['state'])
        report.update(actual_before=target.tolist(), virtual_before=latest['state'].tolist())
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        clients = [ActionClient(node, FollowJointTrajectory, '/'+name+'/follow_joint_trajectory')
                   for name in ['arm_controller', 'gripper_controller']]
        if not all(c.wait_for_server(timeout_sec=3) for c in clients):raise RuntimeError('Mock controllers unavailable')
        component = wait(client.call_async(ListHardwareComponents.Request())).component
        require_virtual_isolation(inventory())
        futures = []
        for c, names, values in [(clients[0], NAMES[:7], target[:7]), (clients[1], ['gripper'], target[7:])]:
            goal = FollowJointTrajectory.Goal();goal.trajectory.joint_names = names
            point = JointTrajectoryPoint(positions=values.tolist());point.time_from_start.sec = 1
            goal.trajectory.points = [point];futures.append(c.send_goal_async(goal))
        report['control_actions_submitted'] = True
        for f in futures:
            handle = wait(f);handles.append(handle)
            if not handle.accepted:raise RuntimeError('Mock synchronization goal rejected')
        for handle in handles:
            result = wait(handle.get_result_async())
            if result.status != 4 or result.result.error_code != 0:raise RuntimeError('Mock goal failed')
        deadline = time.monotonic()+1
        while time.monotonic() < deadline:
            step()
            if max(abs(latest['state']-target)) < .00001:break
        if max(abs(latest['state']-target)) >= .00001:raise RuntimeError('Mock bridge did not match actual target')
        after = np.asarray(health.read_health(require_enabled=True)['state'])
        report.update(success=True, actual_after=after.tolist(), virtual_after=latest['state'].tolist(),
                      can_tx_before=before, can_tx_after=tx(), actual_motion_verified=False,
                      note='Mock initialization only; not an ACT task or a physical motion acceptance')
    except BaseException as exc:
        report.update(success=False,error=f'{type(exc).__name__}: {exc}')
    finally:
        for handle in handles:
            if handle.accepted:
                try:wait(handle.cancel_goal_async(), timeout=1)
                except Exception:pass
        if health is not None:health.disconnect()
        node.destroy_node();rclpy.try_shutdown()
        args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report.get('success'):raise SystemExit(1)


if __name__ == '__main__':main()
