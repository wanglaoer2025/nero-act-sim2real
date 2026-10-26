"""Explicit single ROS CAN driver. Default only prints the profile, without ROS or CAN."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from executor.bundle import verify_model_bundle
from executor.permission import require_experimental_permission


def main(*,permission_check=None):
    if permission_check is None:permission_check=require_experimental_permission
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--readonly', action='store_true')
    for name in ['model', 'evaluation', 'receipt', 'output']:
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if not args.execute and not args.readonly:
        print(json.dumps(dict(mode='offline_profile',hardware_connected=False,arm='nero',
                              can='can0', auto_enable=False, gate_initially_closed=True)))
        return
    if args.execute:
        if not all([args.model, args.evaluation, args.receipt]):
            parser.error('execute requires model, evaluation, receipt')
        if os.environ.get('ROS_DOMAIN_ID') != '42':
            parser.error('motion profile requires ROS_DOMAIN_ID=42 for single-driver discovery')
        permission_check(verify_model_bundle(args.model),
            json.loads(args.evaluation.read_text()), json.loads(args.receipt.read_text()), now=time.time())
    if args.output is None or args.output.exists():
        parser.error('a new --output record path is required before connecting')
    import rclpy
    rclpy.init(args=[])
    guard = rclpy.create_node('act_driver_start_guard')
    deadline = time.monotonic()+1
    while time.monotonic() < deadline:
        rclpy.spin_once(guard, timeout_sec=.02)
    existing = [name for name, _ in guard.get_node_names_and_namespaces()
                if name in ('agx_arm_ctrl_single_node', 'control_gate')]
    guard.destroy_node()
    if existing:
        rclpy.shutdown()
        raise RuntimeError('Existing driver/gate must be closed before starting a new single-sender profile: '+str(existing))
    sys.path.insert(0, '/home/user/zyh')
    from scripts.nero_methylene_teacher import reject_sdk_send
    from pyAgxArm.protocols.can_protocol.comms.can_comm import CanCommImpl
    from pyAgxArm.protocols.can_protocol.drivers.core.arm_driver_abstract import ArmDriverAbstract
    if args.readonly:
        CanCommImpl.send = ArmDriverAbstract._send_msg = ArmDriverAbstract._send_msgs = reject_sdk_send
    from agx_arm_ctrl.agx_arm_ctrl_single_node import AgxArmRosNode
    if args.readonly:
        AgxArmRosNode._enable_arm = lambda *_args, **_kwargs: False
    sys.argv = ['act_full_driver', '--ros-args', '-p', 'can_port:=can0', '-p', 'arm_type:=nero',
        '-p', 'effector_type:=agx_gripper', '-p', 'fw_version:=v120', '-p', 'auto_enable:=false',
        '-p', 'control_enabled:=false', '-p', 'speed_percent:='+('0' if args.readonly else '5'),
        '-p', 'fast_mode:=false', '-p', 'pub_rate:=200', '-p', 'gripper_default_effort:=1.0',
        '-r', 'control/joint_states:=/control/gated_joint_states']
    tx = lambda: int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    before = tx()
    node = None
    try:
        # Rclpy parameters are parsed during init; use the explicit profile arguments.
        rclpy.shutdown()
        rclpy.init(args=sys.argv[1:])
        node = AgxArmRosNode()
        if args.readonly:
            for method in [node.agx_arm._send_msg, node.agx_arm._send_msgs, node.agx_arm._ctx.get_comm().send]:
                try:method(None)
                except PermissionError:pass
                else:raise RuntimeError('Read-only send guard failed')
            if tx() != before:
                raise RuntimeError('CAN TX changed during read-only restoration')
        record = dict(mode='receive_only' if args.readonly else 'scoped_ACT_single_sender',
                      pid=os.getpid(), interpreter=sys.executable,
                      auto_enable=False, control_enabled_at_start=False,
                      speed_percent_parameter=0 if args.readonly else 5,
                      runtime_send_blocked=bool(args.readonly), can_tx_before=before,
                      can_tx_after_start=tx(), motion_submitted_by_entry=False)
        with args.output.open('x') as stream:json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':main()
