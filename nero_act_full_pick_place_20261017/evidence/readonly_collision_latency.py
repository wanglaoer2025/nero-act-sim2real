"""Read-only ROS service timing at measured pose, not a task collision acceptance."""
import copy
import argparse
import json
import os
from pathlib import Path
import time

if os.environ.get('ROS_DOMAIN_ID') != '42':raise RuntimeError('Expected domain42')
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPlanningScene, GetStateValidity

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=root/'evidence/readonly_collision_latency.json')
parser.add_argument('--persistent-executor',action='store_true')
parser.add_argument('--max-in-flight',type=int,default=0);args=parser.parse_args()
output=args.output
if output.exists():raise FileExistsError(output)
tx = lambda: int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
before = tx();rclpy.init(args=[]);node = rclpy.create_node('act_readonly_collision_service_timing')
latest = {};report = dict(mode='readonly_same_pose_service_timing', hardware_commands_sent=False,
                         full_ACT_path_checked=False, cup_scene_acceptance=False,
                         persistent_executor=args.persistent_executor)
report['max_in_flight']=args.max_in_flight
executor=None
if args.persistent_executor:
    from rclpy.executors import SingleThreadedExecutor
    executor=SingleThreadedExecutor();executor.add_node(node)
def spin(seconds):
    if executor is not None:executor.spin_once(timeout_sec=seconds)
    else:rclpy.spin_once(node, timeout_sec=seconds)
def receive(message):latest.update(message=message, receipt=time.monotonic())
subscription = node.create_subscription(JointState, '/feedback/joint_states', receive, qos_profile_sensor_data)
def wait(future, timeout=5):
    end = time.monotonic()+timeout
    while not future.done():
        if time.monotonic() >= end:raise TimeoutError('Read-only ROS service timeout')
        spin(.002)
    if future.exception():raise future.exception()
    return future.result()
try:
    scene_client = node.create_client(GetPlanningScene, '/get_planning_scene')
    validity = node.create_client(GetStateValidity, '/check_state_validity')
    if not all(c.wait_for_service(timeout_sec=5) for c in [scene_client, validity]):
        raise RuntimeError('Read-only planning services unavailable')
    end = time.monotonic()+1
    while time.monotonic() < end:spin(.01)
    if node.count_publishers('/control/gated_joint_states') != 0:
        raise RuntimeError('Unexpected active gated publisher')
    request = GetPlanningScene.Request();request.components.components = 1023
    started = time.monotonic();scene = wait(scene_client.call_async(request)).scene
    report['planning_scene_roundtrip_ms'] = (time.monotonic()-started)*1000
    state = copy.deepcopy(scene.robot_state);state.is_diff = False
    rows = []
    report['batches']=rows
    for count in [1, 10, 40, 40, 40]:
        report['current_batch_count']=count
        if not latest or time.monotonic()-latest['receipt'] > .15:
            raise RuntimeError('Missing fresh measured pose')
        state.joint_state = copy.deepcopy(latest['message'])
        state.joint_state.velocity=[];state.joint_state.effort=[]
        started = time.monotonic();results=[]
        size=args.max_in_flight or count
        for offset in range(0,count,size):
            futures=[]
            for _ in range(min(size,count-offset)):
                item = GetStateValidity.Request();item.group_name='';item.robot_state=copy.deepcopy(state)
                futures.append(validity.call_async(item))
            results.extend(wait(f) for f in futures)
        rows.append(dict(count=count, roundtrip_ms=(time.monotonic()-started)*1000,
                         all_valid=all(r.valid for r in results)))
    report.update(success=True, batches=rows, measured_joint_names=list(state.joint_state.name),
                  measured_position=list(state.joint_state.position))
except BaseException as exc:
    report.update(success=False,error=f'{type(exc).__name__}: {exc}')
finally:
    if executor is not None:executor.shutdown()
    node.destroy_node();rclpy.try_shutdown()
    report.update(can_tx_before=before, can_tx_after=tx())
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if not report['success']:raise SystemExit(1)
