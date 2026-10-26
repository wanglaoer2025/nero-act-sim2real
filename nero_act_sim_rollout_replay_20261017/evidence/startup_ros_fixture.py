"""Separate-process synthetic ROS feedback/driver gate in domain43; no SDK/CAN."""
import json,os,signal,sys,time
from pathlib import Path
assert os.environ.get('ROS_DOMAIN_ID')=='43'
state=json.loads(Path(sys.argv[1]).read_text());sys.argv=sys.argv[:1]
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from executor.gate import gate_class
args=['--ros-args','-p','arm_joint_names:=[joint1,joint2,joint3,joint4,joint5,joint6,joint7]','-p','status_topics:=[/arm_controller/follow_joint_trajectory/_action/status,/gripper_controller/follow_joint_trajectory/_action/status]','-p','arm_sync_tolerance_rad:=0.001','-p','gripper_sync_tolerance_m:=0.0005','-p','state_timeout_s:=0.15']
rclpy.init(args=args,signal_handler_options=SignalHandlerOptions.NO)
node=rclpy.create_node('isolated_feedback_fixture_no_CAN');gate=gate_class()(ttl_s=120)
pubs=[node.create_publisher(JointState,t,10) for t in ['/feedback/joint_states','/control/joint_states']]
def publish():
    msg=JointState();msg.header.stamp=node.get_clock().now().to_msg();msg.name=[f'joint{i}' for i in range(1,8)]+['gripper'];msg.position=state['q'];msg.effort=[0.]*7+[state['force']]
    for pub in pubs:pub.publish(msg)
def set_gate(request,response):
    if request.data:raise RuntimeError('No enable allowed without any trajectory in fixture')
    response.success=True;return response
service=node.create_service(SetBool,'/control_enable',set_gate);timer=node.create_timer(.005,publish)
executor=SingleThreadedExecutor();executor.add_node(node);executor.add_node(gate)
try:
    deadline=time.monotonic()+110
    while time.monotonic()<deadline:executor.spin_once(timeout_sec=.002)
except KeyboardInterrupt:pass
finally:
    gate.latch.stop();gate._set_gate(False);deadline=time.monotonic()+.2
    while gate.latch.pending is not None and time.monotonic()<deadline:executor.spin_once(timeout_sec=.002)
    executor.shutdown();gate.destroy_node();node.destroy_node();rclpy.try_shutdown()
