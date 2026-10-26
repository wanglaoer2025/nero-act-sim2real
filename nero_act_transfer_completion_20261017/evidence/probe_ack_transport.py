"""Acknowledgement-only load test in domain43; fake controllers, no SDK or CAN sender."""
import os,sys,time,json
from pathlib import Path
assert os.environ.get('ROS_DOMAIN_ID')=='43'
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
r=Path(__file__).resolve().parents[1];tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());before=tx()
scene=json.loads((r/'model/frozen_run11/scene_config.json').read_text());q=scene['initial_robot_state']['qpos_rad']+[scene['initial_robot_state']['gripper_total_width_m']]
if '--server' in sys.argv:
 from rclpy.action import ActionServer
 rclpy.init();node=rclpy.create_node('fake_ACT_controllers_no_hardware')
 def execute(handle):
  handle.succeed();result=FollowJointTrajectory.Result();result.error_code=0;return result
 servers=[ActionServer(node,FollowJointTrajectory,f'/{group}/follow_joint_trajectory',execute_callback=execute) for group in ['arm_controller','gripper_controller']]
 pubs=[node.create_publisher(JointState,t,10) for t in ['/feedback/joint_states','/control/joint_states']]
 def publish():
  msg=JointState();msg.header.stamp=node.get_clock().now().to_msg();msg.name=[f'joint{i}' for i in range(1,8)]+['gripper'];msg.position=q
  for pub in pubs:pub.publish(msg)
 timer=node.create_timer(.005,publish)
 print('FAKE CONTROLLERS READY; DOMAIN43; NO HARDWARE',flush=True)
 try:rclpy.spin(node)
 except KeyboardInterrupt:pass
 finally:node.destroy_node();rclpy.try_shutdown()
else:
 from executor.ros_execution import session_class
 from executor.contact import ContactMonitor
 out=r/'evidence/isolated_ack_transport';out.mkdir(exist_ok=False);report={}
 session=session_class()(out,report,None,scene,ContactMonitor());success=False;error=None
 try:
  session.spin(1)
  names=[n for n,_ in session.node.get_node_names_and_namespaces()]
  if any(n in names for n in ['agx_arm_ctrl_single_node','control_gate']):raise RuntimeError('Real control route in isolated domain')
  clients=session.clients();rows=np.tile(q,(10,1));completed=0
  for i in range(600):
   futures=session.send_block(clients,rows)
   for future in futures:session.wait(future,timeout=2)
   completed+=1
  success=True
 except BaseException as exc:error=f'{type(exc).__name__}: {exc}'
 finally:
  acknowledgements=report.get('controller_acknowledgements',[]);latencies=[a['latency_ms'] for a in acknowledgements if 'latency_ms' in a]
  result=dict(success=success,error=error,domain=43,physical_hardware_connected=False,actual_task_validation=False,fake_controller_pairs_completed=completed,acknowledgement_records=acknowledgements,maximum_ack_ms=max(latencies,default=None),p99_ack_ms=float(np.percentile(latencies,99)) if latencies else None,deadline_ms=45,late_ack_count=sum(not a.get('on_time',False) for a in acknowledgements),can_tx_before=before,can_tx_after=tx())
  (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='acknowledgement_records'}))
  session.node.destroy_node();rclpy.try_shutdown()
 if not success:raise SystemExit(1)
