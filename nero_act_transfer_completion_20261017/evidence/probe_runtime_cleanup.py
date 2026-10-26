"""Actual ActSession SIGINT/stop ordering with fake driver in isolated domain43."""
from pathlib import Path
import os,sys,json,time,signal,threading
assert os.environ.get('ROS_DOMAIN_ID')=='43'
output_name=sys.argv[1]
sys.argv=[sys.argv[0],'--ros-args','-p','arm_joint_names:=[joint1,joint2,joint3,joint4,joint5,joint6,joint7]','-p','status_topics:=[/arm_controller/follow_joint_trajectory/_action/status,/gripper_controller/follow_joint_trajectory/_action/status]','-p','arm_sync_tolerance_rad:=0.001','-p','gripper_sync_tolerance_m:=0.0005','-p','state_timeout_s:=0.15']
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import SetBool
from executor.ros_execution import session_class
from executor.contact import ContactMonitor
from executor.gate import gate_class
r=Path(__file__).resolve().parents[1];out=r/'evidence'/output_name;out.mkdir(exist_ok=False)
scene=json.loads((r/'model/frozen_run11/scene_config.json').read_text());report={};session=session_class()(out,report,None,scene,ContactMonitor());fake=rclpy.create_node('fake_runtime_cleanup_driver');events=[];gate=gate_class()(ttl_s=60);gate.watchdog.cancel()
def close(request,response):events.append(dict(open=request.data,stop_latched=gate.latch.failed));response.success=True;return response
service=fake.create_service(SetBool,'/control_enable',close)
executor=SingleThreadedExecutor();executor.add_node(fake);executor.add_node(gate)
def run():
 try:executor.spin()
 except rclpy.executors.ExternalShutdownException:pass
thread=threading.Thread(target=run,daemon=True);thread.start()
def until(predicate):
 end=time.monotonic()+3
 while not predicate():
  if time.monotonic()>end:raise TimeoutError('Isolated stop test timeout')
  session.spin(.002)
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());before=tx();result=dict(domain=43,hardware_connected=False,can_tx_before=before)
try:
 until(lambda:session.stop_client.service_is_ready() and session.gate_closer.service_is_ready())
 try:os.kill(os.getpid(),signal.SIGINT)
 except KeyboardInterrupt:pass
 time.sleep(.05);result['ros_alive_after_SIGINT']=rclpy.ok()
 assert rclpy.ok(),'SIGINT shut down ROS before stop/close cleanup'
 session.cancelling=True;future=session.close_control_gate();response=session.wait(future,timeout=1)
 assert response.success and report.get('act_stop_latched') is True
 assert events[-1]==dict(open=False,stop_latched=True)
 result.update(success=True,stop_latched_before_driver_close=True,driver_close_acknowledged=True,events=events)
except BaseException as exc:result.update(success=False,error=f'{type(exc).__name__}: {exc}')
finally:
 executor.shutdown(timeout_sec=2);thread.join(timeout=2);session.node.destroy_node();gate.destroy_node();fake.destroy_node();rclpy.try_shutdown()
 result['can_tx_after']=tx();(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if not result['success']:raise SystemExit(1)
