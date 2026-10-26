"""Real ROS service integration in domain43, with a fake SetBool driver and no robot."""
import os,json,time,signal
from pathlib import Path
assert os.environ.get('ROS_DOMAIN_ID')=='43'
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import SetBool,Trigger
from executor.gate import gate_class
r=Path(__file__).resolve().parents[1];output=r/'evidence/isolated_stop_gate.json'
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());before=tx()
args=['--ros-args','-p','arm_joint_names:=[joint1,joint2,joint3,joint4,joint5,joint6,joint7]','-p','status_topics:=[/arm_controller/follow_joint_trajectory/_action/status,/gripper_controller/follow_joint_trajectory/_action/status]','-p','arm_sync_tolerance_rad:=0.001','-p','gripper_sync_tolerance_m:=0.0005','-p','state_timeout_s:=0.15']
rclpy.init(args=args,signal_handler_options=SignalHandlerOptions.NO)
driver=rclpy.create_node('fake_driver_never_CAN');events=[]
def set_gate(request,response):events.append(dict(open=request.data,monotonic_s=time.monotonic()));response.success=True;return response
service=driver.create_service(SetBool,'/control_enable',set_gate)
gate=gate_class()(ttl_s=60);gate.watchdog.cancel();executor=SingleThreadedExecutor();executor.add_node(driver);executor.add_node(gate)
def until(predicate,timeout=3):
 end=time.monotonic()+timeout
 while not predicate():
  if time.monotonic()>=end:raise TimeoutError('Isolated gate RPC timeout')
  executor.spin_once(timeout_sec=.002)
try:
 until(lambda:gate._gate_client.service_is_ready());gate._set_gate(False);until(lambda:gate.latch.pending is None)
 gate._set_gate(True);until(lambda:gate.latch.open);assert events[-1]['open']
 # Stop through the real service, not just a pure latch call.
 client=driver.create_client(Trigger,'/act_task_stop');until(client.service_is_ready)
 future=client.call_async(Trigger.Request());until(future.done);assert future.result().success
 until(lambda:gate.latch.pending is None and not gate.latch.open)
 stopped_count=len(events)
 for _ in range(100):gate._set_gate(True);executor.spin_once(timeout_sec=.001)
 assert not any(e['open'] for e in events[stopped_count:])
 # A SIGINT must leave ROS usable for acknowledged closure.
 try:os.kill(os.getpid(),signal.SIGINT)
 except KeyboardInterrupt:pass
 assert rclpy.ok();gate.latch.stop();gate._set_gate(False);until(lambda:gate.latch.pending is None)
 assert events[-1]['open'] is False and tx()==before
 result=dict(success=True,domain=43,physical_driver_present=False,physical_commands_sent=False,stop_service_passed=True,future_enable_requests_rejected=100,ros_context_valid_after_SIGINT=True,close_acknowledged_after_SIGINT=True,events=events,can_tx_before=before,can_tx_after=tx())
 output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:executor.shutdown();gate.destroy_node();driver.destroy_node();rclpy.try_shutdown()
