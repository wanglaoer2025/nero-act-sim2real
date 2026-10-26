"""Actual ROS action transport in domain43; synthetic controllers, no robot/driver."""
import json,os,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
assert os.environ.get('ROS_DOMAIN_ID')=='43'
import rclpy
from rclpy.action import ActionClient,ActionServer,GoalResponse,CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty,String
from std_srvs.srv import SetBool
from executor.gate import gate_class
from executor.replay_session import replay_session_class
from executor.replay_trace import reference_at
from executor.contact import ContactMonitor
from executor.runtime import wait_background,json_write
ROOT=Path(__file__).resolve().parents[1];tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());before=tx()
params=['--ros-args','-p','arm_joint_names:=[joint1,joint2,joint3,joint4,joint5,joint6,joint7]','-p','status_topics:=[/arm_controller/follow_joint_trajectory/_action/status,/gripper_controller/follow_joint_trajectory/_action/status]','-p','arm_sync_tolerance_rad:=0.001','-p','gripper_sync_tolerance_m:=0.0005','-p','state_timeout_s:=0.15']
rclpy.init(args=params,signal_handler_options=SignalHandlerOptions.NO)
server=rclpy.create_node('fake_replay_controllers_no_CAN');node=rclpy.create_node('replay_transport_test')
commands=np.array([0,0,0,1.5,0,0,0,.095]);fake=commands.copy();latest_goal=[None,None];goals=[];gate_events=[]
def set_gate(request,response):
    gate_events.append(dict(open=request.data,stamp=time.monotonic()));response.success=True;return response
service=server.create_service(SetBool,'/control_enable',set_gate,callback_group=ReentrantCallbackGroup())
gate=gate_class()(ttl_s=120)
control=server.create_publisher(JointState,'/control/joint_states',10)
feedback=server.create_publisher(JointState,'/feedback/joint_states',10)
def publish():
    fake[:]=commands
    if gate.ever_opened:fake[0]+=.003
    fake[7]=max(fake[7],.083)
    effort=.2 if fake[7]-commands[7]>=.003 else 0.
    for pub,state in [(control,commands),(feedback,fake)]:
        msg=JointState();msg.header.stamp=server.get_clock().now().to_msg()
        msg.name=[f'joint{i}' for i in range(1,8)]+['gripper'];msg.position=state.tolist();msg.effort=[0.]*7+[effort];pub.publish(msg)
timer=server.create_timer(.005,publish,callback_group=ReentrantCallbackGroup())
def execute(group):
    def callback(handle):
        latest_goal[group]=handle;request=handle.request;points=request.trajectory.points
        cols=list(range(7)) if group==0 else [7]
        rows=np.tile(commands,(len(points),1))
        rows[:,cols]=np.array([p.positions for p in points]);initial=commands.copy();start=time.monotonic()
        stamps=np.array([p.time_from_start.sec+p.time_from_start.nanosec*1e-9 for p in points])
        goals.append(dict(group=group,points=len(points),started_monotonic_s=start))
        while time.monotonic()-start<stamps[-1]:
            if latest_goal[group] is not handle:
                handle.abort();return FollowJointTrajectory.Result(error_code=-1)
            if handle.is_cancel_requested:
                handle.canceled();return FollowJointTrajectory.Result(error_code=0)
            elapsed=time.monotonic()-start
            for c in cols:commands[c]=np.interp(elapsed,np.r_[0,stamps],np.r_[initial[c],rows[:,c]])
            time.sleep(.002)
        commands[cols]=rows[-1,cols]
        handle.succeed();return FollowJointTrajectory.Result(error_code=0)
    return callback
servers=[ActionServer(server,FollowJointTrajectory,'/'+name+'/follow_joint_trajectory',execute(i),
    callback_group=ReentrantCallbackGroup(),goal_callback=lambda _:GoalResponse.ACCEPT,
    cancel_callback=lambda _:CancelResponse.ACCEPT) for i,name in enumerate(['arm_controller','gripper_controller'])]
executor=MultiThreadedExecutor(num_threads=4);executor.add_node(server);executor.add_node(gate)
thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
Base=replay_session_class()
class Session(Base):
    def __init__(self):
        self.node=node;self.report={};self.contact=ContactMonitor();self.contact.authorize('close','GRASP',{'aligned':True})
        self.contact_floor_width=None;self.held_center=None;self.last_target=fake.copy();self.phase_arm=self.phase_grip=None
        self.pending_goal_futures=[];self.block_handles=[];self.cancelling=False;self.awaiting_ack=False
        self.heartbeat=node.create_publisher(Empty,'/act_full_heartbeat',10)
    def actual(self):return fake.copy()
    def guard(self):
        now=time.monotonic()
        if hasattr(self,'last_guard'):
            self.report['max_guard_interval_s']=max(self.report.get('max_guard_interval_s',0),now-self.last_guard)
        self.last_guard=now
        self.heartbeat.publish(Empty())
        if self.phase_arm is not None:
            now=time.monotonic();a,g=self.phase_arm,self.phase_grip
            self.last_target=np.r_[reference_at(a['initial'],a['rows'],now-a['start'])[:7],reference_at(g['initial'],g['rows'],now-g['start'])[7]]
        gap=fake[7]-self.last_target[7]
        self.contact.update(width=float(fake[7]),force=.2 if gap>=.003 else 0.,target=float(self.last_target[7]),stamp=time.monotonic())
    def spin(self,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.001);self.guard()
    def validate_rows(self,rows):
        from executor.replay_trace import validate_reference
        validate_reference(rows);self.report.setdefault('checked_reference_points',[]).append(len(rows))
    def make_goal(self,rows,group,hold_s=0.):
        # Deliberately slower than the45ms ACK deadline: construction must precede timing.
        time.sleep(.06);return super().make_goal(rows,group,hold_s)

session=Session();pool=ThreadPoolExecutor(max_workers=1)
clients=[ActionClient(node,FollowJointTrajectory,'/'+name+'/follow_joint_trajectory') for name in ['arm_controller','gripper_controller']]
report=dict(mode='isolated_ROS_action_transport',hardware_connected=False,real_contact_verified=False)
try:
    assert all(c.wait_for_server(timeout_sec=3) for c in clients)
    permission=node.create_client(SetBool,'/act_contact_permission');assert permission.wait_for_service(timeout_sec=3)
    assert session.wait(permission.call_async(SetBool.Request(data=True))).success
    session.spin(.5)
    rows=np.tile(fake,(120,1));rows[:,7]=np.maximum(.065,.095-(np.arange(120)+1)*.0005)
    rows[:,0]=(np.arange(120)+1)*.00001
    session.run_phase('close',rows,clients,pool,wait_background)
    assert session.contact.state=='holding' and session.contact_floor_width is not None
    assert session.report['phases'][0].get('protected_gripper_remainder')
    assert gate.latch.open and not gate.latch.failed
    session.spin(.5)
    assert abs(fake[0]-commands[0])>.001 and abs(fake[0]-commands[0])<.03
    second=np.tile(session.last_target,(4,1))
    session.run_phase('lift',second,clients,pool,wait_background)
    assert gate.latch.open and not gate.latch.failed
    first_open=next(i for i,e in enumerate(gate_events) if e['open'])
    assert not any(not e['open'] for e in gate_events[first_open:])
    acknowledgements=session.report['controller_acknowledgements'];assert len(acknowledgements)==5
    assert all(a['accepted'] and a['on_time'] for a in acknowledgements)
    assert tx()==before
    report.update(success=True,details=session.report,server_goals=goals,gate_events=gate_events,
        strict_sync_tolerance_rad=.001,static_feedback_offset_rad=.003,held_gate_continuation_passed=True)
except BaseException as exc:
    import traceback
    traceback.print_exc()
    report.update(success=False,error=f'{type(exc).__name__}: {exc}',details=session.report,
        gate_events=gate_events,gate_failed=gate.latch.failed,gate_reason=gate.failure_reason)
finally:
    session.cancelling=True;replies=[h.cancel_goal_async() for h in session.block_handles if h.accepted]
    end=time.monotonic()+2
    while time.monotonic()<end and not all(f.done() for f in replies):rclpy.spin_once(node,timeout_sec=.005)
    pool.shutdown(wait=True,cancel_futures=True);executor.shutdown();thread.join(timeout=2)
    for action in servers:action.destroy()
    gate.destroy_node();node.destroy_node();server.destroy_node();rclpy.try_shutdown()
    report.update(can_tx_before=before,can_tx_after=tx())
    json_write(ROOT/'evidence/replay_transport.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='details'}),flush=True)
if not report['success']:raise SystemExit(1)
