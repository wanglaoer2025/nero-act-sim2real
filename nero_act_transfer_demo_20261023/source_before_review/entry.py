"""One reviewed ACT prefix replay; full-task success is not required for this demo."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent


def validate_plan(plan,rows):
    state=np.asarray(plan['initial_state'],float);poses=np.asarray(plan['sdk_flange_poses'],float)
    if (state.shape!=(8,) or rows.shape!=(plan['target_count'],8) or not 20<=len(rows)<=400
            or poses.shape!=(len(rows),6) or not np.isfinite(np.r_[state,rows.ravel(),poses.ravel()]).all()
            or plan['speed_percent']!=5 or plan['period_s']!=.05 or plan['sdk_flange_floor_m']!=.35
            or plan['joint_step_limit_rad']!=.002 or plan['latest_feedback_lead_limit_rad']!=.012
            or plan['tracking_error_limit_rad']!=.012 or plan['complete_task_not_released'] is not True
            or plan['collision_passed'] is not True or plan['ACT_model_qualification_passed'] is not False
            or any(plan[k]!=0 for k in ['gripper_commands','enable_commands','reset_commands','follower_transitions'])):
        raise ValueError('Only the independently reviewed5percent noncontact ACT demo is allowed')
    lower=np.array([-2.70526,-1.74,-2.75,-1.01,-2.75,-.73,-1.5707963])
    upper=np.array([2.70526,1.74,2.75,2.14,2.75,.95,1.5707963])
    if (np.any(rows[:,:7]<lower) or np.any(rows[:,:7]>upper) or np.min(poses[:,2])<.35
            or np.max(abs(np.diff(np.vstack([state,rows]),axis=0)[:,:7]))>.002001
            or np.max(abs(rows[:,7]-state[7]))>1e-8):
        raise ValueError('Arm path left height/rate/joint limits or attempted gripper motion')
    with np.load(plan['source_trace_path'],allow_pickle=False) as source:
        indices=np.asarray(plan['source_indices'])
        if (indices.shape!=(len(rows),) or not np.issubdtype(indices.dtype,np.integer)
                or indices[0]!=0 or np.any(np.diff(indices)<=0)
                or not np.array_equal(rows[:,:7],source['sent_targets'][indices,:7])
                or any(not str(source['target_sources'][i]).startswith('ACT') for i in indices[1:])):
            raise ValueError('ACT source joint targets or provenance changed')
    for path,digest in plan['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Reviewed motion source changed: '+path)


def validate_health(health,plan,reference):
    state=np.asarray(health['state'],float);flange=np.asarray(health['flange_pose'],float)
    feedback=health['feedback']
    if (state.shape!=(8,) or flange.shape!=(6,) or not np.isfinite(np.r_[state,flange]).all()
            or feedback.get('arm_status') not in (0,2) or feedback.get('err_code')!=0
            or feedback.get('ctrl_mode')!=1 or flange[2]<.35
            or np.max(abs(state[:7]-np.asarray(reference)[:7]))>.012
            or abs(state[7]-plan['initial_state'][7])>.001):
        raise ValueError('Current health/tracking/height/gripper guard rejected; stop new commands')


def load_bound_plan(path):
    raw=path.read_bytes();plan=json.loads(raw)
    target_path=Path(plan['targets_path']);target_raw=target_path.read_bytes()
    if hashlib.sha256(target_raw).hexdigest()!=plan['source_sha256'].get(str(target_path)):
        raise ValueError('Loaded target bytes differ from reviewed plan')
    rows=np.load(io.BytesIO(target_raw),allow_pickle=False)
    validate_plan(plan,rows)
    return plan,rows,hashlib.sha256(raw).hexdigest()


def dump(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist())+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--plan',type=Path,default=ROOT/'preparation/plan.json')
    parser.add_argument('--review',type=Path)
    parser.add_argument('--onsite',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();plan,rows,digest=load_bound_plan(args.plan)
    if not args.execute:
        print(json.dumps(dict(plan_valid=True,target_count=len(rows),hardware_connected=False,motion_sent=False)));return
    if None in (args.review,args.onsite,args.output):parser.error('execute requires review, current onsite and new output')
    review=json.loads(args.review.read_text());onsite=json.loads(args.onsite.read_text())
    if (review.get('plan_sha256')!=digest or review.get('approved_high_air_only') is not True
            or review.get('entry_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or onsite.get('plan_sha256')!=digest or not 0<=time.time()-onsite.get('evidence_epoch_s',0)<=1800
            or not all(onsite.get(k) is True for k in ['stop_access','empty_cup','whole_arm_clear','no_other_controller'])):
        raise PermissionError('Missing exact reviewed plan or current onsite facts')
    args.output.mkdir(exist_ok=False)
    import mujoco
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Int32
    sys.path[:0]=['/home/user/zyh/src']
    from nero_pi05.robot import NeroRobot
    from nero_pi05.semantics import JOINT_LIMITS_RAD
    from scipy.spatial.transform import Rotation
    if __import__('os').environ.get('ROS_DOMAIN_ID')!='42':raise ValueError('ROS_DOMAIN_ID42 required')
    xml_path=Path(plan['collision_model_path']);xml_raw=xml_path.read_bytes()
    if hashlib.sha256(xml_raw).hexdigest()!=plan['source_sha256'].get(str(xml_path)):
        raise ValueError('Loaded collision model differs from reviewed bytes')
    model=mujoco.MjModel.from_xml_string(xml_raw.decode());data=mujoco.MjData(model)
    def collision(state):
        data.qpos[:7]=state[:7];data.qpos[7:9]=[state[7]/2,-state[7]/2];mujoco.mj_forward(model,data)
        for c in data.contact[:data.ncon]:
            if c.dist>=0:continue
            names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in [c.geom1,c.geom2]}
            if names in ({'world','base_link'},{'world','cup'}):continue
            raise RuntimeError('Actual full-arm configuration entered inflated obstacle: '+','.join(sorted(names)))
    def interrupted(signum,_frame):raise KeyboardInterrupt('Signal '+str(signum)+'; stop new sends')
    for signum in [signal.SIGINT,signal.SIGTERM]:signal.signal(signum,interrupted)
    tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    report=dict(scope=plan['scope'],plan_sha256=digest,started_epoch_s=time.time(),can_tx_before=tx(),
        arm_attempts=0,arm_returned=0,gripper_commands=0,enable_commands=0,reset_commands=0,
        follower_transitions=0,speed_percent=5,events=[],complete=False,no_automatic_retry_or_return=True,
        transport='Dedicated ROS JointState target and arrival ACK; sole SDK sender is bounded ROS driver',
        cadence='PTP arrival stability before next target; nominal50ms period is a minimum, not verified moving overwrite')
    dump(args.output/'execution.json',report)
    topic='/nero_act_transfer_demo_20261023/target';ack_topic='/nero_act_transfer_demo_20261023/ack'
    class AirDriver(Node):
        def __init__(self):
            super().__init__('nero_reviewed_air_driver')
            self.robot=NeroRobot(allow_no_solution=True);self.index=0;self.failure=None;self.previous=np.array(plan['initial_state'])
            self.ack=self.create_publisher(Int32,ack_topic,1)
            self.create_subscription(JointState,topic,self.receive,1)
        def prepare(self):
            self.robot.connect()
            initial=self.health(self.previous);report['initial_health']=initial
            if initial['feedback'].get('motion_status')!=0 or np.max(abs(np.asarray(initial['state'])-self.previous))>.0005:
                raise RuntimeError('Real start changed or moving; do not rebase')
            if np.any(rows[:,:7]<JOINT_LIMITS_RAD[:,0]) or np.any(rows[:,:7]>JOINT_LIMITS_RAD[:,1]):raise ValueError('SDK hard joint limit')
            for row,expected in zip(rows,plan['sdk_flange_poses']):
                actual=np.asarray(self.robot.sdk.fk(row[:7].tolist()))
                if (np.max(abs(actual[:3]-np.asarray(expected)[:3]))>1e-7 or actual[2]<.35
                        or Rotation.from_matrix(Rotation.from_euler('xyz',actual[3:]).as_matrix().T@
                            Rotation.from_euler('xyz',expected[3:]).as_matrix()).magnitude()>1e-6):
                    raise ValueError('SDK forward kinematics differs from reviewed whole path')
        def health(self,reference):
            value=self.robot.read_health(require_enabled=True);validate_health(value,plan,reference);collision(value['state']);return value
        def receive(self,msg):
            if self.failure is not None:return
            try:
                stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
                expected_names=[f'joint{i}' for i in range(1,8)]
                if (self.index>=len(rows) or list(msg.name)!=expected_names or len(msg.position)!=7
                        or msg.header.frame_id!=f'{digest}:{self.index}' or not 0<=time.time()-stamp<=.25
                        or not np.array_equal(np.asarray(msg.position),rows[self.index,:7])):
                    raise ValueError('ROS driver rejected unplanned, stale, duplicate or gripper-containing command')
                before=self.health(self.previous);target=rows[self.index]
                if before['feedback'].get('motion_status')!=0 or np.max(abs(np.asarray(before['state'])[:7]-target[:7]))>.012:
                    raise ValueError('Target lacks stationary/latest-feedback lead admission')
                if self.index==0:
                    report['speed_setting_attempted']=True;dump(args.output/'execution.json',report)
                    self.robot.sdk.set_speed_percent(5);report['speed_setting_returned']=True
                    before=self.health(self.previous)
                # Only this ROS driver's callback can write a reviewed arm target to CAN.
                event=dict(index=self.index,target_rad=target[:7].tolist(),before=before,send_attempted=True,send_returned=False)
                report['events'].append(event);report['arm_attempts']+=1;dump(args.output/'execution.json',report)
                current=self.health(self.previous)
                event['send_health']=current
                age=time.time()-stamp;lead=float(np.max(abs(np.asarray(current['state'])[:7]-target[:7])))
                event['admission']=dict(command_age_s=age,feedback_lead_rad=lead,motion_status=current['feedback'].get('motion_status'))
                rejected=[]
                if not 0<=age<=.25:rejected.append('stale_command')
                if current['feedback'].get('motion_status')!=0:rejected.append('robot_still_moving')
                if lead>.012:rejected.append('feedback_lead_exceeded')
                if rejected:raise ValueError('ROS pre-send guard: '+','.join(rejected))
                self.robot.sdk.move_j(target[:7].tolist());event['send_returned']=True;report['arm_returned']+=1
                deadline=time.monotonic()+3;stable=None
                while time.monotonic()<deadline:
                    measured=self.health(target);event['after']=measured
                    arrived=np.max(abs(np.asarray(measured['state'])[:7]-target[:7]))<.0007 and measured['feedback'].get('motion_status')==0
                    stable=(stable or time.monotonic()) if arrived else None
                    if stable is not None and time.monotonic()-stable>=.1:break
                    time.sleep(.01)
                else:raise TimeoutError('PTP arrival ACK missing; no retry or overwrite')
                self.previous=target;dump(args.output/'execution.json',report)
                self.ack.publish(Int32(data=self.index));self.index+=1
            except BaseException as exc:
                self.failure=exc;self.robot.stop_sending();report['error']=repr(exc);dump(args.output/'execution.json',report)
        def close(self):
            self.robot.stop_sending();self.robot.disconnect();self.destroy_node()
    driver=client=executor=None
    try:
        old=json.loads((ROOT/'preflight_01/report.json').read_text())
        if tx()!=old['can_tx_after']:raise RuntimeError('CAN writes appeared after readonly preflight; exclusivity uncertain')
        rclpy.init(args=[])
        rclpy.signals.uninstall_signal_handlers()
        for signum in [signal.SIGINT,signal.SIGTERM]:signal.signal(signum,interrupted)
        driver=AirDriver();driver.prepare();client=Node('nero_reviewed_air_client')
        publisher=client.create_publisher(JointState,topic,1);acks=[]
        client.create_subscription(Int32,ack_topic,lambda msg:acks.append(msg.data),1)
        executor=SingleThreadedExecutor();executor.add_node(driver);executor.add_node(client)
        deadline=time.monotonic()+5
        while publisher.get_subscription_count()!=1:
            if time.monotonic()>deadline:raise TimeoutError('Unique ROS trial receiver unavailable')
            executor.spin_once(timeout_sec=.02)
            if driver.failure is not None:raise driver.failure
        for index,target in enumerate(rows):
            if driver.failure is not None:raise driver.failure
            msg=JointState();msg.header.stamp=client.get_clock().now().to_msg();msg.header.frame_id=f'{digest}:{index}'
            msg.name=[f'joint{i}' for i in range(1,8)];msg.position=target[:7].tolist()
            publisher.publish(msg);deadline=time.monotonic()+4
            while index not in acks:
                executor.spin_once(timeout_sec=.01)
                if driver.failure is not None:raise driver.failure
                if time.monotonic()>deadline:raise TimeoutError('ROS arrival ACK timeout; no reissue')
        final=driver.health(rows[-1]);report.update(complete=True,final_health=final,
            actual_joint_change_deg=np.rad2deg(np.asarray(final['state'])[:7]-np.asarray(report['initial_health']['state'])[:7]).tolist())
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        if executor is not None:executor.shutdown()
        if driver is not None:driver.close()
        if client is not None:client.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        report.update(can_tx_after=tx(),finished_epoch_s=time.time(),sender_stopped=True)
        dump(args.output/'execution.json',report)
    print(json.dumps({k:report[k] for k in ['complete','arm_returned','gripper_commands','actual_joint_change_deg','can_tx_before','can_tx_after']}))


if __name__=='__main__':main()
