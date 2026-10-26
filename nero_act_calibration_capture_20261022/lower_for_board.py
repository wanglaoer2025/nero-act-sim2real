"""One reviewed2–20mm calibration framing motion; no policy or cup contact."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from capture import dump, flange_transform, install_cleanup_signals, finalize_session


def validate_plan(plan):
    drop=plan.get('drop_m',.020)
    if (isinstance(drop,bool) or not isinstance(drop,(int,float)) or not np.isfinite(drop)
            or not .002<=drop<=.020):
        raise ValueError('Framing distance must be2–20mm')
    start=np.asarray(plan['initial_state'],float)
    rows=np.asarray(plan['waypoints_rad'],float)
    poses=np.asarray(plan['flange_poses_fk'],float)
    if (start.shape!=(8,) or rows.ndim!=2 or rows.shape[1]!=7 or not 1<=len(rows)<=26
            or poses.shape!=(len(rows),6) or not np.isfinite(np.r_[start,rows.ravel(),poses.ravel()]).all()
            or plan['speed_percent']!=1 or plan['gripper_commands']!=0
            or not all(plan[k] is True for k in ['no_enable','no_reset','no_follower_transition'])):
        raise ValueError('Reviewed arm-only1percent framing plan required')
    if np.max(abs(np.diff(np.vstack([start[:7],rows]),axis=0)))>.006001:
        raise ValueError('Framing joint step exceeds0.006rad')
    if np.max(abs(rows-start[:7]))>.15:
        raise ValueError('Framing total joint change exceeds0.15rad')
    first=flange_transform(plan['initial_flange_fk'])
    for pose in poses:
        T=flange_transform(pose)
        if (np.max(abs(T[:2,3]-first[:2,3]))>.0001 or not first[2,3]-drop-.0001<=T[2,3]<=first[2,3]+.0001
                or T[2,3]<.4 or Rotation.from_matrix(first[:3,:3].T@T[:3,:3]).magnitude()>.001):
            raise ValueError('Framing path must remain vertical and preserve orientation above0.4m')
    if abs(poses[-1,2]-(first[2,3]-drop))>.0001 or np.any(np.diff(poses[:,2])>.0001):
        raise ValueError('Only the declared monotonic framing descent is accepted')
    for source,digest in plan['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:
            raise ValueError('Reviewed planning source changed: '+source)


def validate_feedback(state,plan,previous):
    q=np.asarray(state,float)
    if (q.shape!=(8,) or not np.isfinite(q).all() or np.max(abs(q[:7]-np.asarray(previous)[:7]))>.012
            or abs(q[7]-plan['initial_state'][7])>.001):
        raise ValueError('Latest feedback departed from bounded target or gripper changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--onsite-confirmation',type=Path)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text());validate_plan(plan)
    if not args.execute:
        print(json.dumps(dict(reviewed_plan_valid=True,motion_sent=False,hardware_connected=False)))
        return
    if args.onsite_confirmation is None:parser.error('Current onsite clearance and stop-access record required')
    onsite=json.loads(args.onsite_confirmation.read_text())
    if (onsite.get('plan_sha256')!=hashlib.sha256(args.plan.read_bytes()).hexdigest()
            or not all(onsite.get(k) is True for k in ['stop_access','whole_arm_and_new_mount_clearance','no_other_controller'])
            or not 0<=time.time()-onsite.get('confirmed_epoch_s',0)<=1800):
        raise PermissionError('Missing or stale onsite confirmation for this exact calibration motion')
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=['/home/user/zyh/src','/home/user/zyh/scripts']
    from nero_pi05.robot import NeroRobot
    from nero_pi05.semantics import JOINT_LIMITS_RAD
    from visible_demo_start import ControlledStart
    robot=NeroRobot(allow_no_solution=True)
    controlled=ControlledStart(robot)
    tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    report=dict(started_local=datetime.datetime.now().astimezone().isoformat(),plan=str(args.plan),
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),onsite=onsite,
        entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),events=[],
        motion_commands=0,gripper_commands=0,enable_commands=0,reset_commands=0,
        readonly_ros_driver_unchanged=True,complete=False,can_tx_before=tx())
    install_cleanup_signals()
    dump(args.output/'execution.json',report)
    try:
        robot.connect();initial=controlled.read_state();report['initial_health']=initial
        if (np.max(abs(np.array(initial['state'])-plan['initial_state']))>.0005
                or initial['feedback']['motion_status']!=0):
            raise RuntimeError('Real start changed or arm moving; regenerate reviewed framing plan')
        rows=np.asarray(plan['waypoints_rad'])
        if np.any(rows<JOINT_LIMITS_RAD[:,0]) or np.any(rows>JOINT_LIMITS_RAD[:,1]):
            raise ValueError('Joint hard limit')
        # Recompute every joint target before the first allowed CAN write.
        for q,expected in zip(rows,plan['flange_poses_fk']):
            actual=flange_transform(robot.sdk.fk(q.tolist()));expected=flange_transform(expected)
            if not np.allclose(actual,expected,rtol=0,atol=1e-8):raise ValueError('SDK FK changed after planning')
        report['speed_setting_attempted']=True
        report['speed_setting_returned']=False
        dump(args.output/'execution.json',report)
        controlled.prepare_execution(confirmed=True,speed_percent=1)
        report['speed_setting_returned']=True
        dump(args.output/'execution.json',report)
        previous=np.array(initial['state'])
        for index,target in enumerate(rows):
            before=controlled.read_state();validate_feedback(before['state'],plan,previous)
            event=dict(index=index,target_rad=target.tolist(),before=before,send_attempted=True,send_returned=False)
            report['events'].append(event);dump(args.output/'execution.json',report)
            controlled.move_j(target)
            event['send_returned']=True
            report['motion_commands']+=1;deadline=time.monotonic()+4
            stable_since=None
            while time.monotonic()<deadline:
                health=controlled.read_state();validate_feedback(health['state'],plan,np.r_[target,previous[7]])
                flange=flange_transform(health['flange_pose'])
                origin=flange_transform(plan['initial_flange_fk'])
                if (np.max(abs(flange[:2,3]-origin[:2,3]))>.002
                        or not origin[2,3]-.022<=flange[2,3]<=origin[2,3]+.002
                        or Rotation.from_matrix(origin[:3,:3].T@flange[:3,:3]).magnitude()>.01):
                    raise RuntimeError('Actual flange left framing corridor')
                arrived=np.max(abs(np.array(health['state'])[:7]-target))<.0007 and health['feedback']['motion_status']==0
                stable_since=(stable_since or time.monotonic()) if arrived else None
                if stable_since is not None and time.monotonic()-stable_since>=.15:break
                time.sleep(.025)
            else:raise TimeoutError('Framing waypoint did not arrive; no retry or automatic return')
            event['after']=health;previous=np.array(health['state'])
            dump(args.output/'execution.json',report)
        report['final_health']=controlled.read_state()
        if np.linalg.norm(np.array(report['final_health']['flange_pose'])[:3]-np.array(plan['flange_poses_fk'][-1])[:3])>.001:
            raise RuntimeError('Final flange position not reached')
        report['complete']=True
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        controlled.stop_sending()
        finalize_session(args.output,report,None,robot,None,tx)
        report['can_tx_after']=report['can_tx_end']
        dump(args.output/'execution.json',report)
    print(json.dumps(dict(complete=report['complete'],motion_commands=report['motion_commands'],
                         final_flange=report['final_health']['flange_pose'])))


if __name__=='__main__':main()
