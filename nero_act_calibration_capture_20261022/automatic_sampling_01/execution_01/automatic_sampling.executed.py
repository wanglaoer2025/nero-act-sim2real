"""Bounded calibration sampling helpers. Importing opens no devices."""
import json
import argparse
import datetime
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from capture import (make_sample,marker_pixels,flange_transform,camera_metadata,dump,
                     record_attempt,finalize_session,install_cleanup_signals)

ROOT=Path(__file__).resolve().parent


def flange_fk(q):
    from pyAgxArm.utiles.mdh_kinematics import get_mdh,fk_from_mdh
    return flange_transform(fk_from_mdh(get_mdh('nero'),np.asarray(q).tolist()))


def validate_route(plan):
    sys.path.insert(0,'/home/user/zyh/src')
    from nero_pi05.semantics import JOINT_LIMITS_RAD
    stages=plan['stages'];initial=np.asarray(plan['initial_state'],float)
    workspace=np.asarray(plan['flange_workspace_m'],float)
    if workspace.shape!=(3,2) or not np.isfinite(workspace).all() or np.any(workspace[:,0]>=workspace[:,1]):
        raise ValueError('Finite ordered three-axis workspace required')
    if (not plan['nominal_route_passed'] or plan['speed_percent']!=1 or plan['max_step_rad']!=.006
            or plan['max_latest_feedback_lead_rad']!=.012 or initial.shape!=(8,)
            or not np.isfinite(initial).all() or not .094<=initial[7]<=.096
            or [s['role'] for s in stages]!=['prepare']+['fit']*15+['pose_check']*5):
        raise ValueError('Reviewed21stage arm-only1percent calibration route required')
    previous=initial[:7];total=0;poses=[];fit_positions=[]
    for stage in stages:
        rows=np.asarray(stage['waypoints_rad'],float);goal=np.asarray(stage['goal_q_rad'],float)
        if (rows.ndim!=2 or rows.shape[1]!=7 or len(rows)<1 or goal.shape!=(7,)
                or not np.isfinite(np.r_[rows.ravel(),goal]).all()
                or np.any(rows<JOINT_LIMITS_RAD[:,0]+.002) or np.any(rows>JOINT_LIMITS_RAD[:,1]-.002)
                or np.max(abs(np.diff(np.vstack([previous,rows]),axis=0)))>.006001
                or not np.allclose(rows[-1],goal,rtol=0,atol=1e-9)):
            raise ValueError('Unreviewed joint range, step or endpoint')
        T=flange_fk(goal)
        if stage['role']!='prepare':
            expected=np.asarray(stage['goal_transform'],float)
            if expected.shape!=(4,4) or not np.allclose(T,expected,rtol=0,atol=.0002):
                raise ValueError('FK endpoint differs from reviewed pose')
            for old in poses:
                if (np.linalg.norm(old[:3,3]-T[:3,3])<.001
                        and Rotation.from_matrix(old[:3,:3].T@T[:3,:3]).magnitude()<np.deg2rad(1)):
                    raise ValueError('Duplicate fit/check endpoint')
            poses.append(T)
            if stage['role']=='fit':fit_positions.append(T[:3,3])
        previous=goal;total+=len(rows)
    if total!=plan['waypoint_count'] or total>2500 or np.count_nonzero(np.ptp(fit_positions,axis=0)>=.0998)<2:
        raise ValueError('Route budget or two-axis100mm coverage invalid')
    for filename,digest in plan['source_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:
            raise ValueError('Reviewed geometry source changed: '+filename)


def load_route(path):
    snapshot=path.read_bytes()
    plan=json.loads(snapshot);validate_route(plan)
    return plan,hashlib.sha256(snapshot).hexdigest()


def guard_marker(plan,state,pixels,K):
    pixels=np.asarray(pixels,float)
    if (pixels.shape!=(4,2) or not np.isfinite(pixels).all() or np.any(pixels[:,0]<1)
            or np.any(pixels[:,0]>638) or np.any(pixels[:,1]<1) or np.any(pixels[:,1]>358)):
        raise RuntimeError('Full ID0 corners required before continuing motion')
    BF=flange_fk(np.asarray(state)[:7]);BC=np.asarray(plan['provisional_camera_base'])
    h=.095/2;obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
    residual=[]
    for mount in plan['provisional_mount_hypotheses']:
        CM=np.linalg.inv(BC)@BF@np.asarray(mount);points=obj@CM[:3,:3].T+CM[:3,3]
        if np.any(points[:,2]<=0):continue
        projected=points@np.asarray(K).T;projected=projected[:,:2]/projected[:,2,None]
        residual.append(float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1)))))
    if not residual or min(residual)>12:
        raise RuntimeError('Marker departed from provisional pose envelope; possible slip or geometry mismatch')
    return min(residual)


def acquire_pair(payload,robot,image,detector,K,*,role):
    window=[];payload.update(health_window=window,sample=None,frame_candidates=[])
    deadline=time.monotonic()+3.
    while not window or window[-1]['monotonic']-window[0]['monotonic']<.65:
        if time.monotonic()>=deadline:raise TimeoutError('Stationary window did not become ready')
        window.append(robot.read_health(require_enabled=True));time.sleep(.02)
    previous_stamp=None
    for _ in range(30):
        if time.monotonic()>=deadline:break
        before=robot.read_health(require_enabled=True);window.append(before)
        native,stamp_ns=image()
        payload.update(native=native,image_stamp_ns=int(stamp_ns),image_received_monotonic=time.monotonic())
        after=robot.read_health(require_enabled=True);payload['health_after']=after
        stamps=np.asarray(before['feedback_timestamps']+after['feedback_timestamps'],float)
        max_offset=float(np.max(abs(stamps-stamp_ns/1e9))*1000)
        fresh=(stamp_ns!=previous_stamp and np.isfinite(stamps).all() and max_offset<=50)
        payload['frame_candidates'].append(dict(image_stamp_ns=int(stamp_ns),max_pair_offset_ms=max_offset,
            new_frame=stamp_ns!=previous_stamp,within50ms=bool(fresh)))
        previous_stamp=stamp_ns
        if fresh:
            corners,ids,_=detector.detectMarkers(native)
            payload['sample']=make_sample(window,after,stamp_ns/1e9,marker_pixels(corners,ids),K,role=role)
            return
        window.append(after);time.sleep(.005)
    raise TimeoutError('No new image/SDK pair within50ms; no threshold relaxation')


def append_event(path,event):
    with path.open('a') as stream:
        stream.write(json.dumps(event,allow_nan=False,default=lambda v:v.tolist())+'\n')
        stream.flush();os.fsync(stream.fileno())


def audited_call(path,kind,details,callback):
    event=dict(kind=kind,details=details,monotonic=time.monotonic(),state='attempted')
    append_event(path,event)
    try:result=callback()
    except BaseException as exc:
        append_event(path,dict(event,state='failed',error=repr(exc),monotonic=time.monotonic()))
        raise
    append_event(path,dict(event,state='returned',monotonic=time.monotonic()))
    return result


def live_guard(health,plan,reference):
    state=np.asarray(health['state'],float)
    if (state.shape!=(8,) or not np.isfinite(state).all()
            or np.max(abs(state[:7]-np.asarray(reference)[:7]))>.012
            or abs(state[7]-plan['initial_state'][7])>.001):
        raise RuntimeError('Actual state departed from target or clamped-board width changed')
    expected=flange_fk(state[:7]);actual=flange_transform(health['flange_pose'])
    workspace=np.asarray(plan['flange_workspace_m'])
    if (np.any(actual[:3,3]<workspace[:,0]) or np.any(actual[:3,3]>workspace[:,1])
            or np.linalg.norm(expected[:3,3]-actual[:3,3])>.003
            or Rotation.from_matrix(expected[:3,:3].T@actual[:3,:3]).magnitude()>.02):
        raise RuntimeError('Actual flange left reviewed workspace or disagrees with FK')


def observe_motion(camera,plan,health):
    frame,_=camera.read();pixels=camera.corners(frame)
    if pixels is None:raise RuntimeError('Exactly one complete ID0 required; stop without retry')
    residual=guard_marker(plan,health['state'],pixels,camera.K)
    camera.show(frame,'CALIBRATION1% | Q/ESC STOP | keep hands clear')
    return residual


class DeadlineSDK:
    """Preserve the SDK interface, checking the bound after the final health read."""
    def __init__(self,sdk,deadline):self.sdk,self.deadline=sdk,deadline
    def __getattr__(self,name):return getattr(self.sdk,name)
    def check(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('Calibration send deadline reached')
    def move_j(self,q):
        self.check();return self.sdk.move_j(q)
    def set_speed_percent(self,value):
        self.check();return self.sdk.set_speed_percent(value)


def drive_stage(robot,camera,plan,stage,log_path,deadline):
    sys.path.insert(0,'/home/user/zyh/scripts')
    from visible_demo_start import ControlledStart
    rows=np.asarray(stage['waypoints_rad']);reference=np.asarray(robot.read_health(require_enabled=True)['state'])
    control=None;original_sdk=robot.sdk;robot.sdk=DeadlineSDK(original_sdk,deadline)
    try:
        for index,target in enumerate(rows):
            if time.monotonic()>=deadline:raise TimeoutError('Whole calibration route deadline')
            health=robot.read_health(require_enabled=True);live_guard(health,plan,reference)
            observe_motion(camera,plan,health)
            if index%26==0:
                if control is not None:control.stop_sending()
                # The reviewed global route is finite; each local block also retains the existing26-target guard.
                control=ControlledStart(robot)
                audited_call(log_path,'speed',dict(stage=stage['index'],block=index//26,percent=1),
                             lambda:control.prepare_execution(confirmed=True,speed_percent=1))
            audited_call(log_path,'arm',dict(stage=stage['index'],waypoint=index,target=target.tolist(),before=health),
                         lambda:control.move_j(target))
            stop=min(deadline,time.monotonic()+4.)
            while time.monotonic()<stop:
                current=control.read_state();live_guard(current,plan,np.r_[target,reference[7]])
                observe_motion(camera,plan,current)
                if np.max(abs(np.asarray(current['state'])[:7]-target))<.0007 and current['feedback']['motion_status']==0:
                    reference=np.asarray(current['state']);break
                time.sleep(.01)
            else:raise TimeoutError('Calibration waypoint did not arrive; no reissue or automatic return')
        return current
    finally:
        try:
            if control is not None:control.stop_sending()
        finally:robot.sdk=original_sdk


def finish(output,report,camera,robot,log,tx):
    errors=[];events=[]
    try:
        if robot is not None:
            try:robot.stop_sending()
            except BaseException as exc:errors.append('stop: '+repr(exc))
        if log.exists():
            for index,line in enumerate(log.read_text().splitlines()):
                try:
                    event=json.loads(line)
                    if not isinstance(event,dict) or event.get('kind') not in ('arm','speed') or event.get('state') not in ('attempted','returned','failed'):
                        raise ValueError('Incomplete audit event')
                    events.append(event)
                except Exception as exc:errors.append(f'line{index+1}: {exc!r}')
    except BaseException as exc:errors.append(repr(exc))
    finally:
        report['arm_send_attempts']=sum(e['kind']=='arm' and e['state']=='attempted' for e in events)
        report['arm_send_returns']=sum(e['kind']=='arm' and e['state']=='returned' for e in events)
        report['audit_errors']=errors
        report['motion_sent_or_possible']=report['arm_send_attempts']>0 or bool(errors)
        if errors:report['complete']=False
        finalize_session(output,report,camera,robot,None,tx)


class Camera:
    def __init__(self,profile,output,*,window):
        import cv2
        import pyzed.sl as sl
        self.cv2,self.sl,self.output,self.window=cv2,sl,output,window
        self.cam=sl.Camera();self.last_stamp=None;self.last_preview=0.
        self.title='NERO calibration1percent | Q/ESC STOP'
        try:
            init=sl.InitParameters();init.camera_resolution=sl.RESOLUTION.HD720;init.camera_fps=30
            init.depth_mode=sl.DEPTH_MODE.NONE;init.set_from_serial_number(int(profile['camera_serial']))
            if self.cam.open(init)!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('ZED open failed')
            self.res,self.mat,self.runtime=sl.Resolution(640,360),sl.Mat(),sl.RuntimeParameters()
            info=self.cam.get_camera_information(self.res);intr=info.camera_configuration.calibration_parameters.left_cam
            self.K=np.array([[intr.fx,0.,intr.cx],[0.,intr.fy,intr.cy],[0.,0.,1.]])
            self.metadata=camera_metadata(profile,info.serial_number,self.K)
            params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),params)
            if window:cv2.namedWindow(self.title,cv2.WINDOW_NORMAL);cv2.resizeWindow(self.title,1100,619)
        except BaseException:
            self.close();raise

    def read(self):
        sl=self.sl;deadline=time.monotonic()+.30
        while time.monotonic()<deadline:
            if self.cam.grab(self.runtime)!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('ZED grab failed')
            if self.cam.retrieve_image(self.mat,sl.VIEW.LEFT,sl.MEM.CPU,self.res)!=sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('ZED retrieve failed')
            stamp=self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            age=time.time()-stamp/1e9
            if stamp!=self.last_stamp and -.05<=age<=.100:
                intr=self.cam.get_camera_information(self.res).camera_configuration.calibration_parameters.left_cam
                if not np.array_equal(self.K,[[intr.fx,0.,intr.cx],[0.,intr.fy,intr.cy],[0.,0.,1.]]):
                    raise RuntimeError('Actual intrinsics changed; stop this dataset')
                self.last_stamp=stamp
                return np.asarray(self.mat.get_data()).copy()[...,:3],stamp
            time.sleep(.002)
        raise TimeoutError('No new native frame within100ms stamp age; clock mapping remains unverified')

    def corners(self,frame):
        corners,ids,_=self.detector.detectMarkers(frame)
        return marker_pixels(corners,ids)

    def show(self,frame,message):
        cv2=self.cv2;upright=frame[::-1,::-1].copy();pixels=self.corners(frame)
        if pixels is not None:
            points=np.c_[639-pixels[:,0],359-pixels[:,1]].astype(np.int32)
            cv2.polylines(upright,[points],True,(0,220,0),2)
        cv2.putText(upright,message[:90],(6,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,0),1)
        if time.monotonic()-self.last_preview>1:
            temporary=self.output/'preview.tmp.png'
            if cv2.imwrite(str(temporary),upright):temporary.replace(self.output/'latest_upright.png')
            self.last_preview=time.monotonic()
        if self.window:
            cv2.imshow(self.title,upright)
            if cv2.waitKey(1)&0xff in (ord('q'),27) or cv2.getWindowProperty(self.title,cv2.WND_PROP_VISIBLE)<1:
                raise KeyboardInterrupt('Operator closed or stopped calibration window')

    def close(self):
        try:self.cam.close()
        finally:
            if self.window:self.cv2.destroyAllWindows()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--capture-current',action='store_true')
    for name in ['plan','review','onsite-confirmation','output']:parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    plan=None;digest=None
    if args.plan is not None:plan,digest=load_route(args.plan)
    if not args.execute and not args.capture_current:
        print(json.dumps(dict(route_valid=plan is not None,hardware_connected=False,motion_sent=False)));return
    if args.output is None:parser.error('A new output directory is required')
    if args.execute:
        if plan is None or args.review is None or args.onsite_confirmation is None:
            parser.error('Execute requires reviewed plan, software/geometry review and current onsite confirmation')
        review=json.loads(args.review.read_text());onsite=json.loads(args.onsite_confirmation.read_text())
        if (review.get('plan_sha256')!=digest or review.get('software_review_complete') is not True
                or onsite.get('plan_sha256')!=digest
                or not 0<=time.time()-onsite.get('confirmed_epoch_s',0)<=1800
                or not all(onsite.get(k) is True for k in ['stop_access','whole_arm_and_clamped_board_clearance','board_rigidly_clamped','no_other_controller'])):
            raise PermissionError('Exact calibration route review and fresh complete onsite conditions required')
        for filename,expected in review['source_sha256'].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:raise ValueError('Reviewed execution source changed')
    args.output.mkdir(parents=True,exist_ok=False)
    tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    profile=json.loads((ROOT/'marker_measured_95mm/marker_profile.json').read_text())
    report=dict(started_local=datetime.datetime.now().astimezone().isoformat(),running=True,
        mode='scoped_calibration_motion' if args.execute else 'receive_only_static_candidate',
        events=[],fit_samples=0,pose_check_samples=0,sdk_send_blocked=not args.execute,
        gripper_commands=0,enable_commands=0,reset_commands=0,policy_inference_count=0,
        calibration_verified=False,hardware_deployment_allowed=False,complete=False,
        can_tx_start=tx(),source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        marker_profile=profile,plan=None if args.plan is None else str(args.plan),plan_sha256=digest)
    robot=None;camera=None;install_cleanup_signals();log=args.output/'control_events.jsonl'
    try:
        if args.execute:
            sys.path.insert(0,'/home/user/zyh/src')
            from nero_pi05.robot import NeroRobot
            robot=NeroRobot(allow_no_solution=True);robot.connect()
        else:
            sys.path.insert(0,'/home/user/wq/nero_act_transfer_resume_20261021')
            from robot_health import observer
            robot=observer()
        initial=robot.read_health(require_enabled=True);report['initial_health']=initial
        camera=Camera(profile,args.output,window=bool(args.execute));report['camera_metadata']=camera.metadata
        dump(args.output/'manifest.json',report)
        if args.execute:
            if np.max(abs(np.asarray(initial['state'])-plan['initial_state']))>.0005 or initial['feedback']['motion_status']!=0:
                raise RuntimeError('Real start changed; reviewed route cannot be reused')
            observe_motion(camera,plan,initial);deadline=time.monotonic()+1200
            for stage in plan['stages']:
                report['active_stage']=stage['index'];dump(args.output/'manifest.json',report)
                drive_stage(robot,camera,plan,stage,log,deadline)
                if stage['role']=='prepare':continue
                record_attempt(args.output,report,stage['role'],
                    lambda payload:acquire_pair(payload,robot,camera.read,camera.detector,camera.K,role=stage['role']),tx)
                if not report['events'][-1]['accepted']:raise RuntimeError('Stationary candidate rejected; stop without automatic retry')
                print(json.dumps(dict(stage=stage['index'],fit=report['fit_samples'],pose_check=report['pose_check_samples'])),flush=True)
            report['complete']=report['fit_samples']==15 and report['pose_check_samples']==5
        else:
            record_attempt(args.output,report,'fit',
                lambda payload:acquire_pair(payload,robot,camera.read,camera.detector,camera.K,role='fit'),tx)
        report['final_health']=robot.read_health(require_enabled=True)
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        finish(args.output,report,camera,robot,log,tx)
    print(json.dumps(dict(complete=report['complete'],fit=report['fit_samples'],pose_check=report['pose_check_samples'])),flush=True)


if __name__=='__main__':main()
