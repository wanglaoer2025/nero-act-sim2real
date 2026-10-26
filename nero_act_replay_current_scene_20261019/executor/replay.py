"""Protected full-amplitude replay of a frozen ACT physical simulation rollout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from executor.bundle import verify_model_bundle
from executor.contact import ContactMonitor
from executor.execution_guards import require_current_scene,actual_path_verified
from executor.permission import require_experimental_permission
from executor.replay_trace import MODE,verify_replay
from executor.runtime import consume_attempt,json_write,confirm,wait_background,SceneObserver
from executor.release_hold import can_request_support_confirmation
from executor.scene_acceptance import execution_scene

ROOT=Path(__file__).resolve().parents[1]


def require_lift_ready(contact):
    if not contact.close_authorized or contact.state!='holding' or contact.contact_width is None:
        raise RuntimeError('Stable measured real-cup contact is required before lift')


def require_carry_ready(contact,pose,scene):
    require_lift_ready(contact)
    if pose[2]<.20 or np.linalg.norm(pose[:2]-scene['pick_xy_m'])>.03:
        raise RuntimeError('Real cup must be high above source before carry confirmation')


def require_release_ready(contact,pose,scene):
    require_lift_ready(contact)
    if not contact.carry_authorized or not can_request_support_confirmation(pose,scene['place_xy_m']):
        raise RuntimeError('Cup must be in the supported placement region before RELEASE')


def observe(camera,sensor,pool,session,count=8,minimum_seconds=0):
    measured=[]
    for _ in range(count):
        started=time.monotonic()
        frame=wait_background(session,pool.submit(camera.capture))
        row=wait_background(session,pool.submit(sensor.measure,frame))
        if time.monotonic()-row['camera_monotonic_s']>.75:
            raise RuntimeError('Scene measurement became stale')
        measured.append(row)
        session.spin(max(.001,.5-(time.monotonic()-started)))
    cups=np.array([row['cup_xyz'] for row in measured]);cards=np.array([row['target_xyz'] for row in measured])
    if (np.max(np.ptp(cups,axis=0))>.005 or np.max(np.ptp(cards,axis=0))>.005
            or measured[-1]['camera_monotonic_s']-measured[0]['camera_monotonic_s']<minimum_seconds):
        raise RuntimeError('Cup/card observations are not independently stable')
    return measured


def execute(args,metadata,manifest,rows):
    from executor.replay_session import replay_session_class
    from executor.camera import SceneCamera
    sys.path.insert(0,'/home/user/wq/nero_full_task_20261016')
    from task_replay import observer
    scene=execution_scene(ROOT,json.loads((args.model/'scene_config.json').read_text()),manifest,require_evidence=True)
    report=dict(mode=MODE,success=False,physical_success=False,teacher_action_fallback=False,
                online_visual_policy=False,trace_sha256=manifest['source_rollout_sha256'],
                execution_goal_submitted=False,read_only_restoration_pending=True,
                nominal_reference_duration_s=len(rows)*.05)
    report['execution_source_xy_m']=scene['pick_xy_m']
    session=health=camera=pool=None;contact=ContactMonitor();sent=[]
    try:
        # Torch/YOLO lazy imports and first inference must finish before SDK/ROS
        # receivers start; their GIL pauses otherwise invalidate live feedback.
        print('COLD_INPUTS_WARMING',flush=True)
        pool=ThreadPoolExecutor(max_workers=1)
        sensor=SceneObserver(scene,args.output)
        camera=pool.submit(SceneCamera,args.output/'camera').result(timeout=45)
        sensor.measure(pool.submit(camera.capture).result(timeout=10))
        report['cold_inputs_warmed_before_robot_receiver']=True
        print('COLD_INPUTS_READY',flush=True)
        health=observer();session=replay_session_class()(args.output,report,health,scene,contact)
        actual=session.inventory();session.ready(actual)
        require_current_scene(scene,actual,json.loads(args.geometry.read_text()),now=time.time())
        session.warm_services()
        initial=observe(camera,sensor,pool,session)
        measured=initial[-1]
        if (np.linalg.norm(measured['cup_xyz'][:2]-scene['pick_xy_m'])>.005
                or np.linalg.norm(measured['target_xyz'][:2]-scene['place_xy_m'])>.005):
            raise RuntimeError('Current source cup/card differs from frozen ACT scene')
        report['initial_scene_observations']=initial
        session.configure_initial_scene();clients=session.clients()
        require_current_scene(scene,session.actual(),json.loads(args.geometry.read_text()),now=time.time())
        session.enable_monitoring()
        for phase in manifest['phases']:
            name=phase['name'];pose=session.geometry.pose(session.actual())['tcp']
            if name=='close':
                if not can_request_support_confirmation(pose,scene['pick_xy_m']):
                    raise RuntimeError('ACT did not reach the actual grasp region')
                confirm(session,'GRASP','两指在黄色空杯壁两侧，未顶杯口或桌面，杯竖直，手已离开，仍可立即急停。',report)
                fresh=observe(camera,sensor,pool,session)
                if np.linalg.norm(fresh[-1]['cup_xyz'][:2]-scene['pick_xy_m'])>.005:
                    raise RuntimeError('Cup moved during GRASP confirmation')
                contact.authorize('close','GRASP',{'aligned':True})
                session.contact_hold_pose=session.geometry.pose(session.actual())['tcp'].copy()
                session.allow_finger_contact(True);session.contact_permission(True)
            elif name=='lift':
                require_lift_ready(contact);session.attach_actual_source()
                report['actual_grasp_width_m']=contact.contact_width
                report['cup_attachment_actual_state']=session.actual()
            elif name=='carry_place':
                require_carry_ready(contact,pose,scene)
                confirm(session,'NEXT','杯已随夹爪离桌，夹持稳定、杯竖直，完整搬运和下降路径净空。',report)
                require_carry_ready(contact,session.geometry.pose(session.actual())['tcp'],scene)
                contact.authorize('carry','NEXT',{'cup_lifted':True})
            elif name=='release':
                require_release_ready(contact,pose,scene)
                confirm(session,'RELEASE','杯底已落在蓝卡上，桌面承托且杯竖直；悬空、偏位或不稳请停止。',report)
                require_release_ready(contact,session.geometry.pose(session.actual())['tcp'],scene)
                supported=observe(camera,sensor,pool,session)
                if np.linalg.norm(supported[-1]['cup_xyz'][:2]-scene['place_xy_m'])>.02:
                    raise RuntimeError('Real supported cup is outside target region')
                report['supported_release_observations']=supported
                contact.authorize('release','RELEASE',{'supported':True,'upright':True})
                session.detach_for_supported_release(supported[-1]['cup_xyz'][:2])
                session.contact_hold_pose=session.geometry.pose(session.actual())['tcp'].copy()
            elif name=='retreat':
                if contact.state!='released':raise RuntimeError('Measured gripper has not released cup')
                placed=observe(camera,sensor,pool,session)
                if np.linalg.norm(placed[-1]['cup_xyz'][:2]-scene['place_xy_m'])>.02:
                    raise RuntimeError('Released cup is outside accepted placement region')
                confirm(session,'NEXT','杯子独立站稳、夹爪与杯壁分离，退出通道净空。',report)
                session.finish_supported_release(placed[-1]['cup_xyz'][:2]);session.contact_permission(False)
            report['execution_goal_submitted']=True
            commanded=session.run_phase(name,rows[phase['start']:phase['end']],clients,pool,wait_background)
            sent.extend(commanded.tolist())
            print('PHASE_COMPLETE '+name,flush=True)
        final=observe(camera,sensor,pool,session,count=12,minimum_seconds=5)
        if any(np.linalg.norm(r['cup_xyz'][:2]-r['target_xyz'][:2])>.02 for r in final):
            raise RuntimeError('Cup did not remain on blue target for final5seconds')
        report['final_scene_observations']=final
        report['actual_motion_verified']=actual_path_verified(sent,[item['position'] for item in session.trace])
        report['physical_success']=report['success']=bool(report['actual_motion_verified']
            and contact.state=='released' and contact.carry_authorized and contact.release_authorized)
        if not report['success']:raise RuntimeError('Full actual motion/contact evidence is incomplete')
    except BaseException as exc:
        report.update(success=False,physical_success=False,error=f'{type(exc).__name__}: {exc}')
    finally:
        if session is not None:
            session.monitoring=False
            try:session.cancel_all()
            except Exception as exc:report['abort_error']=str(exc)
            try:session.contact_permission(False)
            except Exception as exc:report['contact_permission_close_error']=str(exc)
        if camera is not None:camera.close()
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        if session is not None:
            json_write(args.output/'actual_feedback.json',session.trace)
            json_write(args.output/'independent_health.json',session.health_trace)
            close_receiver=getattr(session,'close_health_receiver',None)
            if close_receiver is not None:
                try:close_receiver()
                except Exception as exc:report['health_receiver_cleanup_error']=str(exc)
            session.node.destroy_node();session.ros.try_shutdown()
        if health is not None:health.disconnect()
        json_write(args.output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['model','evaluation','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--geometry',type=Path);parser.add_argument('--receipt',type=Path)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    args.model=args.model.resolve();args.output=args.output.resolve()
    metadata=verify_model_bundle(args.model);manifest,rows=verify_replay(ROOT,metadata)
    evaluation=json.loads(args.evaluation.read_text())
    if not args.execute:
        args.output.mkdir(parents=True,exist_ok=False)
        report=dict(mode=MODE,hardware_connected=False,motion_sent=False,points=len(rows),
            nominal_duration_s=len(rows)*.05,arm_span_deg=np.rad2deg(np.ptp(rows[:,:7],axis=0)),
            source_counts=manifest['sources'],execution_permission=False)
        json_write(args.output/'report.json',report);print(json.dumps(report,default=lambda a:a.tolist()));return
    if not args.receipt or not args.geometry:parser.error('execute requires current receipt and geometry')
    require_experimental_permission(metadata,evaluation,json.loads(args.receipt.read_text()),now=time.time())
    for name,expected in json.loads((ROOT/'package_sha256.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise PermissionError('Source changed: '+name)
    consume_attempt(args.receipt,args.output);args.output.mkdir(parents=True,exist_ok=False)
    report=execute(args,metadata,manifest,rows)
    print(json.dumps(dict(success=report['success'],output=str(args.output),mode=MODE)),flush=True)
    if not report['success']:raise SystemExit(1)


if __name__=='__main__':main()
