"""One explicitly authorized fixed-scene trial. Default is offline, without ROS/CAN."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from executor.bundle import verify_model_bundle
from executor.contract import FullTaskContract
from executor.fixed_permission import require_fixed_trial_permission,MODEL,EVALUATION,ROOT
from executor.runtime import (LiveWorker,SceneObserver,observe_stable,wait_background,
                              json_write,consume_attempt,execute_task)
from executor.execution_guards import require_current_scene


def verify_seal():
    for name,expected in json.loads((ROOT/'package_sha256.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise PermissionError('Fixed-trial package changed: '+name)


def require_clean_handoff(report):
    if (report.get('driver_gate_closed_for_abort') is not True
            or report.get('unresolved_goal_acknowledgements')!=0
            or report.get('unresolved_cancellation_replies')!=0
            or any(report.get(k) for k in ['cancel_errors','cancel_gate_close_error','cancel_spin_error'])):
        raise RuntimeError('Preparation gate close/cancellation did not complete cleanly')


def opening_rows(start,scene):
    from executor.ros_execution import RobotGeometry
    sys.path.insert(0,'/home/user/wq/nero_empty_demo_20261016')
    from air_prepare import preparation_rows
    start=np.asarray(start,float)
    expected=np.asarray(scene['initial_robot_state']['qpos_rad'])
    if (start.shape!=(8,) or not np.isfinite(start).all()
            or np.max(abs(start[:7]-expected))>.005 or not .075<=start[7]<=.095
            or RobotGeometry(scene).pose(start)['tcp'][2]<scene['table_top_z_m']+.4):
        raise ValueError('High empty-gripper opening requires the exact trained high arm start')
    target=start.copy();target[7]=.095
    contract=FullTaskContract(dict(task_scope='complete_pick_carry_place_retreat',chunk_size=20,state_dim=8,action_dim=8))
    rows=np.asarray(preparation_rows(start,target,np.column_stack([contract.lower,contract.upper])))
    if (not np.array_equal(rows[:,:7],np.broadcast_to(start[:7],rows[:,:7].shape))
            or np.max(abs(np.diff(np.r_[start[7],rows[:,7]])))>.0005):
        raise ValueError('Preparation changed an arm target or original gripper rate')
    return rows


def scene_record(rows,scene):
    last=rows[-1]
    return dict(capture_wall_s=time.time()-(time.monotonic()-last['camera_monotonic_s']),
                geometry_fit_passed=True,table_z_base_mm=scene['table_top_z_m']*1000,
                targets=dict(source_bottom_mm=(last['cup_xyz']*1000).tolist(),
                             destination_mm=(last['target_xyz']*1000).tolist()),
                observations=rows,geometry_source='current camera fit to accepted approximate table/TCP model')


def prepare(args):
    from executor.contact import ContactMonitor
    from executor.ros_execution import session_class
    sys.path[:0]=['/home/user/wq/nero_full_task_20261016','/home/user/wq/nero_empty_demo_20261016']
    from task_replay import observer,check_action_results
    from air_prepare import preparation_result
    scene=json.loads((MODEL/'scene_config.json').read_text())
    report=dict(mode='fixed_trial_high_empty_gripper_preparation',success=False,
                execution_goal_submitted=False,arm_targets_held=True,force_limit_n=1.,
                automatic_retry=False,automatic_return=False)
    session=health=worker=pool=None
    try:
        health=observer();session=session_class()(args.output,report,health,scene,ContactMonitor())
        start=np.asarray(session.inventory());session.ready(start)
        rows=opening_rows(start,scene)
        # No heartbeat or control enable until current camera geometry has passed.
        session.last_target=start.copy()
        worker=LiveWorker(MODEL,args.output,session);pool=ThreadPoolExecutor(max_workers=1)
        sensor=wait_background(session,pool.submit(SceneObserver,scene,args.output))
        observations=observe_stable(worker,sensor,pool,session,count=3)
        geometry=scene_record(observations,scene)
        prep_scene=copy.deepcopy(scene)
        prep_scene['initial_robot_state']['gripper_total_width_m']=float(start[7])
        require_current_scene(prep_scene,start,geometry,now=time.time())
        report['initial_scene']=geometry
        session.warm_services();session.configure_initial_scene()
        session.validate_rows(rows);session.ready(start)
        session.enable_monitoring();clients=session.clients()
        begin=len(session.trace);sent=time.monotonic()
        report.update(start=start,target=rows[-1],duration_s=len(rows)*.05,
                      execution_goal_submitted=True)
        json_write(args.output/'preparation_rows.json',rows)
        futures=session.send_block(clients,rows)
        while not all(f.done() for f in futures):
            session.spin(.005);check_action_results(futures)
            actual=np.asarray(session.actual());elapsed=time.monotonic()-sent
            expected=rows[min(len(rows)-1,max(0,int(elapsed/.05)-1))]
            if (np.max(abs(actual[:7]-start[:7]))>.002
                    or not start[7]-.001<=actual[7]<=.096
                    or abs(actual[7]-expected[7])>.004):
                raise RuntimeError('Actual high-open preparation left its held-arm/tracking corridor')
            if elapsed>report['duration_s']+5:raise TimeoutError('Preparation timed out')
        results=[session.wait(f) for f in futures];session.spin(1.)
        report['physical_result']=preparation_result(start,rows[-1],
            [t['position'] for t in session.trace[begin:]],
            all(r.status==4 and r.result.error_code==0 for r in results))
        if not report['physical_result']['success']:raise RuntimeError('Physical opening not verified')
        geometry=scene_record(observe_stable(worker,sensor,pool,session,count=3),scene)
        require_current_scene(scene,session.actual(),geometry,now=time.time())
        json_write(args.output/'geometry.json',geometry)
        report.update(success=True,planned_gate_handoff_allowed=True,
                      final_actual=session.actual(),finished_wall_s=time.time())
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if session is not None:
            session.monitoring=False
            try:
                session.cancel_all();session.contact_permission(False)
                require_clean_handoff(report)
            except Exception as exc:
                report['cleanup_error']=str(exc);report['success']=False
        if worker is not None:worker.close()
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        if session is not None:
            json_write(args.output/'actual_feedback.json',session.trace)
            json_write(args.output/'independent_health.json',session.health_trace)
            session.node.destroy_node();session.ros.try_shutdown()
        if health is not None:health.disconnect()
        json_write(args.output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--stage',choices=['prepare','task'],default='prepare')
    parser.add_argument('--attempt',type=Path,default=ROOT/'fixed_trial/attempt_01')
    parser.add_argument('--receipt',type=Path,default=ROOT/'fixed_trial/receipt.json')
    args=parser.parse_args();args.model=MODEL;args.output=args.attempt/args.stage
    metadata=verify_model_bundle(MODEL);evaluation=json.loads(EVALUATION.read_text())
    if not args.execute:
        print(json.dumps(dict(mode='offline_only',hardware_connected=False,motion_sent=False,
            evaluation_passed=evaluation['passed'],evaluation_total=evaluation['total'])));return
    receipt=json.loads(args.receipt.read_text())
    require_fixed_trial_permission(metadata,evaluation,receipt,now=time.time());verify_seal()
    if args.stage=='prepare':
        # One shared receipt ledger claims the entire physical prep + task, never each phase.
        consume_attempt(args.receipt,args.attempt)
        args.output.mkdir(parents=True,exist_ok=False)
        report=prepare(args)
    else:
        prep=args.attempt/'prepare'
        previous=json.loads((prep/'report.json').read_text())
        require_clean_handoff(previous)
        canonical=json.dumps(receipt,sort_keys=True,separators=(',',':'),allow_nan=False)
        marker=ROOT/'evidence/authorization_attempts'/(hashlib.sha256(canonical.encode()).hexdigest()+'.json')
        claim=json.loads(marker.read_text())
        if (claim['output']!=str(args.attempt.resolve()) or previous.get('success') is not True
                or previous.get('planned_gate_handoff_allowed') is not True
                or not 0<=time.time()-previous['finished_wall_s']<=90):
            raise PermissionError('Task requires this same one-shot successful preparation')
        args.geometry=prep/'geometry.json'
        args.output.mkdir(parents=True,exist_ok=False)
        (args.attempt/'task_started').open('x').close()
        # Load CPU perception libraries before the monitored ROS loop exists.
        # The original runtime still constructs its observer and applies every guard.
        SceneObserver(json.loads((MODEL/'scene_config.json').read_text()),args.output)
        report=execute_task(args,metadata,evaluation,receipt)
    print(json.dumps(dict(stage=args.stage,success=report['success'],error=report.get('error'),
                         output=str(args.output))),flush=True)
    if not report['success']:raise SystemExit(1)

if __name__=='__main__':main()
