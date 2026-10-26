"""Supervised complete-task ACT execution. Default is an offline artifact report."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import queue
import select
import subprocess
import sys
import threading
import time

import numpy as np
from executor.health_trace import HealthTrace

from executor.bundle import verify_model_bundle
from executor.contact import ContactMonitor
from executor.contract import FullTaskContract
from executor.execution_guards import require_current_scene, actual_path_verified
from executor.permission import require_experimental_permission
from executor.phases import PhaseGuard
from executor.release_hold import hold_act_arm_during_release,require_support_pose,release_reference_transfer,contact_hold_required,hold_open_before_grasp_depth,first_lift_index,can_request_support_confirmation

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_LEDGER = ROOT/'evidence/authorization_attempts'


def consume_attempt(receipt, output):
    receipt = Path(receipt)
    # Identity follows authorization content, never its filename or output directory.
    content = json.loads(receipt.read_text())
    canonical = json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False)
    identity = hashlib.sha256(canonical.encode()).hexdigest()
    ATTEMPT_LEDGER.mkdir(parents=True, exist_ok=True)
    marker = ATTEMPT_LEDGER/(identity+'.json')
    with marker.open('x') as stream:
        json.dump(dict(authorization_sha256=identity, receipt_path=str(receipt.resolve()),
                       output=str(Path(output).resolve()), started_wall_s=time.time()), stream, indent=2)
    return marker


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x:
        x.tolist() if isinstance(x, np.ndarray) else list(x) if isinstance(x, HealthTrace) else str(x)))


class LiveWorker:
    def __init__(self, model, output, session):
        self.session, self.records = session, []
        self.log = (output/'policy_worker.log').open('x')
        self.process = subprocess.Popen([sys.executable, '-u', '-m', 'executor.policy_worker',
            '--model', str(model), '--stream', '--output', str(output/'camera_inputs')], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        try:
            self.ready = self.receive(45)
            if (self.ready.get('type') != 'ready' or self.ready.get('checkpoint_sha256')
                    != verify_model_bundle(model)['checkpoint_sha256']):
                raise RuntimeError('Frozen ACT worker handshake mismatch')
        except BaseException:
            self.close()
            raise

    def receive(self, timeout):
        deadline = time.monotonic()+timeout
        while not select.select([self.process.stdout], [], [], 0)[0]:
            if self.process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Live ACT worker unavailable or timed out')
            self.session.spin(.002)
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Live ACT worker closed its response stream')
        return json.loads(line)

    def predict(self):
        self.session.spin(.005)
        state = self.session.actual()
        receipt = self.session.latest['/feedback/joint_states']['receipt']
        if self.session.last_target is None:raise RuntimeError('No initialized submitted reference for ACT context')
        previous=float(self.session.last_target[7])
        self.process.stdin.write(json.dumps(dict(state=state, observed_monotonic_s=receipt,previous_gripper_target_m=previous))+'\n')
        self.process.stdin.flush()
        result = self.receive(5)
        details = result['details']
        if (result.get('type') != 'prediction' or details['observed_monotonic_s'] != receipt
                or not np.array_equal(details['feedback_state'], state)
                or details.get('previous_gripper_target_m')!=previous
                or not 0 <= time.monotonic()-details['camera_received_monotonic_s'] <= .25):
            raise RuntimeError('ACT response does not match its current state/camera request')
        self.records.append(result)
        return np.asarray(result['chunk'], float), details

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"stop":true}\n');self.process.stdin.flush()
                self.process.wait(timeout=3)
            except Exception:
                self.process.kill();self.process.wait()
        self.log.close()


class SceneObserver:
    """Cup/card measurements only. No transfer-plan or robot-control calls."""
    def __init__(self, config, output):
        import cv2
        sys.path[:0] = ['/home/user/wq/nero_pick_place_20261016', '/home/user/zyh', '/home/user/zyh/src']
        import perception
        from scripts import nero_methylene_teacher as teacher
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(2)
        self.cv2, self.perception, self.teacher = cv2, perception, teacher
        weights = '/home/user/zyh/data/nero-teacher/runtime-v4/models/yolo.pt'
        self.model = YOLO(weights)
        self.cup_id = next(index for index, name in self.model.names.items() if name == 'cup')
        self.matrix = np.asarray(config['camera_calibration']['intrinsic_matrix'])
        self.transform = np.asarray(config['camera_calibration']['transform_base_from_raw_optical_m']).copy()
        self.transform[:3, 3] *= 1000
        self.args = teacher.parse_args(['--weights', weights, '--pick-place', '--cup-height-mm', '95', '--cup-radius-mm', '37.5',
            '--table-z-base-mm', '0', '--allow-approximate-tcp', '--act-output', str(output/'unused_sensor_only.hdf5'),
            '--act-position-group', '1', '--observe-only'])

    def measure(self, details):
        bgr = self.cv2.imread(details['camera_frame'])
        if bgr is None:
            raise RuntimeError('Cannot read actual inference frame for safety observation')
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        result = self.model.predict(bgr, classes=[self.cup_id], conf=.2, imgsz=960, device='cpu', verbose=False)[0]
        cups = [] if result.boxes is None else result.boxes.xyxy.cpu().numpy()
        measured = self.teacher.localize_pair(
            [self.perception.yellow_cup_box(rgb, cups)], [self.perception.unique_blue_box(rgb)],
            rgb.shape, self.matrix, self.transform, self.args)
        return dict(cup_xyz=np.asarray(measured['source_bottom_mm'])/1000,
                    target_xyz=np.asarray(measured['destination_mm'])/1000,
                    camera_monotonic_s=details['camera_received_monotonic_s'], frame=details['camera_frame'],
                    fit_rmse_px=measured['fit_rmse_px'])


def wait_background(session, future, timeout=45):
    deadline = time.monotonic()+timeout
    while not future.done():
        if time.monotonic() >= deadline:
            raise TimeoutError('Read-only observation computation timed out')
        session.spin(.005)
    return future.result()


def confirm(session, token, message, report):
    replies = queue.Queue()
    print('CONFIRM '+token+': '+message, flush=True)
    def read():
        try:replies.put(input().strip())
        except Exception:replies.put('')
    threading.Thread(target=read, daemon=True).start()
    deadline = time.monotonic()+180
    while replies.empty():
        if time.monotonic() >= deadline:
            raise TimeoutError('Onsite confirmation expired')
        session.spin(.01)
    received = replies.get()
    report.setdefault('confirmations', []).append(dict(requested=token, received=received, wall_s=time.time()))
    if received != token:
        raise PermissionError('Operator did not supply the requested current-stage token')


def observe_stable(worker, sensor, pool, session, *, count=8, minimum_seconds=0):
    rows = []
    for _ in range(count):
        started = time.monotonic()
        _, details = worker.predict()
        rows.append(wait_background(session, pool.submit(sensor.measure, details)))
        elapsed = time.monotonic()-started
        if elapsed < .5:
            session.spin(.5-elapsed)
    points = np.array([r['cup_xyz'] for r in rows])
    targets = np.array([r['target_xyz'] for r in rows])
    if (np.max(np.ptp(points, axis=0)) > .005 or np.max(np.ptp(targets, axis=0)) > .005
            or rows[-1]['camera_monotonic_s']-rows[0]['camera_monotonic_s'] < minimum_seconds):
        raise RuntimeError('Fresh cup/card observations are not stable for the required window')
    return rows


def execute_task(args, metadata, evaluation, receipt):
    from executor.ros_execution import session_class
    sys.path.insert(0, '/home/user/wq/nero_full_task_20261016')
    from task_replay import observer, check_action_results
    config = json.loads((args.model/'training_config.json').read_text())
    scene = json.loads((args.model/'scene_config.json').read_text())
    report = dict(mode='supervised_complete_ACT', success=False, task_outcome_candidate=False,
        task_motion_targets_from_ACT=True, teacher_action_fallback=False, confirmations=[],
        checkpoint_sha256=metadata['checkpoint_sha256'], execution_goal_submitted=False,
        contact_phase_arm_target_source='measured zero-motion hold via original rate limits; gripper remains ACT',
        safety_reference_transfers=[],
        general_hardware_deployment_allowed=False, read_only_restoration_pending=True)
    session = worker = health = pool = None
    contact, phases = ContactMonitor(), PhaseGuard(scene['pick_xy_m'], scene['place_xy_m'],scene['table_top_z_m']+scene['grasp_tcp_height_m'])
    sent, blocks, placed = [], [], None
    active_points, hold_pose = 0, None
    try:
        health = observer()
        session = session_class()(args.output, report, health, scene, contact)
        actual = session.inventory()
        session.ready(actual)
        require_current_scene(scene, actual, json.loads(args.geometry.read_text()), now=time.time())
        session.warm_services()
        session.enable_monitoring()
        worker = LiveWorker(args.model, args.output, session)
        pool = ThreadPoolExecutor(max_workers=1)
        sensor = wait_background(session, pool.submit(SceneObserver, scene, args.output))
        observed = observe_stable(worker, sensor, pool, session)
        measured = observed[-1]
        if (np.linalg.norm(measured['cup_xyz'][:2]-scene['pick_xy_m']) > .005
                or np.linalg.norm(measured['target_xyz'][:2]-scene['place_xy_m']) > .005):
            raise RuntimeError('Fresh worker image differs from trained cup/card scene')
        report['initial_scene_observations'] = observed
        session.configure_initial_scene()
        clients = session.clients()
        guard = FullTaskContract(config)
        previous = np.asarray(session.actual())
        def dispatch(rows,observed_monotonic):
            session.validate_rows(rows)
            if time.monotonic()-observed_monotonic>.25:
                raise RuntimeError('Collision validation exhausted250ms freshness; no stale block sent')
            session.guard();report['execution_goal_submitted']=True
            futures=session.send_block(clients,rows)
            deadline=time.monotonic()+len(rows)*.05+.25
            while not all(f.done() for f in futures):
                session.spin(.005);check_action_results(futures)
                if time.monotonic()>deadline or time.monotonic()-observed_monotonic>.75:
                    raise RuntimeError('Trajectory/result or750ms observation-tail deadline exceeded')
                if np.max(abs(np.asarray(session.actual())[:7]-session.last_target[:7]))>.03:
                    raise RuntimeError('Actual arm is not following its submitted reference')
            check_action_results(futures)
        def prepare_posture_hold(last_target,kind):
            measured=np.asarray(session.actual());pose=session.geometry.pose(measured)['tcp']
            session.contact_hold_pose=pose.copy()
            reference_rows=release_reference_transfer(last_target,measured)
            record=dict(source='feedback_safety_hold_not_ACT_task_action',kind=kind,
                measured_posture=measured,actual_before=measured,reference_targets=reference_rows)
            report['safety_reference_transfers'].append(record)
            for start in range(0,len(reference_rows),10):
                part=reference_rows[start:start+10]
                session.spin(.005);fresh_state=session.actual()
                stamp=session.latest['/feedback/joint_states']['receipt']
                padded=np.pad(part,((0,20-len(part)),(0,0)),mode='edge')
                checked=guard.validate_prefix(padded,dict(state=fresh_state,
                    age_s=time.monotonic()-stamp),last_target)[:len(part)]
                dispatch(checked,stamp);sent.extend(checked.tolist());last_target=checked[-1].copy()
            session.spin(2.);record['actual_after']=session.actual()
            return last_target,pose
        max_points = config.get('reach_deadline_steps', 3200)
        while active_points < max_points:
            session.guard()
            raw, details = worker.predict()
            actual = np.asarray(session.actual())
            age = time.monotonic()-details['observed_monotonic_s']
            rows = guard.validate_prefix(raw, dict(state=details['feedback_state'], age_s=age), previous)
            current_pose = session.geometry.pose(actual)['tcp']
            contact_arm_hold = hold_pose is not None and contact_hold_required(contact)
            if contact_arm_hold:
                rows = hold_act_arm_during_release(rows,previous)
            predicted_poses = np.array([session.geometry.pose(row)['tcp'] for row in rows])
            fine_grasp_query=phases.single_point_grasp_prefix(current_pose,predicted_poses,contact)
            event = phases.inspect(current_pose, predicted_poses, rows[:, 7], previous[7], contact)
            pregrasp_open_hold=(not contact.close_authorized and
                abs(current_pose[2]-(scene['table_top_z_m']+scene['grasp_tcp_height_m']))>.002)
            if pregrasp_open_hold:
                rows=hold_open_before_grasp_depth(rows,previous)
                if event=='GRASP':event=None
            if event == 'WAIT_CONTACT':
                end = time.monotonic()+4
                while contact.state != 'holding' and time.monotonic() < end:
                    session.spin(.01)
                if contact.state != 'holding':
                    raise RuntimeError('No stable measured cup contact; refusing lift')
                continue
            if contact.state == 'holding' and not contact.release_authorized:
                if session.held_center is None:
                    first_lift=first_lift_index(current_pose,predicted_poses)
                    if first_lift==0:
                        session.attach_actual_source()
                        report['cup_attachment_actual_state']=session.actual()
                    elif first_lift is not None:
                        rows=rows[:first_lift]
                bottom = np.array([session.cup_bottom(row) for row in rows]) if session.held_center is not None else np.zeros(len(rows))
                unsafe = np.flatnonzero(bottom < -1e-6)
                if len(unsafe):
                    if unsafe[0] > 0:
                        rows = rows[:unsafe[0]]
                    elif can_request_support_confirmation(current_pose,scene['place_xy_m']):
                        # Stop before the colliding row; only onsite support permits detachment.
                        event = 'RELEASE'
                    else:
                        raise RuntimeError('ACT would carry cup through the table before observed support')
            if event == 'GRASP':
                confirm(session, 'GRASP', '两指位于空杯壁两侧，未顶杯口/桌面，杯竖直，手已离开；确认后才允许低力闭爪。', report)
                fresh = observe_stable(worker, sensor, pool, session)
                if np.linalg.norm(fresh[-1]['cup_xyz'][:2]-scene['pick_xy_m']) > .005:
                    raise RuntimeError('Cup changed source position during GRASP confirmation')
                contact.authorize('close', 'GRASP', {'aligned': True})
                session.allow_finger_contact(True);session.contact_permission(True)
                previous,hold_pose=prepare_posture_hold(previous,'GRASP')
                continue
            if event == 'CARRY':
                confirm(session, 'NEXT', '杯已随夹爪离桌、竖直且夹持稳定，转运路径净空。', report)
                contact.authorize('carry', 'NEXT', {'cup_lifted': True})
                continue
            if event == 'RELEASE':
                confirm(session, 'RELEASE', '杯底已落在蓝卡区域，桌面承托杯子且杯子竖直；悬空或不稳请取消。', report)
                supported = observe_stable(worker, sensor, pool, session)
                if np.linalg.norm(supported[-1]['cup_xyz'][:2]-scene['place_xy_m']) > .02:
                    raise RuntimeError('Supported cup does not match the accepted placement region')
                report['supported_release_observations'] = supported
                contact.authorize('release', 'RELEASE', {'supported': True, 'upright': True})
                session.detach_for_supported_release(supported[-1]['cup_xyz'][:2])
                previous,hold_pose=prepare_posture_hold(previous,'RELEASE')
                continue
            if event == 'RETREAT':
                observed = observe_stable(worker, sensor, pool, session)
                placed = observed[-1]
                if (np.linalg.norm(placed['cup_xyz'][:2]-placed['target_xyz'][:2]) > .02
                        or np.linalg.norm(placed['target_xyz'][:2]-scene['place_xy_m']) > .005):
                    raise RuntimeError('Released cup is not on the accepted target region')
                confirm(session, 'NEXT', '杯子独立站稳、两指已分离，垂直退出通道无人无障碍。', report)
                session.finish_supported_release(placed['cup_xyz'][:2])
                phases.retreat_confirmed = True
                session.contact_permission(False)
                continue
            if contact_arm_hold:
                require_support_pose(current_pose,hold_pose)
            if fine_grasp_query:rows=rows[:1]
            if phases.retreat_confirmed and current_pose[2] >= .245:
                final = observe_stable(worker, sensor, pool, session, count=12, minimum_seconds=5)
                if any(np.linalg.norm(r['cup_xyz'][:2]-r['target_xyz'][:2]) > .02 for r in final):
                    raise RuntimeError('Cup left target during final independent stability observations')
                report['final_scene_observations'] = final
                actual_trace = [item['position'] for item in session.trace]
                report['actual_motion_verified'] = actual_path_verified(sent, actual_trace)
                report['task_outcome_candidate'] = bool(report['actual_motion_verified'] and
                    contact.state == 'released' and contact.carry_authorized and contact.release_authorized)
                report['success'] = report['task_outcome_candidate']
                if not report['success']:
                    raise RuntimeError('Controller completion lacks full actual-motion/contact evidence')
                break
            attempt = dict(index=len(blocks), observation=details, raw=raw, protected_targets=rows,
                           contact_arm_hold=contact_arm_hold,
                           pregrasp_open_hold=pregrasp_open_hold,
                           fine_grasp_query=fine_grasp_query,
                           raw_sent_delta=rows-raw[:len(rows)], actual_before=actual,
                           age_before_validation_s=time.monotonic()-details['observed_monotonic_s'])
            blocks.append(attempt)
            dispatch(rows,details['observed_monotonic_s'])
            attempt['actual_after'] = session.actual()
            attempt['finished_monotonic_s'] = time.monotonic()
            sent.extend(rows.tolist())
            previous = rows[-1].copy()
            active_points += len(rows)
        if not report['success']:
            raise RuntimeError('Complete ACT task did not finish within its declared point budget')
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['success'] = report['task_outcome_candidate'] = False
    finally:
        if session is not None:
            session.monitoring = False
            try:session.cancel_all()
            except Exception as exc:report['abort_error'] = str(exc)
            try:session.contact_permission(False)
            except Exception as exc:report['contact_permission_close_error'] = str(exc)
        if worker is not None:
            worker.close();json_write(args.output/'predictions.json', worker.records)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if session is not None:
            json_write(args.output/'actual_feedback.json', session.trace)
            json_write(args.output/'independent_health.json', session.health_trace)
            session.node.destroy_node();session.ros.try_shutdown()
        if health is not None:
            health.disconnect()
        report['completed_points'] = active_points
        json_write(args.output/'blocks.json', blocks)
        json_write(args.output/'report.json', report)
    return report


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'evaluation', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute and not all([args.geometry, args.receipt]):
        parser.error('execute requires current --geometry and one --receipt')
    args.model, args.output = args.model.resolve(), args.output.resolve()
    metadata = verify_model_bundle(args.model)
    evaluation = json.loads(args.evaluation.read_text())
    if not args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        report = dict(mode='offline_artifact_report', hardware_connected=False, motion_sent=False,
            gripper_sent=False, execution_permission=False,
            checkpoint_matches_evaluation=metadata.get('checkpoint_sha256') == evaluation.get('checkpoint_sha256'),
            evaluation_passed=evaluation.get('passed'), evaluation_total=evaluation.get('total'))
        json_write(args.output/'report.json', report);print(json.dumps(report));return
    receipt = json.loads(args.receipt.read_text())
    require_experimental_permission(metadata, evaluation, receipt, now=time.time())
    seal = json.loads((ROOT/'package_sha256.json').read_text())
    for name, expected in seal.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise PermissionError('Execution package source changed: '+name)
    consume_attempt(args.receipt, args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    report = execute_task(args, metadata, evaluation, receipt)
    print(json.dumps(dict(success=report['success'], output=str(args.output),
                          read_only_restoration_pending=True)), flush=True)
    if not report['success']:
        raise SystemExit(1)


if __name__ == '__main__':main()
