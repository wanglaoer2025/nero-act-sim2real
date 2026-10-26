"""Aligned ACT noncontact air trial: 2-Hz target queries, 20-Hz trajectory points.

Only the exported bounded air-trial contract is accepted. No general grasp permission.
"""
import argparse
import copy
import csv
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
from async_runtime import AsyncPolicyWorker, ActionPrefix, next_query_due, NEXT_QUERY_AFTER_S
from full_core import FullGate
from ros_support import Session, ARM, NAMES
from trial_contract import air_trial_allowed

ROOT = Path(__file__).resolve().parent


def execution_limits(seconds=None, steps=None):
    if steps is not None:
        if seconds is not None or type(steps) is not int or not 10 <= steps <= 400 or steps % 10:
            raise ValueError('--steps requires a multiple of 10 in [10,400], without --seconds')
        return 20., steps
    seconds = 18. if seconds is None else seconds
    if not 0 < seconds <= 20:
        raise ValueError('--seconds must be finite and in (0,20]')
    return seconds, None


def execution_pending(seconds, steps, completed_steps, elapsed):
    if steps is None:
        return elapsed < seconds
    if completed_steps >= steps:
        return False
    if elapsed + .5 > seconds:
        raise TimeoutError('step target incomplete before air-trial time limit')
    return True


def verify_distribution():
    entries = json.loads((ROOT/'package_sha256.json').read_text())
    for name, expected in entries.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'package file checksum mismatch: {name}')


def preview(directory, state_path, chunk_path, output):
    summary = json.loads((directory/'safety_summary.json').read_text())
    meta = json.loads((directory/'runtime_metadata.json').read_text())
    state = np.asarray(json.loads(state_path.read_text())['state'], dtype=float)
    raw = np.load(chunk_path, allow_pickle=False)
    gate = FullGate(summary, air_trial_allowed(meta))
    issues = gate.startup_issues(state, raw)
    if raw.shape != (100, 8) or not np.isfinite(raw).all():
        raise ValueError('preview requires finite [100,8] prediction')
    with (output/'full_chunk.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'nominal_prediction_time_s', *[f'{n}_rad' for n in ARM], 'gripper_total_width_m'])
        for i, row in enumerate(raw):
            writer.writerow([i, (i+1)/20, *row.tolist()])
    result = {'success': not issues, 'model_startup_checks_passed': not issues,
        'full_execution_ready': False, 'blocking_reasons': issues,
        'actual_state': state.tolist(), 'raw_chunk_shape': list(raw.shape),
        'raw_chunk_min': raw.min(0).tolist(), 'raw_chunk_max': raw.max(0).tolist(),
        'first_action_gap_rad': (raw[0, :7]-state[:7]).tolist(),
        'first_action_gripper_gap_m': float(raw[0, 7]-state[7]),
        'motion_sent': False, 'gripper_sent': False,
        'scope': 'all100 saved raw predictions; frozen observation, not a real closed-loop rollout',
        'state_source': str(state_path), 'prediction_source': str(chunk_path),
        'pairing_note': 'latest user-reported state and earlier saved prediction are separate sources; no claim of a new paired inference'}
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


class PredictionProcess:
    def __init__(self, interpreter, directory, output):
        self.log = (output/'policy_worker.log').open('x')
        self.process = subprocess.Popen([interpreter, str(ROOT/'stream_worker.py'), str(directory), str(output.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.records = []
        try:
            ready = self.receive(45)
            if ready.get('type') != 'ready' or ready.get('experimental_air_trial_allowed') is not True:
                raise RuntimeError('actual checkpoint refuses full execution')
            self.ready = ready
        except BaseException:
            self.close()
            raise

    def receive(self, timeout):
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not readable:
            raise TimeoutError('ACT process response timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('ACT process exited; see policy_worker.log')
        return json.loads(line)

    def predict(self, payload):
        query_sent = time.monotonic()
        self.process.stdin.write(json.dumps(payload)+'\n'); self.process.stdin.flush()
        result = self.receive(10)
        response_received = time.monotonic()
        if result.get('type') != 'prediction':
            raise RuntimeError('unexpected inference response')
        result['details'].update(query_sent_monotonic_s=query_sent,
            response_received_monotonic_s=response_received,
            inference_round_trip_s=response_received-query_sent)
        self.records.append(result)
        return np.asarray(result['chunk'], dtype=float), result['details']

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        self.log.close()


class FullSession(Session):
    def call(self, service, kind, request):
        if service not in ('/get_planning_scene', '/check_state_validity'):
            return super().call(service, kind, request)
        if not hasattr(self, 'validation_clients'):
            self.validation_clients = {}
        started = time.monotonic()
        key = (service, kind)
        # Keep DDS discovery out of repeated point checks; every result remains fresh.
        if key not in self.validation_clients:
            client = self.node.create_client(kind, service)
            if not client.wait_for_service(timeout_sec=3):
                self.node.destroy_client(client)
                raise RuntimeError(f'Service unavailable: {service}')
            self.validation_clients[key] = client
        client = self.validation_clients[key]
        try:
            return self.wait(client.call_async(request))
        finally:
            self.report.setdefault('validation_service_calls', []).append(
                {'service': service, 'started_monotonic_s': started,
                 'duration_s': time.monotonic()-started})

    def display_chunk(self, chunk):
        from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = NAMES
        for i, row in enumerate(chunk):
            point = JointTrajectoryPoint(positions=[float(v) for v in row])
            point.time_from_start.sec, point.time_from_start.nanosec = divmod((i+1)*50000000, 1000000000)
            trajectory.joint_trajectory.points.append(point)
        publisher = self.node.create_publisher(DisplayTrajectory, '/display_planned_path', 1)
        self.spin(.3)
        publisher.publish(DisplayTrajectory(trajectory_start=self.scene().robot_state, trajectory=[trajectory]))
        self.spin(.1)

    def confirm_full(self):
        replies = queue.Queue()
        def read():
            try:
                replies.put(input('非接触空中ACT序列，确认路径及急停后输入 ACT_AIR：').strip())
            except Exception:
                replies.put('')
        threading.Thread(target=read, daemon=True).start()
        end = time.monotonic()+60
        while replies.empty():
            self.spin(.02); self.actual()
            if time.monotonic() > end:
                raise TimeoutError('confirmation expired')
        if replies.get() != 'ACT_AIR':
            raise RuntimeError('confirmation not given')

    def validate_block(self, rows, group):
        from moveit_msgs.srv import GetStateValidity
        state = copy.deepcopy(self.scene().robot_state)
        state.is_diff = False
        state.joint_state.name = NAMES
        state.joint_state.velocity = []; state.joint_state.effort = []
        previous = self.actual()
        for row in rows:
            count = max(1, int(np.ceil(np.max(np.abs(np.asarray(row[:7])-previous[:7]))/.0005)))
            for k in range(1, count+1):
                position = (np.asarray(previous)+(np.asarray(row)-previous)*k/count).tolist()
                request = GetStateValidity.Request()
                request.group_name = group; request.robot_state = copy.deepcopy(state)
                request.robot_state.joint_state.position = position
                if not self.call('/check_state_validity', GetStateValidity, request).valid:
                    raise RuntimeError('full arm/gripper block fails collision/state validity')
            previous = row

    def clients(self):
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        result = [ActionClient(self.node, FollowJointTrajectory, f'/{name}/follow_joint_trajectory')
                  for name in ['arm_controller', 'gripper_controller']]
        if not all(client.wait_for_server(timeout_sec=3) for client in result):
            raise RuntimeError('arm and gripper trajectory actions are required')
        return result

    def send_block(self, clients, rows):
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        goals = []
        for names, columns in [(ARM, list(range(7))), (['gripper'], [7])]:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = names
            for i, row in enumerate(rows):
                point = JointTrajectoryPoint(positions=[float(row[c]) for c in columns])
                point.time_from_start.sec, point.time_from_start.nanosec = divmod((i+1)*50000000, 1000000000)
                goal.trajectory.points.append(point)
            goals.append(goal)
        # Send both asynchronously before waiting; never silently ignore one group.
        futures = [client.send_goal_async(goal) for client, goal in zip(clients, goals)]
        self.pending_goal_futures = futures
        self.block_handles = []
        for future in futures:
            handle = self.wait(future, timeout=2)
            self.block_handles.append(handle)
            if not handle.accepted:
                raise RuntimeError('arm/gripper block goal rejected')
        return [handle.get_result_async() for handle in self.block_handles]

    def cancel_all(self):
        handles = list(getattr(self, 'block_handles', []))
        for future in getattr(self, 'pending_goal_futures', []):
            try:
                handle = self.wait(future, timeout=2)
                if not any(handle is known for known in handles):
                    handles.append(handle)
            except Exception as exc:
                self.report.setdefault('unresolved_goal_requests', []).append(str(exc))
        for handle in handles:
            if handle.accepted:
                try:
                    self.wait(handle.cancel_goal_async(), timeout=2)
                except Exception as exc:
                    self.report.setdefault('cancel_errors', []).append(str(exc))
        print('已请求取消arm/gripper轨迹；软件取消不保证CAN末条目标立即停止，异常使用实体急停。')


def wait_plan(session, worker, minimum, timeout=15):
    end = time.monotonic()+timeout
    while True:
        session.spin(.01); session.actual()
        result = worker.latest()
        if result is not None and result['observation_step'] >= minimum:
            return result
        if time.monotonic() > end:
            raise TimeoutError('fresh live ACT prediction unavailable')


def validate_fresh_block(session, rows, group, plan, report):
    details = plan['details']
    started = time.monotonic()
    attempt = {'observation_step': plan['observation_step'],
        'observation_wall_s': details['observation_wall_s'],
        'validation_started_monotonic_s': started,
        'observation_age_before_validation_s': started-details['observation_wall_s'],
        'ready_prediction_wait_s': started-details.get('response_received_monotonic_s', started),
        'inference_ms': details.get('inference_ms'), 'pre_send_age_limit_s': .25,
        'freshness_passed': False}
    report.setdefault('block_attempts', []).append(attempt)
    try:
        session.validate_block(rows, group)
    finally:
        finished = time.monotonic()
        attempt.update(validation_finished_monotonic_s=finished,
            validation_duration_s=finished-started,
            observation_age_after_validation_s=finished-details['observation_wall_s'])
    if attempt['observation_age_after_validation_s'] > .25:
        raise RuntimeError('block validation consumed freshness budget; no stale block sent')
    attempt['freshness_passed'] = True
    return attempt


def timing_summary(blocks, started, now):
    periods = np.diff([block['sent_monotonic_s'] for block in blocks])
    return {'trajectory_point_hz': 20, 'target_query_hz': 2,
        'query_offset_within_block_s': NEXT_QUERY_AFTER_S,
        'pre_send_age_limit_s': .25, 'maximum_action_observation_age_s': .75,
        'execution_elapsed_s': now-started,
        'actual_block_hz': float(1/np.mean(periods)) if len(periods) else None,
        'maximum_block_interval_s': float(max(periods)) if len(periods) else None,
        'physical_timing_acceptance_completed': False,
        'scope': '20Hz trajectory points; actual block/service timing is recorded even on failure'}


def live(args, output, report):
    directory = args.model_dir
    meta = json.loads((directory/'runtime_metadata.json').read_text())
    summary = json.loads((directory/'safety_summary.json').read_text())
    for filename, key in [('policy.ckpt', 'checkpoint_sha256'), ('dataset_stats.pkl', 'stats_sha256')]:
        if hashlib.sha256((directory/filename).read_bytes()).hexdigest() != meta[key]:
            raise ValueError('model/profile integrity mismatch')
    gate = FullGate(summary, air_trial_allowed(meta))
    if not gate.allowed:
        raise RuntimeError('checkpoint has no bounded air-trial permission; no override offered')
    if summary.get('trial_profile') != meta['trial_profile'] or args.seconds > meta['trial_profile']['maximum_seconds']:
        raise RuntimeError('trial safety profile differs from checkpoint contract or duration exceeds scope')
    report.update(execution_scope=meta['execution_scope'], trial_profile=meta['trial_profile'],
                  formal_hardware_deployment_allowed=meta.get('hardware_deployment_allowed'),
                  experimental_air_trial_allowed=True, model_checkpoint_sha256=meta['checkpoint_sha256'],
                  requested_steps=args.steps, nominal_requested_duration_s=args.steps / 20 if args.steps else args.seconds,
                  execution_time_limit_s=args.seconds, submitted_step_count=0, completed_step_count=0)
    process = worker = session = None
    execution_started = None
    try:
        process = PredictionProcess(args.policy_python, directory, output)
        if process.ready['checkpoint_sha256'] != meta['checkpoint_sha256']:
            raise RuntimeError('worker loaded another checkpoint')
        if process.ready['trial_profile'] != meta['trial_profile']:
            raise RuntimeError('worker checkpoint and external trial profile mismatch')
        session = FullSession(output, report)
        start = session.inventory()
        active = {c['name'] for c in report['controllers'] if c['state'] == 'active'}
        if not report['execution_readiness_checks_passed']:
            raise RuntimeError('ROS readiness: '+report.get('execution_blocker', 'unknown local readiness failure'))
        if 'gripper_controller' not in active:
            raise RuntimeError('gripper_controller must be active for this arm+gripper sequence')
        issues = gate.startup_issues(start)
        if issues:
            raise RuntimeError('; '.join(issues))
        candidates = [g for g, joints in report['groups'].items() if set(joints) == set(ARM)]
        group = args.group or (candidates[0] if len(candidates) == 1 else None)
        if group not in candidates:
            raise RuntimeError(f'choose --group from {candidates}')
        worker = AsyncPolicyWorker(process.predict)
        def snapshot():
            return {'state': session.actual(), 'observed_monotonic_s': time.monotonic()}
        worker.submit(-2, snapshot()); wait_plan(session, worker, -2)
        worker.submit(-1, snapshot()); first = wait_plan(session, worker, -1)
        issues = gate.startup_issues(session.actual(), first['chunk'])
        if issues:
            raise RuntimeError('; '.join(issues))
        # Preview every raw arm/gripper point, not just one selected endpoint.
        np.save(output/'initial_raw_chunk.npy', first['chunk'])
        session.validate_block(first['chunk'].tolist(), group)
        session.display_chunk(first['chunk'])
        session.confirm_full(); session.ready(start)
        clients = session.clients()
        # Observation used for execution is refreshed after operator confirmation.
        worker.submit(0, snapshot()); first = wait_plan(session, worker, 0)
        prefix = ActionPrefix(10, 15, .75)
        origin = first['details']['observation_wall_s']; previous = np.asarray(session.actual())
        execution_started = time.monotonic()
        next_control_step = 0; report['executed_blocks'] = []
        begin = len(session.trace)
        while execution_pending(args.seconds, args.steps, report['completed_step_count'],
                                time.monotonic()-execution_started):
            control_step = max(next_control_step, int((time.monotonic()-origin)/.05))
            latest = worker.latest()
            state = session.actual(); rows = []; selections = []; raw_rows = []
            for k in range(10):
                raw, selection = prefix.action(control_step+k, latest, time.monotonic()+k*.05)
                if raw is None:
                    raise RuntimeError(f'prediction prefix unavailable: {selection}')
                safe = gate.point(raw, state, previous)
                raw_rows.append(raw.tolist())
                rows.append(safe.tolist()); selections.append(selection); previous = safe
            attempt = validate_fresh_block(session, rows, group, latest, report)
            gate.feedback_check(session.actual(), rows[0])
            report['full_execution_ready'] = True
            sent = time.monotonic()
            before = session.actual()
            futures = session.send_block(clients, rows)
            attempt.update(sent_monotonic_s=sent,
                observation_age_at_send_s=sent-latest['details']['observation_wall_s'],
                goal_ack_duration_s=time.monotonic()-sent)
            report['execution_goal_submitted'] = True
            report['submitted_step_count'] += len(rows)
            report['executed_blocks'].append({'sent_monotonic_s': sent, 'control_step': control_step,
                'observation_step': latest['observation_step'], 'points': rows, 'selection': selections,
                'raw_actions': raw_rows, 'observation_wall_s': latest['details']['observation_wall_s'],
                'policy_chunk_sha256': hashlib.sha256(latest['chunk'].tobytes()).hexdigest(), 'actual_before': before})
            queried = False
            while not all(f.done() for f in futures) or time.monotonic()-sent < .5:
                session.spin(.005)
                elapsed = time.monotonic()-sent
                expected_index = min(9, max(0, int(elapsed/.05)))
                gate.feedback_check(session.actual(), rows[expected_index])
                if next_query_due(elapsed, queried):
                    observation = snapshot()
                    observed_step = int((observation['observed_monotonic_s']-origin)/.05)
                    worker.submit(observed_step, observation); queried = True
                if elapsed > 1.5:
                    raise TimeoutError('arm/gripper block execution timed out')
            results = [session.wait(f) for f in futures]
            if any(r.status != 4 or r.result.error_code != 0 for r in results):
                raise RuntimeError('arm/gripper action failed')
            gate.feedback_check(session.actual(), rows[-1])
            after = session.actual()
            delta = np.asarray(rows[-1][:7])-before[:7]
            axis = int(np.argmax(abs(delta)))
            signed_progress = float((after[axis]-before[axis])*np.sign(delta[axis]))
            report['executed_blocks'][-1].update(actual_after=after,
                signed_dominant_axis_progress_rad=signed_progress,
                direction_response_verified=bool(abs(delta[axis]) > .0015 and signed_progress > .001))
            attempt['block_completed_monotonic_s'] = time.monotonic()
            report['completed_step_count'] += len(rows)
            next_control_step = control_step+10
        states = np.asarray([row['position'] for row in session.trace[begin:]])
        displacement = np.max(np.abs(states[:, :7]-np.asarray(start[:7]))) if len(states) else 0.
        report['measured_maximum_arm_displacement_rad'] = float(displacement)
        report['actual_motion_observed'] = bool(displacement > .001)
        report['success'] = bool(displacement > .001 and any(
            block.get('direction_response_verified') for block in report['executed_blocks']))
        check_endpoint = (args.steps / 20 if args.steps else args.seconds) >= 15
        report['sequence_completion_checked'] = check_endpoint
        if check_endpoint:
            endpoint = np.asarray(meta['trial_profile']['endpoint_state'])
            final = np.asarray(session.actual())
            metrics = {'endpoint_max_error_rad': float(np.max(abs(final[:7]-endpoint[:7]))),
                'j1_progress_rad': float(final[0]-start[0]),
                'gripper_span_m': float(np.ptp(states[:, 7])) if len(states) else 0.,
                'end_gripper_error_m': float(abs(final[7]-endpoint[7]))}
            completed = (metrics['endpoint_max_error_rad'] <= .008 and metrics['j1_progress_rad'] >= .025
                and metrics['gripper_span_m'] >= .003 and metrics['end_gripper_error_m'] <= .004)
            report.update(sequence_motion_metrics=metrics, sequence_completed=bool(completed))
            report['success'] = bool(report['success'] and completed)
    finally:
        if args.steps is not None:
            report['step_target_completed'] = report.get('completed_step_count', 0) == args.steps
            report['success'] = bool(report.get('success') and report['step_target_completed'])
        if execution_started is not None:
            report['timing'] = timing_summary(report.get('executed_blocks', []),
                                             execution_started, time.monotonic())
        if session:
            session.cancel_all()
        if worker:
            try:
                worker.close()
            except Exception as exc:
                report['worker_close_error'] = str(exc)
            (output/'inference_timing.json').write_text(json.dumps(worker.records, indent=2))
        if process:
            (output/'policy_predictions.json').write_text(json.dumps(process.records, indent=2))
            process.close()
        if session:
            (output/'actual_feedback.json').write_text(json.dumps(session.trace, indent=2))
            session.node.destroy_node(); session.ros.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument('--return-to-start', action='store_true',
                        help='separate MoveIt return to trained start; default plan-only, execute requires RETURN_START')
    recovery.add_argument('--restore-gripper-start', action='store_true',
                          help='separate empty-gripper opening recovery; execute requires GRIPPER_START')
    parser.add_argument('--model-dir', type=Path, default=ROOT/'model')
    parser.add_argument('--state', type=Path, default=ROOT/'latest_reported_state.json')
    parser.add_argument('--chunk', type=Path, default=ROOT/'reference_preview/raw_act_chunk.npy')
    parser.add_argument('--policy-python', default='/home/user/nero/bin/python')
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--seconds', type=float, help='wall-clock duration, default 18; maximum 20 seconds')
    duration.add_argument('--steps', type=int,
                          help='exact completed 20-Hz points, multiple of 10; 20-second wall-clock watchdog retained')
    parser.add_argument('--group')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        args.seconds, args.steps = execution_limits(args.seconds, args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.return_to_start or args.restore_gripper_start) and args.steps is not None:
        parser.error('--steps applies only to the ACT sequence, not recovery')
    verify_distribution()
    args.model_dir = args.model_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'mode': 'full_loop_execute' if args.execute else 'full_chunk_preview', 'success': False, 'full_execution_ready': False,
              'execution_goal_submitted': False, 'sdk_connected': False, 'scope': 'full arm/gripper trajectory framework; no raw-action rebasing'}
    try:
        if args.restore_gripper_start:
            from restore_gripper import run_gripper_return
            run_gripper_return(args, args.output, report, FullSession)
        elif args.return_to_start:
            from restore_start import run_return
            run_return(args, args.output, report, FullSession)
        elif args.execute:
            live(args, args.output, report)
        else:
            report.update(preview(args.model_dir, args.state, args.chunk, args.output))
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report.get('success') else 2


if __name__ == '__main__':
    raise SystemExit(main())
