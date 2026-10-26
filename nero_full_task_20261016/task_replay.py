"""Exact original-amplitude simulation teacher replay, empty scene only.

Default validates/displays the whole path without motion. --execute additionally
requires FULL_TASK_EMPTY and fresh health, feedback and ROS gate checks.
This is trajectory replay, not ACT inference or contact-grasp validation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/user/wq/nero_cup_retrained_v2_20261016')
AIR = Path('/home/user/wq/nero_aligned_air_sequence_20261016_onsite')


def initial_delay(start, first):
    delta = abs(np.asarray(first)-np.asarray(start))
    needed = max(.05, float(np.max(delta[:7]))/.04, float(delta[7])/.01)
    return max(.05, math.ceil((needed-1e-8)/.05)*.05)


def validate_task(start, rows, initial, limits, first_point_s=.05):
    a, q, initial, limits = [np.asarray(v, float) for v in (start, rows, initial, limits)]
    if (a.shape != (8,) or initial.shape != (8,) or q.ndim != 2 or q.shape[1] != 8
        or len(q) < 2 or limits.shape != (7, 2)
        or not np.isfinite(np.r_[a, q.ravel(), initial, limits.ravel()]).all()
        or np.any(limits[:, 0] >= limits[:, 1])):
        raise ValueError('Invalid finite task data')
    if max(abs(a[:7]-initial[:7])) > .005 or abs(a[7]-initial[7]) > .004:
        raise ValueError('Actual pose differs from original simulation start; no automatic alignment')
    if (np.any(q[:, :7] < limits[:, 0]) or np.any(q[:, :7] > limits[:, 1])
        or np.any(q[:, 7] < 0) or np.any(q[:, 7] > .095)):
        raise ValueError('Task leaves original physical joint/gripper limits')
    delta = np.diff(np.r_[a[None], q], axis=0)
    if (not math.isfinite(first_point_s) or not .05 <= first_point_s <= .5
        or max(abs(q[0, :7]-a[:7])) > .005 or abs(q[0, 7]-a[7]) > .004):
        raise ValueError('Task first point is discontinuous')
    # One microradian/metre only accounts for stored float32 rounding.
    if (np.any(abs(delta[0, :7]) > first_point_s*.04+.000001)
        or abs(delta[0, 7]) > first_point_s*.01+.000001
        or np.any(abs(delta[1:, :7]) > .002001) or np.any(abs(delta[1:, 7]) > .000501)):
        raise ValueError('Task exceeds original 0.04 rad/s or 0.5 mm/point rate')
    return q


def check_clearance(clearance, now):
    if ((clearance.get('cup_absent_from_swept_volume') is not True
         and clearance.get('cup_absent_in_new_camera_frame') is not True)
        or clearance.get('operator_swept_volume_clear') is not True
        or not 0 <= now-clearance['frame_capture_wall_s'] <= 90
        or hashlib.sha256(Path(clearance['image']).read_bytes()).hexdigest() != clearance['image_sha256']):
        raise RuntimeError('A fresh verified clear-scene record is required; no motion submitted')


def check_action_results(futures):
    for future in futures:
        if future.done():
            if future.exception(): raise future.exception()
            result = future.result()
            if result.status != 4 or result.result.error_code != 0:
                raise RuntimeError('An arm/gripper controller failed; cancel both trajectories')


def task_session_class(base):
    class TaskSession(base):
        execution_monitor = None
        awaiting_ack = cancelling = False

        def wait(self, future, timeout=10):
            if self.execution_monitor is None or self.cancelling:
                return super().wait(future, timeout)
            deadline = time.monotonic()+timeout
            if self.awaiting_ack:
                deadline = min(deadline, self.scheduled_monotonic-.1)
            while not future.done():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Action acknowledgement missed the scheduled-start deadline')
                self.execution_monitor()
            if self.awaiting_ack and time.monotonic() >= deadline:
                raise TimeoutError('Action acknowledgement missed the scheduled-start deadline')
            if future.exception(): raise future.exception()
            return future.result()

        def send_block(self, clients, rows):
            self.pending_goal_futures = []; self.block_handles = []
            session = self
            class RecordedClient:
                def __init__(self, client): self.client = client
                def send_goal_async(self, goal):
                    goal.trajectory.header.stamp = session.common_ros_start
                    for point in goal.trajectory.points:
                        stamp = point.time_from_start
                        ns = stamp.sec*1000000000+stamp.nanosec+round((session.first_delay_s-.05)*1e9)
                        stamp.sec, stamp.nanosec = divmod(ns,1000000000)
                    future = self.client.send_goal_async(goal)
                    session.pending_goal_futures.append(future)
                    return future
            self.awaiting_ack = True
            try:
                return super().send_block([RecordedClient(c) for c in clients], rows)
            finally:
                self.awaiting_ack = False

        def close_control_gate(self):
            from std_srvs.srv import SetBool
            if not self.gate_closer.wait_for_service(timeout_sec=0.):
                self.report['cancel_gate_close_error'] = 'Driver gate service unavailable'
                return None
            return self.gate_closer.call_async(SetBool.Request(data=False))

        def cancel_all(self):
            self.cancelling = True
            gate_future = None
            try: gate_future = self.close_control_gate()
            except Exception as exc: self.report['cancel_gate_close_error'] = str(exc)
            cancelled = set(); replies = []
            def cancel_handle(handle):
                if handle.accepted and id(handle) not in cancelled:
                    cancelled.add(id(handle)); replies.append(handle.cancel_goal_async())
            def late_ack(future):
                try: cancel_handle(future.result())
                except Exception as exc: self.report.setdefault('cancel_errors',[]).append(str(exc))
            for handle in getattr(self,'block_handles',[]):
                try: cancel_handle(handle)
                except Exception as exc: self.report.setdefault('cancel_errors',[]).append(str(exc))
            pending = list(getattr(self,'pending_goal_futures',[]))
            for future in pending:
                if future.done(): late_ack(future)
                else: future.add_done_callback(late_ack)
            deadline = time.monotonic()+2
            while time.monotonic() < deadline and (any(not f.done() for f in pending+replies)
                    or gate_future is not None and not gate_future.done()):
                try: self.spin(.01)
                except Exception as exc:
                    self.report['cancel_spin_error'] = str(exc)
                    break
            self.report['cancel_requested_for_handles'] = len(cancelled)
            self.report['unresolved_goal_acknowledgements'] = sum(not f.done() for f in pending)
            self.report['unresolved_cancellation_replies'] = sum(not f.done() for f in replies)
            if gate_future is not None and gate_future.done() and not gate_future.exception():
                self.report['driver_gate_closed_for_abort'] = bool(gate_future.result().success)
            print('已请求关闭驱动命令门并取消本次轨迹；软件取消不保证CAN最后目标瞬间停止。')
    return TaskSession


def tracking_ok(actual, expected, rows):
    q, target, rows = np.asarray(actual), np.asarray(expected), np.asarray(rows)
    return bool(q.shape == target.shape == (8,) and np.isfinite(q).all()
                and np.max(abs(q[:7]-target[:7])) <= .03
                and abs(q[7]-target[7]) <= .012
                and np.all(q[:7] >= rows[:, :7].min(0)-.005)
                and np.all(q[:7] <= rows[:, :7].max(0)+.005)
                and 0 <= q[7] <= .0955)


def physical_result(start, rows, samples, action_success):
    a, rows, q = np.asarray(start), np.asarray(rows), np.asarray(samples)
    if q.ndim != 2 or q.shape[1] != 8 or len(q) < 5 or not np.isfinite(q).all():
        raise ValueError('Insufficient real task feedback')
    expected_excursion = np.max(abs(rows[:, :7]-a[:7]), axis=0)
    excursion = np.max(abs(q[:, :7]-a[:7]), axis=0)
    axes = expected_excursion > .02
    traversed = bool(np.all(excursion[axes] >= .85*expected_excursion[axes]))
    arm_error = float(np.max(abs(q[-5:, :7]-rows[-1, :7])))
    grip_error = float(np.max(abs(q[-5:, 7]-rows[-1, 7])))
    span = float(np.ptp(q[:, 7])); expected_span = float(np.ptp(rows[:, 7]))
    return dict(success=bool(action_success and traversed and arm_error <= .008
                            and grip_error <= .002 and span >= max(0, expected_span-.004)),
                actual_feedback_required=True, action_success=bool(action_success),
                full_amplitude_traversed=traversed, maximum_joint_excursion_deg=np.rad2deg(excursion).tolist(),
                last5_arm_endpoint_error_rad=arm_error, last5_gripper_endpoint_error_m=grip_error,
                actual_gripper_span_m=span)


def observer():
    sys.path[:0] = ['/home/user/zyh', '/home/user/zyh/src']
    from scripts.nero_methylene_teacher import reject_sdk_send
    from pyAgxArm.protocols.can_protocol.comms.can_comm import CanCommImpl
    from pyAgxArm.protocols.can_protocol.drivers.core.arm_driver_abstract import ArmDriverAbstract
    CanCommImpl.send = ArmDriverAbstract._send_msg = ArmDriverAbstract._send_msgs = reject_sdk_send
    from pyAgxArm import create_agx_arm_config, AgxArmFactory
    from nero_pi05.robot import NeroRobot
    sdk = AgxArmFactory.create_arm(create_agx_arm_config(robot='nero', comm='can',
                                    channel='can0', firmeware_version='v120'))
    robot = NeroRobot(sdk=sdk, settle_s=.5, max_age_s=.15)
    try:
        robot.connect()
        comm = sdk._ctx.get_comm()
        if getattr(comm.send, '__func__', comm.send) is not reject_sdk_send:
            raise RuntimeError('Observer transport send guard is missing')
        robot.read_health(require_enabled=True)
        return robot
    except BaseException:
        robot.disconnect()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(AIR))
    from full_runtime import FullSession
    from restore_start import confirm_return
    report = dict(mode='full_task_execute' if args.execute else 'full_task_plan', success=False,
                  scope='Exact simulation teacher trajectory, full amplitude, empty scene',
                  execution_goal_submitted=False, action_projection_or_scaling=False,
                  act_inference=False, contact_grasp_verified=False, phase_checks=[])
    session = health = None
    try:
        seal = json.loads((ROOT/'task_seal.json').read_text())
        for path, expected in seal.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Task source hash mismatch: {path}')
        data = np.load(ROOT/'full_simulation_task.npz', allow_pickle=False)
        rows, stages = data['actions'], data['stages']
        inventory = json.loads((ROOT/'full_task_inventory.json').read_text())
        model = ET.parse(SOURCE/'simulation/scene_template.xml').getroot()
        limits = [list(map(float, model.find(f".//joint[@name='joint{i}']").get('range').split()))
                  for i in range(1,8)]
        health = observer(); session = task_session_class(FullSession)(args.output, report)
        from std_srvs.srv import SetBool
        session.gate_closer = session.node.create_client(SetBool, '/control_enable')
        start = session.inventory()
        first_delay_s = initial_delay(start, rows[0])
        rows = validate_task(start, rows, inventory['initial_state'], limits, first_delay_s)
        if len(rows) != 1229 or list(dict.fromkeys(stages.tolist())) != ['recover_open','approach','descend','close','lift','hold']:
            raise ValueError('Expected exact complete six-phase task')
        # Complete robot, both fingers, all current planning-scene objects.
        session.validate_block(rows.tolist(), '')
        session.display_chunk(rows.tolist())
        report.update(plan_complete=True, planned_monotonic_s=time.monotonic(),
                      duration_s=len(rows)/20+first_delay_s-.05, original_point_count=len(rows),
                      first_point_delay_s=first_delay_s, original_nominal_duration_s=len(rows)/20,
                      original_maximum_joint_excursion_deg=inventory['max_joint_excursion_deg'],
                      task_source_hashes=seal, stages=inventory['stages'])
        if args.execute:
            clients = session.clients()
            confirm_return(session, 'FULL_TASK_EMPTY',
                '完整仿真原幅度空载回放：1229点/61.45秒，J2约64°、J4约48°、J7约78°；杯子已移走、扫掠区清空且急停可用后输入 FULL_TASK_EMPTY：')
            session.ready(start); health.read_health(require_enabled=True)
            clearance = json.loads((ROOT/'scene_clearance.json').read_text())
            check_clearance(clearance, time.time()); report['scene_clearance'] = clearance
            # Both controllers share a future ROS start; ACK waiting is monitored.
            ros_start_ns = session.node.get_clock().now().nanoseconds+1000000000
            session.common_ros_start = session.node.get_clock().now().to_msg()
            session.common_ros_start.sec, session.common_ros_start.nanosec = divmod(ros_start_ns,1000000000)
            begin = len(session.trace); submitted = time.monotonic(); sent = submitted+1.
            session.scheduled_monotonic = sent; session.first_delay_s = first_delay_s
            report.update(goal_submission_monotonic_s=submitted, scheduled_start_monotonic_s=sent,
                          goal_submission_feedback_index=begin)
            last_health = 0.; health_samples = []; checked = set(); futures = []
            boundaries = [(str(stage), int(np.flatnonzero(stages == stage)[-1]))
                          for stage in dict.fromkeys(stages.tolist())]
            def monitor():
                nonlocal last_health
                session.spin(.005); now = time.monotonic(); q = session.actual()
                check_action_results(futures)
                elapsed = now-sent
                index = min(len(rows)-1,max(0,int((elapsed-(first_delay_s-.05))/.05)-1))
                if not tracking_ok(q, rows[index], rows):
                    raise RuntimeError('Actual robot left original task envelope/tracking limits')
                if now-last_health >= .02:
                    measured = health.read_health(require_enabled=True)
                    health_samples.append({'monotonic_s':now,'state':np.asarray(measured['state']).tolist(),
                                           'controller':measured['feedback'],'all_driver_fault_checks_passed':True})
                    last_health = now
                for stage, end in boundaries:
                    if stage not in checked and elapsed >= (end+1)/20+first_delay_s-.05:
                        err = max(abs(q[i]-rows[end,i]) for i in range(7))
                        ge = abs(q[7]-rows[end,7])
                        report['phase_checks'].append(dict(stage=stage, elapsed_s=elapsed,
                            actual=q, endpoint_error_rad=err, gripper_error_m=ge,
                            success=bool(err <= .008 and ge <= .002)))
                        checked.add(stage)
                        if err > .008 or ge > .002:
                            raise RuntimeError(f'Actual stage endpoint failed: {stage}')
                if elapsed > report['duration_s']+5:
                    raise TimeoutError('Full task execution timed out')
            try:
                session.execution_monitor = monitor
                # Recheck after preparation calls and immediately before either send.
                check_clearance(clearance,time.time())
                report['execution_goal_submitted'] = True
                futures = session.send_block(clients, rows.tolist())
                while not all(f.done() for f in futures): monitor()
                check_action_results(futures)
                results = [session.wait(f) for f in futures]
                deadline = time.monotonic()+1
                while time.monotonic() < deadline: monitor()
                physical = physical_result(start, rows, [t['position'] for t in session.trace[begin:]],
                    all(r.status == 4 and r.result.error_code == 0 for r in results))
                report.update(physical_result=physical, elapsed_s=time.monotonic()-sent,
                    completed_point_count=len(rows) if physical['success'] else 0,
                    all_six_stages_completed=len(checked)==6)
                if not physical['success'] or len(checked)!=6:
                    raise RuntimeError('Full-amplitude physical task acceptance failed')
            finally:
                (args.output/'physical_health_during_execution.json').write_text(json.dumps(health_samples,indent=2))
        report['success'] = True
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if session:
            if report['execution_goal_submitted'] and not report['success']: session.cancel_all()
            (args.output/'actual_feedback.json').write_text(json.dumps(session.trace,indent=2))
            session.node.destroy_node(); session.ros.try_shutdown()
        if health: health.disconnect()
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:report.get(k) for k in ('mode','success','duration_s','physical_result','error')}))
        print('Report:', args.output/'report.json')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
