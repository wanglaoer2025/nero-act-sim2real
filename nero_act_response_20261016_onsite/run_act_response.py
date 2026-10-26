"""One ACT-to-real bounded response, planned by the existing ROS/MoveIt stack.

Default previews only. All inference is real ACT; projected motion is a
diagnostic, not execution approval for the raw out-of-distribution trajectory.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

from ros_support import Session, ARM, NAMES, ordered, retime_seconds

ROOT = Path(__file__).resolve().parent
CAP = .005


def projected_target(state, chunk):
    """Preserve seven-joint direction of the tenth ACT action; bound total step."""
    if len(state) != 8 or len(chunk) < 10 or any(len(row) != 8 for row in chunk):
        raise ValueError('expected state8 and ACT chunk with at least ten action8 rows')
    if not all(math.isfinite(float(v)) for v in state) or not all(math.isfinite(float(v)) for row in chunk for v in row):
        raise ValueError('nonfinite state/prediction')
    delta = [float(chunk[9][i])-float(state[i]) for i in range(7)]
    peak = max(abs(v) for v in delta)
    if peak < .0035:
        raise ValueError('ACT predicted no sufficiently observable arm displacement; no fabricated motion')
    scale = min(1., CAP/peak)
    target = [float(state[i])+scale*delta[i] for i in range(7)]
    return target, {'raw_action_index': 9, 'raw_displacement_rad': delta,
                    'projection_scale': scale, 'maximum_projected_displacement_rad': min(peak, CAP),
                    'gripper_prediction_not_executed': True,
                    'raw_policy_execution': False,
                    'scope': 'one bounded diagnostic displacement, not replay of the simulation trajectory'}


def path_check(names, points, start, target):
    if len(names) != 7 or set(names) != set(ARM):
        raise ValueError('planner must return exactly seven arm joints')
    rows = [ordered(names, point, ARM) for point in points]
    if len(rows) < 2 or max(abs(a-b) for a, b in zip(rows[0], start[:7])) > 1e-6:
        raise ValueError('planner changed measured start')
    if max(abs(a-b) for a, b in zip(rows[-1], target)) > .0005:
        raise ValueError('planner did not reach projected ACT endpoint')
    for row in rows:
        for i in range(7):
            if not min(start[i], target[i])-.0006 <= row[i] <= max(start[i], target[i])+.0006:
                raise ValueError('planner left bounded ACT corridor')
    return rows


def actual_check(state, start, target):
    if len(state) != 8 or not all(math.isfinite(float(v)) for v in state):
        raise ValueError('invalid actual feedback')
    if abs(state[7]-start[7]) > .001:
        raise RuntimeError('unexpected gripper motion')
    for i in range(7):
        if not min(start[i], target[i])-.002 <= state[i] <= max(start[i], target[i])+.002:
            raise RuntimeError('actual feedback left bounded ACT corridor')


def response_result(start, target, states, action_success):
    if len(states) < 5:
        raise ValueError('insufficient real feedback')
    for state in states:
        actual_check(state, start, target)
    delta = [target[i]-start[i] for i in range(7)]
    axis = max(range(7), key=lambda i: abs(delta[i]))
    tail = states[-5:]
    error = max(abs(state[i]-target[i]) for state in tail for i in range(7))
    progress = min((state[axis]-start[axis])*math.copysign(1., delta[axis]) for state in tail)
    return {'success': bool(action_success and error <= .0012 and progress >= .0035),
            'dominant_model_axis': ARM[axis], 'signed_actual_progress_rad': progress,
            'tail_maximum_endpoint_error_rad': error, 'action_success': bool(action_success),
            'physical_feedback_required': True, 'grasp_success_claimed': False}


def verify_package():
    manifest = json.loads((ROOT/'package_sha256.json').read_text())
    for name, expected in manifest.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'package integrity failed: {name}')


class ResponseSession(Session):
    def worker(self, interpreter, mode):
        console = (self.output/f'worker_{mode}.log').open('x')
        process = subprocess.Popen([interpreter, str(ROOT/'policy_worker.py'), mode,
                                    '--output', str(self.output)], stdout=console, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+45
            while process.poll() is None:
                self.spin(.02)
                self.actual()
                if time.monotonic() > deadline:
                    raise RuntimeError(f'{mode} worker timed out')
            if process.returncode:
                raise RuntimeError(f'{mode} worker failed; see worker_{mode}.log')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            console.close()

    def plan_response(self, start, target, group):
        from moveit_msgs.srv import GetMotionPlan, GetStateValidity
        from moveit_msgs.msg import Constraints, JointConstraint, DisplayTrajectory
        from rosidl_runtime_py.convert import message_to_ordereddict
        # Joint limits are checked independently of the learned data distribution.
        urdf = ET.fromstring((self.output/'robot.urdf').read_text())
        joints = {j.get('name'): j for j in urdf.findall('joint')}
        for i, name in enumerate(ARM):
            limits = joints[name].find('limit')
            if limits is None or not float(limits.get('lower')) <= target[i] <= float(limits.get('upper')):
                raise ValueError(f'projected endpoint violates {name} limits')
        request = GetMotionPlan.Request()
        req = request.motion_plan_request
        req.group_name, req.num_planning_attempts, req.allowed_planning_time = group, 1, 3.
        req.max_velocity_scaling_factor = req.max_acceleration_scaling_factor = .05
        req.start_state = copy.deepcopy(self.scene_msg.robot_state)
        req.start_state.is_diff = False
        req.start_state.joint_state.name = NAMES
        req.start_state.joint_state.position = start
        req.start_state.joint_state.velocity = []; req.start_state.joint_state.effort = []
        req.start_state.joint_state.header.stamp = self.node.get_clock().now().to_msg()
        goal = Constraints()
        for name, value in zip(ARM, target):
            goal.joint_constraints.append(JointConstraint(joint_name=name, position=value,
                tolerance_above=.0001, tolerance_below=.0001, weight=1.))
        req.goal_constraints = [goal]
        response = self.call('/plan_kinematic_path', GetMotionPlan, request).motion_plan_response
        if response.error_code.val != 1:
            raise RuntimeError(f'MoveIt planning failed: {response.error_code.val}')
        trajectory = response.trajectory
        if trajectory.multi_dof_joint_trajectory.points:
            raise ValueError('unexpected multi-DOF trajectory')
        jt = trajectory.joint_trajectory
        rows = path_check(jt.joint_names, [point.positions for point in jt.points], start, target)
        check_rows = [rows[0]]
        for a, b in zip(rows, rows[1:]):
            count = max(1, math.ceil(max(abs(x-y) for x, y in zip(a, b))/.0005))
            check_rows.extend([[x+(y-x)*k/count for x, y in zip(a, b)] for k in range(1, count+1)])
        if len(check_rows) > 250:
            raise ValueError('excessive path for bounded probe')
        for row in check_rows:
            validity = GetStateValidity.Request()
            validity.group_name = group
            validity.robot_state = copy.deepcopy(req.start_state)
            validity.robot_state.joint_state.position = row+[start[7]]
            if not self.call('/check_state_validity', GetStateValidity, validity).valid:
                raise RuntimeError('collision/state validity rejected ACT response path')
        times = retime_seconds(rows)
        for point, seconds in zip(jt.points, times):
            ns = round(seconds*1e9)
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 1000000000)
            point.velocities = [0.]*7; point.accelerations = [0.]*7; point.effort = []
        jt.header.stamp.sec = jt.header.stamp.nanosec = 0
        (self.output/'trajectory.json').write_text(json.dumps(message_to_ordereddict(trajectory), indent=2))
        self.report.update(plan_complete=True, target_rad=target, duration_s=times[-1],
            checked_states=len(check_rows), planned_monotonic_s=time.monotonic())
        pub = self.node.create_publisher(DisplayTrajectory, '/display_planned_path', 1)
        self.spin(.3)
        pub.publish(DisplayTrajectory(trajectory_start=req.start_state, trajectory=[trajectory]))
        self.spin(.1)
        return trajectory

    def confirm(self):
        replies = queue.Queue()
        def ask():
            try:
                replies.put(input('现场确认空中路径后输入 ACT_RESPONSE：').strip())
            except Exception:
                replies.put('')
        threading.Thread(target=ask, daemon=True).start()
        deadline = time.monotonic()+60
        while replies.empty():
            self.spin(.02); self.actual()
            if time.monotonic() > deadline:
                raise RuntimeError('confirmation expired; no goal submitted')
        if replies.get() != 'ACT_RESPONSE':
            raise RuntimeError('confirmation not given')

    def execute_response(self, trajectory, start, target):
        from rclpy.action import ActionClient
        from moveit_msgs.action import ExecuteTrajectory
        self.ready(start)
        print('真实ACT输出经方向保持限幅：七轴最多0.005rad，慢速单次运动；夹爪不下发，不自动回位。')
        print('当前示范范围检查结果见prediction.json；本轮不执行原始绝对目标，不声称抓取/原轨迹迁移通过。')
        print('确认ROS独占CAN、扫掠范围清空、实体急停可用；取消动作不能保证CAN末条目标立即停止。')
        self.confirm()
        self.ready(start)
        camera = json.loads((self.output/'camera.json').read_text())
        if time.monotonic()-camera['host_monotonic_s'] > 60:
            raise RuntimeError('camera observation expired; rerun locally')
        client = ActionClient(self.node, ExecuteTrajectory, '/execute_trajectory')
        if not client.wait_for_server(timeout_sec=3):
            raise RuntimeError('ExecuteTrajectory server absent')
        self.ready(start)
        goal = ExecuteTrajectory.Goal(trajectory=trajectory)
        if hasattr(goal, 'controllers'):
            goal.controllers = ['arm_controller']
        begin = len(self.trace); sent = time.monotonic()
        self.submitted = True
        self.report.update(execution_goal_submitted=True, goal_submission_index=begin,
                           goal_submission_monotonic_s=sent)
        self.goal_handle = self.wait(client.send_goal_async(goal), timeout=3)
        if not self.goal_handle.accepted:
            raise RuntimeError('ACT response trajectory rejected')
        self.report['execution_goal_accepted'] = True
        future = self.goal_handle.get_result_async()
        while not future.done():
            self.spin(.02); actual_check(self.actual(), start, target)
            if time.monotonic()-sent > self.report['duration_s']+5:
                raise RuntimeError('response execution timeout')
        action = self.wait(future)
        until = max(sent+self.report['duration_s']+1, time.monotonic()+1)
        while time.monotonic() < until:
            self.spin(.02); actual_check(self.actual(), start, target)
        states = [r['position'] for r in self.trace[begin:]]
        result = response_result(start, target, states, action.status == 4 and action.result.error_code.val == 1)
        self.report['physical_result'] = result
        if not result['success']:
            raise RuntimeError('model-to-physical-response acceptance failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='confirmed bounded ACT response; default preview only')
    parser.add_argument('--policy-python', default='/home/user/nero/bin/python', help='existing Python with torch, pyzed, cv2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group')
    args = parser.parse_args()
    verify_package()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(mode='act_response_execute' if args.execute else 'act_response_preview',
        success=False, preflight_complete=False, plan_complete=False,
        execution_goal_submitted=False, execution_goal_accepted=False,
        model='simulation ACT v5; state8/action8', sdk_connected=False,
        enable_or_parameter_mutation=False, raw_policy_execution=False,
        grasp_success_claimed=False, real_cup_test=False,
        scope='bounded ACT-output direction to real feedback; not original simulation trajectory replay',
        ros_domain_id=os.environ.get('ROS_DOMAIN_ID', '0'))
    session = None
    try:
        session = ResponseSession(args.output, report)
        start = session.inventory()
        initial_stationary_state = start[:]
        if not report['execution_readiness_checks_passed'] and args.execute:
            raise RuntimeError(report['execution_blocker'])
        candidates = [g for g, joints in report['groups'].items() if set(joints) == set(ARM)]
        group = args.group or (candidates[0] if len(candidates) == 1 else None)
        if group not in candidates:
            raise RuntimeError(f'select --group from {candidates}')
        session.worker(args.policy_python, 'capture')
        # Read actual joints immediately after capture, keeping ROS spinning.
        session.spin(.02)
        start = session.actual()
        if max(abs(a-b) for a, b in zip(start, initial_stationary_state)) > .0007:
            raise RuntimeError('robot changed pose during camera capture; rerun while stationary')
        camera = json.loads((args.output/'camera.json').read_text())
        report['camera_to_state_read_host_gap_s'] = time.monotonic()-camera['host_monotonic_s']
        report['input_pairing_scope'] = 'host times and stationary-state check; not hardware synchronization'
        (args.output/'input_state.json').write_text(json.dumps({'state': start,
            'feedback': session.latest['/feedback/joint_states'], 'host_monotonic_s': time.monotonic()}, indent=2))
        session.worker(args.policy_python, 'infer')
        prediction = json.loads((args.output/'prediction.json').read_text())
        if prediction['state'] != [float(v) for v in start]:
            # Worker uses float32 normalization; allow only conversion precision.
            if max(abs(a-b) for a, b in zip(prediction['state'], start)) > 1e-6:
                raise ValueError('inference state differs from selected real feedback')
        if max(abs(a-b) for a, b in zip(session.actual(), start)) > .0007:
            raise RuntimeError('robot moved during inference')
        target, projection = projected_target(start, prediction['raw_first10'])
        report.update(projection=projection, outside_demonstration_joints=prediction['outside_demonstration_joints'],
                      checkpoint_sha256=prediction['checkpoint_sha256'])
        trajectory = session.plan_response(start, target, group)
        print(json.dumps({'input_state': start, 'raw_act_action10': prediction['raw_first10'][9],
                          'bounded_target': target, 'projection': projection}, indent=2))
        if args.execute:
            session.execute_response(trajectory, start, target)
        report['success'] = True
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        print(report['error'])
        if session:
            session.cancel()
    finally:
        if session:
            (args.output/'actual_feedback.json').write_text(json.dumps(session.trace, indent=2))
            session.node.destroy_node(); session.ros.shutdown()
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print(f'Report: {args.output / "report.json"}')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
