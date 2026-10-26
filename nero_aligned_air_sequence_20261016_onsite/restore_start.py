"""Separate, explicitly confirmed MoveIt return to this model's known start."""
import copy
import json
import select
import sys
import time

import numpy as np
from ros_support import ARM, NAMES, ordered, retime_seconds
from trial_contract import air_trial_allowed


def return_target(state, profile):
    state = np.asarray(state, dtype=float)
    initial = np.asarray(profile['initial_state'], dtype=float)
    if state.shape != (8,) or initial.shape != (8,) or not np.isfinite(np.r_[state, initial]).all():
        raise ValueError('Return requires finite eight-dimensional states')
    if np.max(abs(state[:7]-initial[:7])) > .05:
        raise ValueError('Known-start return is limited to 0.05 rad per axis')
    if abs(state[7]-initial[7]) > profile['initial_gripper_tolerance_m']:
        raise ValueError('Restore gripper separately; this return never commands the gripper')
    return [*initial[:7].tolist(), float(state[7])]


def return_path(names, points, start, target):
    if len(names) != 7 or set(names) != set(ARM):
        raise ValueError('Return trajectory must contain only seven arm joints')
    rows = [ordered(names, row, ARM) for row in points]
    if len(rows) < 2 or max(abs(a-b) for a,b in zip(rows[0], start[:7])) > 1e-6:
        raise ValueError('Return planner changed the measured start')
    if max(abs(a-b) for a,b in zip(rows[-1], target[:7])) > .0005:
        raise ValueError('Return planner did not reach the known start')
    for row in rows:
        if any(not min(start[i],target[i])-.0006 <= row[i] <= max(start[i],target[i])+.0006 for i in range(7)):
            raise ValueError('Return planner left the endpoint corridor')
    return rows


def within_corridor(state, start, target):
    return (len(state) == 8 and np.isfinite(state).all() and abs(state[7]-start[7]) <= .001
        and all(min(start[i],target[i])-.002 <= state[i] <= max(start[i],target[i])+.002 for i in range(7)))


def return_result(start, target, states, action_success):
    if len(states) < 5:
        raise ValueError('Insufficient physical return feedback')
    error = max(abs(row[i]-target[i]) for row in states[-5:] for i in range(7))
    return {'success': bool(action_success and error <= .0007 and
                           all(within_corridor(row,start,target) for row in states)),
            'last5_max_endpoint_error_rad': error, 'action_success': bool(action_success),
            'maximum_gripper_drift_m': max(abs(row[7]-start[7]) for row in states),
            'actual_feedback_required': True}


def plan_return(session, start, target, group):
    from moveit_msgs.srv import GetMotionPlan
    from moveit_msgs.msg import Constraints, JointConstraint, DisplayTrajectory
    from rosidl_runtime_py.convert import message_to_ordereddict
    request = GetMotionPlan.Request(); req = request.motion_plan_request
    req.group_name = group; req.num_planning_attempts = 1; req.allowed_planning_time = 3.
    req.max_velocity_scaling_factor = req.max_acceleration_scaling_factor = .05
    req.start_state = copy.deepcopy(session.scene().robot_state); req.start_state.is_diff = False
    req.start_state.joint_state.name = NAMES; req.start_state.joint_state.position = start
    req.start_state.joint_state.velocity = []; req.start_state.joint_state.effort = []
    req.start_state.joint_state.header.stamp = session.node.get_clock().now().to_msg()
    goal = Constraints()
    for name,value in zip(ARM,target[:7]):
        goal.joint_constraints.append(JointConstraint(joint_name=name,position=value,
            tolerance_above=.0001,tolerance_below=.0001,weight=1.))
    req.goal_constraints = [goal]
    response = session.call('/plan_kinematic_path',GetMotionPlan,request).motion_plan_response
    if response.error_code.val != 1:
        raise RuntimeError(f'Return planning failed: {response.error_code.val}')
    trajectory = response.trajectory
    if trajectory.multi_dof_joint_trajectory.points:
        raise ValueError('Unexpected multi-DOF return trajectory')
    jt = trajectory.joint_trajectory
    rows = return_path(jt.joint_names,[p.positions for p in jt.points],start,target)
    session.validate_block([row+[start[7]] for row in rows],group)
    times = retime_seconds(rows)
    if times[-1] > 20:
        raise ValueError('Return trajectory exceeds 20 seconds')
    for point,seconds in zip(jt.points,times):
        point.time_from_start.sec,point.time_from_start.nanosec = divmod(round(seconds*1e9),1000000000)
        point.velocities = [0.]*7; point.accelerations = [0.]*7; point.effort = []
    jt.header.stamp.sec = jt.header.stamp.nanosec = 0
    (session.output/'return_trajectory.json').write_text(json.dumps(message_to_ordereddict(trajectory),indent=2))
    session.report.update(plan_complete=True,planned_monotonic_s=time.monotonic(),
                          duration_s=times[-1],return_target=target,gripper_command_sent=False)
    publisher = session.node.create_publisher(DisplayTrajectory,'/display_planned_path',1)
    session.spin(.3)
    publisher.publish(DisplayTrajectory(trajectory_start=req.start_state,trajectory=[trajectory]))
    session.spin(.1)
    return trajectory


def confirm_return(session, phrase='RETURN_START', message=None):
    print(message or '仅回到本轮ACT训练起点；夹爪保持，不自动执行ACT。确认回程通道和急停后输入 RETURN_START：',end='',flush=True)
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        session.spin(.01); session.actual()
        if select.select([sys.stdin],[],[],0)[0]:
            if sys.stdin.readline().strip() != phrase:
                raise RuntimeError('Return confirmation not given')
            return
    raise TimeoutError('Return confirmation expired; no goal submitted')


def run_return(args, output, report, session_class):
    from rclpy.action import ActionClient
    from moveit_msgs.action import ExecuteTrajectory
    meta = json.loads((args.model_dir/'runtime_metadata.json').read_text())
    if not air_trial_allowed(meta):
        raise ValueError('Unrecognized bounded training-start profile')
    report.update(mode='return_start_execute' if args.execute else 'return_start_plan',
                  scope='Separate MoveIt known-start return, not ACT inference',
                  gripper_command_sent=False,plan_complete=False)
    session = session_class(output,report)
    try:
        start = session.inventory()
        if not report['execution_readiness_checks_passed']:
            raise RuntimeError(report.get('execution_blocker'))
        target = return_target(start,meta['trial_profile'])
        if max(abs(a-b) for a,b in zip(start[:7],target[:7])) <= .0007:
            report.update(success=True,already_at_trained_start=True,return_target=target)
            return
        groups = [g for g,joints in report['groups'].items() if set(joints) == set(ARM)]
        if len(groups) != 1: raise ValueError('Return requires one exact seven-axis group')
        trajectory = plan_return(session,start,target,groups[0])
        if not args.execute:
            report['success'] = True
            return
        client = ActionClient(session.node,ExecuteTrajectory,'/execute_trajectory')
        if not client.wait_for_server(timeout_sec=3): raise RuntimeError('ExecuteTrajectory unavailable')
        confirm_return(session); session.ready(start)
        goal = ExecuteTrajectory.Goal(trajectory=trajectory)
        if hasattr(goal,'controllers'): goal.controllers = ['arm_controller']
        begin = len(session.trace); sent = time.monotonic()
        report.update(execution_goal_submitted=True,goal_submission_index=begin,goal_submission_monotonic_s=sent)
        session.pending_goal_futures = [client.send_goal_async(goal)]
        session.block_handles = []
        handle = session.wait(session.pending_goal_futures[0],timeout=3); session.block_handles.append(handle)
        if not handle.accepted: raise RuntimeError('Return goal rejected')
        report['execution_goal_accepted'] = True; future = handle.get_result_async()
        while not future.done():
            session.spin(.01)
            if not within_corridor(session.actual(),start,target): raise RuntimeError('Actual return left safe corridor')
            if time.monotonic()-sent > report['duration_s']+5: raise TimeoutError('Return execution timed out')
        result = session.wait(future); deadline = time.monotonic()+1
        while time.monotonic() < deadline:
            session.spin(.01)
            if not within_corridor(session.actual(),start,target): raise RuntimeError('Return settling left safe corridor')
        physical = return_result(start,target,[s['position'] for s in session.trace[begin:]],
                                 result.status == 4 and result.result.error_code.val == 1)
        report['physical_result'] = physical
        if not physical['success']: raise RuntimeError('Physical return acceptance failed')
        report['success'] = True
    finally:
        if report.get('execution_goal_submitted') and not report.get('success'):
            session.cancel_all()
        (output/'actual_feedback.json').write_text(json.dumps(session.trace,indent=2))
        session.node.destroy_node(); session.ros.shutdown()
