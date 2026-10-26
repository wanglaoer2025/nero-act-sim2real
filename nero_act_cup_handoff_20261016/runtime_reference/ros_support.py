"""On-site Nero ROS/MoveIt air test. Default is read-only preflight.

No SDK, CAN socket, state publisher, enable service or parameter mutation is used.
--plan calls GetMotionPlan; --execute additionally sends ExecuteTrajectory after
fresh feedback, bridge synchronisation and explicit on-site confirmation.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

ARM = [f'joint{i}' for i in range(1, 8)]
NAMES = ARM + ['gripper']
DELTA = .005
SYNC = .001


def ordered(names, values, wanted=NAMES):
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError('Duplicate or mismatched joint names')
    mapping = dict(zip(names, values))
    result = [float(mapping[n]) for n in wanted]
    if not all(math.isfinite(v) for v in result):
        raise ValueError('Nonfinite joint value')
    return result


def group_joints(urdf, srdf):
    """Resolve chains/subgroups from the running model, not the MuJoCo SRDF."""
    robot, semantic = ET.fromstring(urdf), ET.fromstring(srdf)
    joints = {j.get('name'): j for j in robot.findall('joint')}
    by_child = {j.find('child').get('link'): j for j in joints.values()}
    groups = {g.get('name'): g for g in semantic.findall('group')}

    def active(j):
        return j.get('type') != 'fixed' and j.find('mimic') is None

    def resolve(name, ancestors=()):
        if name in ancestors:
            raise ValueError('Cyclic SRDF group')
        result = set()
        for element in groups[name]:
            if element.tag == 'joint':
                j = joints[element.get('name')]
                if active(j): result.add(j.get('name'))
            elif element.tag == 'chain':
                link, base = element.get('tip_link'), element.get('base_link')
                visited = set()
                while link != base:
                    if link in visited: raise ValueError('Cyclic URDF chain')
                    visited.add(link)
                    j = by_child[link]
                    if active(j): result.add(j.get('name'))
                    link = j.find('parent').get('link')
            elif element.tag == 'group':
                result |= resolve(element.get('name'), ancestors + (name,))
            elif element.tag == 'link':
                # An active joint leading to a link is part of that group.
                j = by_child.get(element.get('name'))
                if j is not None and active(j): result.add(j.get('name'))
        return result
    return {name: sorted(resolve(name)) for name in groups}


def validate_positions(names, points, start, target):
    if len(names) != 7 or set(names) != set(ARM):
        raise ValueError('Trajectory must contain exactly joint1..joint7, no gripper')
    rows = [ordered(names, p, ARM) for p in points]
    if len(rows) < 2: raise ValueError('Trajectory needs at least two points')
    if max(abs(a-b) for a, b in zip(rows[0], start)) > .000001:
        raise ValueError('Planner changed the measured start')
    if max(abs(a-b) for a, b in zip(rows[-1], target)) > .0005:
        raise ValueError('Trajectory did not reach the requested small goal')
    for row in rows:
        if abs(row[0]-start[0]) > DELTA+.0006:
            raise ValueError('J1 path exceeds small-motion envelope')
        if any(abs(row[i]-start[i]) > .0006 for i in range(1, 7)):
            raise ValueError('Other joints leave the small-motion envelope')
    return rows


def retime_seconds(rows):
    """Rest at every waypoint; bound quintic speed and acceleration.

    Zero velocities/accelerations at both ends give monotonic quintic segments.
    Its maxima are 1.875*d/T and <5.78*d/T^2, respectively.
    """
    times = [.25]
    for a, b in zip(rows, rows[1:]):
        distance = max(abs(x-y) for x, y in zip(a, b))
        times.append(times[-1] + max(.1, 1.875*distance/.005,
                                    math.sqrt(5.78*distance/.01)))
    if times[-1] < 4:
        times = [t * 4 / times[-1] for t in times]
    if times[-1] > 30: raise ValueError('Small trajectory duration exceeds 30 s')
    return times


def physical_result(start, target, samples, action_success):
    if not samples: raise ValueError('No actual feedback during execution')
    final = samples[-1]
    endpoint = max(abs(final[i]-target[i]) for i in range(7))
    measured_delta = final[0]-start[0]
    gripper_drift = max(abs(row[7]-start[7]) for row in samples)
    excursion = max(max(abs(row[i]-start[i]) for i in range(7)) for row in samples)
    other_axes = max(max(abs(row[i]-start[i]) for i in range(1, 7)) for row in samples)
    success = (action_success and endpoint <= .0012 and
               measured_delta >= .0035 and gripper_drift <= .001 and excursion <= .008 and
               other_axes <= .002)
    return dict(success=success, moveit_action_success=action_success,
                maximum_endpoint_error_rad=endpoint, measured_j1_delta_rad=measured_delta,
                maximum_gripper_drift_m=gripper_drift, maximum_joint_excursion_rad=excursion,
                maximum_other_axis_excursion_rad=other_axes,
                acceptance='real feedback required; mock action success alone is insufficient')


class Session:
    def __init__(self, output, report):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from action_msgs.msg import GoalStatusArray
        self.ros = rclpy
        rclpy.init()
        self.node = rclpy.create_node('nero_moveit_real_air_test')
        self.output, self.report, self.latest, self.trace = output, report, {}, []
        self.errors, self.active_goals, self.subscriptions = [], {}, []
        self.last_stamp, self.last_advance = None, None
        for topic in ['/feedback/joint_states', '/control/joint_states', '/control/gated_joint_states']:
            self.subscriptions.append(self.node.create_subscription(
                JointState, topic, lambda msg, t=topic: self.feedback(t, msg), qos_profile_sensor_data))
        for name in ['arm_controller', 'gripper_controller']:
            topic = f'/{name}/follow_joint_trajectory/_action/status'
            self.subscriptions.append(self.node.create_subscription(
                GoalStatusArray, topic,
                lambda msg, t=topic: self.active_goals.update({t: [s.status for s in msg.status_list
                                                                 if s.status in (1, 2, 3)]}), 10))
        self.goal_handle, self.submitted = None, False

    def feedback(self, topic, msg):
        try:
            wanted = NAMES
            if topic == '/control/gated_joint_states':
                if set(msg.name) == set(NAMES):
                    wanted = NAMES
                elif set(msg.name) == set(ARM):
                    wanted = ARM
                elif list(msg.name) == ['gripper']:
                    wanted = ['gripper']
                else:
                    raise ValueError('Unexpected gated command joint group')
            q = ordered(msg.name, msg.position, wanted)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
            now = time.monotonic()
            if topic == '/feedback/joint_states':
                if self.last_stamp is None or stamp > self.last_stamp:
                    self.last_advance = now
                elif stamp < self.last_stamp:
                    raise ValueError('Actual feedback ROS stamp regressed')
                self.last_stamp = stamp
                self.trace.append(dict(monotonic_s=now, ros_stamp_s=stamp, position=q))
            self.latest[topic] = dict(receipt=now, stamp=stamp, q=q)
        except (ValueError, KeyError) as exc:
            self.errors.append(f'{topic}: {exc}')

    def spin(self, seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            self.ros.spin_once(self.node, timeout_sec=min(.02, max(0, end-time.monotonic())))

    def wait(self, future, timeout=10):
        end = time.monotonic()+timeout
        while not future.done() and time.monotonic() < end:
            self.ros.spin_once(self.node, timeout_sec=.02)
        if not future.done(): raise RuntimeError('ROS request timed out')
        if future.exception(): raise future.exception()
        return future.result()

    def call(self, service, kind, request):
        client = self.node.create_client(kind, service)
        try:
            if not client.wait_for_service(timeout_sec=3):
                raise RuntimeError(f'Service unavailable: {service}')
            return self.wait(client.call_async(request))
        finally:
            self.node.destroy_client(client)

    def parameters(self, node, names):
        from rcl_interfaces.srv import GetParameters
        from rclpy.parameter import parameter_value_to_python
        response = self.call(node+'/get_parameters', GetParameters, GetParameters.Request(names=names))
        return {name: parameter_value_to_python(value) for name, value in zip(names, response.values)}

    def actual(self):
        if self.errors: raise RuntimeError(self.errors[-1])
        item = self.latest.get('/feedback/joint_states')
        if item is None or time.monotonic()-item['receipt'] > .2:
            raise RuntimeError('Missing or stale actual feedback')
        if self.last_advance is None or time.monotonic()-self.last_advance > .2:
            raise RuntimeError('Actual feedback timestamp is frozen')
        ros_age = self.node.get_clock().now().nanoseconds*1e-9-item['stamp']
        if not -.05 <= ros_age <= .25:
            raise RuntimeError(f'Actual feedback ROS clock age is invalid: {ros_age}')
        return item['q'][:]

    def scene(self):
        from moveit_msgs.srv import GetPlanningScene
        request = GetPlanningScene.Request()
        request.components.components = 1023
        return self.call('/get_planning_scene', GetPlanningScene, request).scene

    def inventory(self):
        from controller_manager_msgs.srv import ListControllers
        from rosidl_runtime_py.convert import message_to_ordereddict
        # Allow DDS discovery before applying the unchanged live-age checks.
        discovery_deadline = time.monotonic() + 5.0
        while len(self.trace) < 10 and time.monotonic() < discovery_deadline:
            self.spin(.1)
        actual = self.actual()
        if len(self.trace) < 10: raise RuntimeError('Too few actual feedback samples')
        if max(max(abs(row['position'][i]-actual[i]) for i in range(7))
               for row in self.trace[-50:]) > .0007:
            raise RuntimeError('Robot is not stationary at test start')
        driver = self.parameters('/agx_arm_ctrl_single_node',
                                 ['arm_type', 'effector_type', 'can_port', 'fw_version',
                                  'control_enabled', 'speed_percent', 'auto_enable', 'use_sim_time'])
        moveit = self.parameters('/move_group', ['robot_description', 'robot_description_semantic',
                                                 'use_sim_time', 'allow_trajectory_execution'])
        if not moveit['robot_description']:
            moveit['robot_description'] = self.parameters('/robot_state_publisher',
                                                         ['robot_description'])['robot_description']
        for key, filename in [('robot_description', 'robot.urdf'),
                              ('robot_description_semantic', 'robot.srdf')]:
            (self.output/filename).write_text(moveit[key] or '')
        groups = group_joints(moveit.pop('robot_description'), moveit.pop('robot_description_semantic'))
        controllers = self.call('/controller_manager/list_controllers', ListControllers,
                                ListControllers.Request())
        scene = self.scene()
        self.scene_msg = scene
        (self.output/'planning_scene.json').write_text(json.dumps(message_to_ordereddict(scene), indent=2))
        topics = ['/feedback/joint_states', '/control/joint_states', '/control/gated_joint_states']
        owners = {t: [dict(name=p.node_name, namespace=p.node_namespace)
                      for p in self.node.get_publishers_info_by_topic(t)] for t in topics}
        direct_topics = ['/control/'+n for n in
                         ['move_j', 'move_js', 'move_p', 'move_l', 'move_c', 'move_cpv', 'move_mit']]
        direct = {t: self.node.count_publishers(t) for t in direct_topics}
        self.report.update(actual_start=actual, driver=driver, moveit_parameters=moveit,
                           groups=groups, topic_publishers=owners, direct_command_publishers=direct,
                           controllers=[dict(name=c.name, state=c.state, type=c.type)
                                        for c in controllers.controller],
                           world_object_ids=[o.id for o in scene.world.collision_objects],
                           observed_bridge_states=copy.deepcopy(self.latest),
                           geometry_scope='running MoveIt scene only; operator must verify clear air locally')
        self.report['preflight_complete'] = True
        try:
            self.ready(actual)
            self.report['execution_readiness_checks_passed'] = True
        except (RuntimeError, ValueError, KeyError) as exc:
            self.report['execution_readiness_checks_passed'] = False
            self.report['execution_blocker'] = str(exc)
        return actual

    def bridge_publishers(self):
        started = time.monotonic()
        while True:
            owners = {t: [dict(name=p.node_name, namespace=p.node_namespace)
                          for p in self.node.get_publishers_info_by_topic(t)]
                      for t in self.report['topic_publishers']}
            # DDS can discover the driver before the independent gate process.
            # Wait only for absent endpoints; ambiguous publishers still fail.
            identities_known = all(owners.values()) and not any(
                p['name'] == '_NODE_NAME_UNKNOWN_' or p['namespace'] == '_NODE_NAMESPACE_UNKNOWN_'
                for endpoints in owners.values() for p in endpoints)
            if (identities_known or any(len(v) > 1 for v in owners.values()) or
                    time.monotonic() - started >= 5.0):
                self.report['bridge_discovery_wait_s'] = time.monotonic() - started
                self.report['topic_publishers'] = owners
                return owners
            self.spin(.05)
            self.actual()
            if any(self.active_goals.values()):
                raise RuntimeError('Another trajectory goal is active')

    def ready(self, planned_start):
        self.spin(.1)
        if self.report.get('planned_monotonic_s') is not None and time.monotonic()-self.report['planned_monotonic_s'] > 60:
            raise RuntimeError('Plan is older than 60 seconds; run a fresh test')
        q = self.actual()
        if max(abs(a-b) for a, b in zip(q[:7], planned_start[:7])) > .0007:
            raise RuntimeError('Actual start moved after planning; replan required')
        if abs(q[7]-planned_start[7]) > .0005:
            raise RuntimeError('Gripper changed after planning')
        driver = self.parameters('/agx_arm_ctrl_single_node',
                                 ['arm_type', 'effector_type', 'control_enabled', 'speed_percent', 'use_sim_time'])
        if driver['arm_type'] != 'nero' or driver['effector_type'] != 'agx_gripper':
            raise RuntimeError('Running driver is not Nero with original gripper')
        # The installed gate opens only for a synchronized action; this
        # parameter records startup configuration, not live gate state.
        if driver['use_sim_time'] is not False or driver['control_enabled'] is not False:
            raise RuntimeError('Driver must start in real time with idle control gate closed')
        speed = driver['speed_percent']
        if type(speed) not in (int, float) or not 0 < speed <= 5:
            raise RuntimeError('Existing ROS driver speed_percent must be in (0,5]; script does not change it')
        if self.report['moveit_parameters']['use_sim_time'] is not False:
            raise RuntimeError('MoveIt uses simulated or unknown time')
        if self.report['moveit_parameters']['allow_trajectory_execution'] is not True:
            raise RuntimeError('MoveIt trajectory execution is not explicitly enabled')
        active = {c['name'] for c in self.report['controllers'] if c['state'] == 'active'}
        if 'arm_controller' not in active: raise RuntimeError('arm_controller is not active')
        if any(self.active_goals.values()): raise RuntimeError('Another trajectory goal is active')
        owners = self.bridge_publishers()
        if len(owners['/feedback/joint_states']) != 1 or owners['/feedback/joint_states'][0]['name'] != 'agx_arm_ctrl_single_node':
            raise RuntimeError('Actual feedback must have exactly one real driver publisher')
        if len(owners['/control/joint_states']) != 1:
            raise RuntimeError('Control bridge publisher is missing or ambiguous')
        if len(owners['/control/gated_joint_states']) != 1 or owners['/control/gated_joint_states'][0]['name'] != 'control_gate':
            raise RuntimeError('Expected the existing control_gate as the sole gated publisher')
        def driver_subscribers(topic):
            return [p for p in self.node.get_subscriptions_info_by_topic(topic)
                    if p.node_name == 'agx_arm_ctrl_single_node' and p.node_namespace == '/']
        if (len(driver_subscribers('/control/gated_joint_states')) != 1 or
                driver_subscribers('/control/joint_states')):
            raise RuntimeError('Real driver must subscribe only to gated commands, never raw bridge states')
        if any(self.node.count_publishers(t) for t in self.report['direct_command_publishers']):
            raise RuntimeError('Other direct ROS motion publishers are present')
        item = self.latest.get('/control/joint_states')
        if item is None or time.monotonic()-item['receipt'] > .2:
            raise RuntimeError('Control bridge has no fresh state; do not initialise mock state by moving hardware')
        if max(abs(a-b) for a, b in zip(item['q'][:7], q[:7])) > SYNC or abs(item['q'][7]-q[7]) > .0005:
            raise RuntimeError('Control bridge and actual robot are not synchronised')
        scene = self.scene()
        model = ordered(scene.robot_state.joint_state.name, scene.robot_state.joint_state.position)
        if max(abs(a-b) for a, b in zip(model[:7], q[:7])) > SYNC or abs(model[7]-q[7]) > .0005:
            raise RuntimeError('MoveIt current state differs from actual robot')
        # Recheck feedback after service calls, which may have taken seconds.
        if max(abs(a-b) for a, b in zip(self.actual()[:7], planned_start[:7])) > .0007:
            raise RuntimeError('Actual robot moved during readiness checks')

    def plan(self, start, group):
        from moveit_msgs.srv import GetMotionPlan, GetStateValidity
        from moveit_msgs.msg import Constraints, JointConstraint
        from rosidl_runtime_py.convert import message_to_ordereddict
        target = start[:7]
        target[0] += DELTA
        request = GetMotionPlan.Request()
        req = request.motion_plan_request
        req.group_name, req.num_planning_attempts, req.allowed_planning_time = group, 1, 3.
        req.max_velocity_scaling_factor = req.max_acceleration_scaling_factor = .05
        req.start_state = copy.deepcopy(self.scene_msg.robot_state)
        req.start_state.is_diff = False
        req.start_state.joint_state.name = NAMES
        req.start_state.joint_state.position = start
        req.start_state.joint_state.velocity = []
        req.start_state.joint_state.effort = []
        req.start_state.joint_state.header.stamp = self.node.get_clock().now().to_msg()
        goal = Constraints()
        for name, value in zip(ARM, target):
            goal.joint_constraints.append(JointConstraint(joint_name=name, position=value,
                                                          tolerance_above=.0001, tolerance_below=.0001,
                                                          weight=1.))
        req.goal_constraints = [goal]
        response = self.call('/plan_kinematic_path', GetMotionPlan, request).motion_plan_response
        if response.error_code.val != 1:
            raise RuntimeError(f'MoveIt planning failed: {response.error_code.val}')
        trajectory = response.trajectory
        if trajectory.multi_dof_joint_trajectory.points:
            raise RuntimeError('Unexpected multi-DOF trajectory')
        jt = trajectory.joint_trajectory
        rows = validate_positions(jt.joint_names, [p.positions for p in jt.points], start[:7], target)
        # Independently check interpolated states against the running collision scene.
        check_rows = [rows[0]]
        for a, b in zip(rows, rows[1:]):
            count = max(1, math.ceil(max(abs(x-y) for x, y in zip(a, b))/.0005))
            check_rows.extend([[x+(y-x)*k/count for x, y in zip(a, b)] for k in range(1, count+1)])
        if len(check_rows) > 250: raise RuntimeError('Excessively long small-motion path')
        for row in check_rows:
            validity = GetStateValidity.Request()
            validity.group_name = group
            validity.robot_state = copy.deepcopy(req.start_state)
            validity.robot_state.joint_state.position = row+[start[7]]
            result = self.call('/check_state_validity', GetStateValidity, validity)
            if not result.valid: raise RuntimeError('Collision or state-validity check failed')
        times = retime_seconds(rows)
        for point, seconds in zip(jt.points, times):
            point.time_from_start.sec = int(seconds)
            point.time_from_start.nanosec = round((seconds-int(seconds))*1e9)
            if point.time_from_start.nanosec == 1000000000:
                point.time_from_start.sec += 1; point.time_from_start.nanosec = 0
            point.velocities = [0.]*7
            point.accelerations = [0.]*7
            point.effort = []
        jt.header.stamp.sec = jt.header.stamp.nanosec = 0
        self.report.update(plan_complete=True, planning_group=group, target_rad=target,
                           delta_j1_rad=DELTA, duration_s=times[-1], checked_states=len(check_rows),
                           planned_monotonic_s=time.monotonic())
        (self.output/'trajectory.json').write_text(json.dumps(message_to_ordereddict(trajectory), indent=2))
        from moveit_msgs.msg import DisplayTrajectory
        pub = self.node.create_publisher(DisplayTrajectory, '/display_planned_path', 1)
        self.spin(.3)
        pub.publish(DisplayTrajectory(trajectory_start=req.start_state, trajectory=[trajectory]))
        self.spin(.1)
        return trajectory, target

    def execute(self, trajectory, start, target):
        from rclpy.action import ActionClient
        from moveit_msgs.action import ExecuteTrajectory
        self.ready(start)
        print('将发送：J1 +0.005 rad（约0.286度），至少4秒；其余轴保持、夹爪不下发。\n'
              '确认空夹爪和整臂周围无碰撞风险、只有ROS驱动拥有CAN控制权，现场可操作实体急停。\n'
              '只做一次向外小幅运动，不自动返回、不执行抓杯。')
        if input('现场确认后输入 MOVEIT：').strip() != 'MOVEIT':
            raise RuntimeError('On-site confirmation not given')
        self.ready(start)
        client = ActionClient(self.node, ExecuteTrajectory, '/execute_trajectory')
        if not client.wait_for_server(timeout_sec=3): raise RuntimeError('ExecuteTrajectory unavailable')
        self.ready(start)
        goal = ExecuteTrajectory.Goal(trajectory=trajectory)
        # Jazzy can optionally select controllers; Humble has no such field.
        if hasattr(goal, 'controllers'): goal.controllers = ['arm_controller']
        index = len(self.trace)
        sent_at = time.monotonic()
        self.submitted = True
        self.report['execution_goal_submitted'] = True
        self.goal_handle = self.wait(client.send_goal_async(goal), timeout=3)
        if not self.goal_handle.accepted: raise RuntimeError('Execution goal rejected')
        self.report['execution_goal_accepted'] = True
        result = self.goal_handle.get_result_async()
        while not result.done():
            self.spin(.02)
            q = self.actual()
            if (max(abs(q[i]-start[i]) for i in range(7)) > .008 or abs(q[7]-start[7]) > .001 or
                    max(abs(q[i]-start[i]) for i in range(1, 7)) > .002):
                raise RuntimeError('Actual feedback left small-motion/gripper envelope')
            if time.monotonic()-sent_at > self.report['duration_s']+5:
                raise RuntimeError('Execution timeout')
        action = self.wait(result)
        # Keep checking real feedback for a full settling window; do not accept
        # immediate success from a mock controller without the physical motion.
        deadline = max(sent_at+self.report['duration_s']+1, time.monotonic()+1)
        while time.monotonic() < deadline:
            self.spin(.02)
            q = self.actual()
            if (max(abs(q[i]-start[i]) for i in range(7)) > .008 or abs(q[7]-start[7]) > .001 or
                    max(abs(q[i]-start[i]) for i in range(1, 7)) > .002):
                raise RuntimeError('Actual feedback left envelope after action result')
        self.report['physical_result'] = physical_result(
            start, target, [r['position'] for r in self.trace[index:]],
            action.status == 4 and action.result.error_code.val == 1)
        if not self.report['physical_result']['success']:
            raise RuntimeError('MoveIt/actual-feedback motion acceptance failed')

    def cancel(self):
        if self.goal_handle is not None and self.goal_handle.accepted:
            try:
                reply = self.wait(self.goal_handle.cancel_goal_async(), timeout=2)
                self.report['cancel_requested'] = True
                self.report['cancel_acknowledged'] = bool(reply.goals_canceling)
            except Exception as exc:
                self.report['cancel_error'] = str(exc)
        if self.submitted:
            print('已结束/取消本次MoveIt请求。软件取消不保证CAN末条目标停止；异常时使用现场实体急停。')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan', action='store_true', help='Plan and display only; no motion action')
    modes.add_argument('--execute', action='store_true', help='Fresh plan and confirmed small air motion')
    parser.add_argument('--output', required=True, type=Path, help='New directory; never overwrite')
    parser.add_argument('--group', help='Running SRDF seven-axis group; auto if unambiguous')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(mode='execute' if args.execute else 'plan' if args.plan else 'preflight',
                  ros_domain_id=os.environ.get('ROS_DOMAIN_ID', '0'), preflight_complete=False,
                  plan_complete=False, execution_goal_submitted=False, execution_goal_accepted=False,
                  sdk_connected=False, enable_or_parameter_mutation=False, real_cup_test=False,
                  scope='MoveIt ROS air test; does not validate ACT sim2real', success=False)
    session = None
    try:
        session = Session(args.output, report)
        start = session.inventory()
        groups = [g for g, joints in report['groups'].items() if set(joints) == set(ARM)]
        group = args.group
        if group is None and len(groups) == 1: group = groups[0]
        if group not in groups:
            raise RuntimeError(f'Select an exact seven-axis group with --group; candidates={groups}')
        if args.plan or args.execute:
            trajectory, target = session.plan(start, group)
            if args.execute: session.execute(trajectory, start, target)
        report['success'] = True
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        print(report['error'])
        if session: session.cancel()
    finally:
        if session:
            (args.output/'actual_feedback.json').write_text(json.dumps(session.trace, indent=2))
            session.node.destroy_node()
            session.ros.shutdown()
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print(f'Report: {args.output / "report.json"}')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
