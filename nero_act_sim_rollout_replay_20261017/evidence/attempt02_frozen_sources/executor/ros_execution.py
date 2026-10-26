"""ACT-only target transport and collision checks; no IK or motion-plan generation."""
import copy
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
from executor.transport import track_ack

import mujoco
import numpy as np

from executor.contract import FullTaskContract
from simulation.scene import build_scene
from executor.release_hold import require_support_pose,contact_hold_required

AIR = Path('/home/user/wq/nero_aligned_air_sequence_20261016_onsite')
REPLAY = Path('/home/user/wq/nero_full_task_20261016')
NAMES = [f'joint{i}' for i in range(1, 8)]+['gripper']


def checked_interpolation(start, rows):
    previous = np.asarray(start, float)
    result = []
    for row in np.asarray(rows, float):
        count = max(1, int(np.ceil(max(np.max(abs(row[:7]-previous[:7]))/.0005,
                                       abs(row[7]-previous[7])/.0005))))
        result.extend(previous+(row-previous)*i/count for i in range(1, count+1))
        previous = row
    return np.asarray(result)


def validate_in_batches(checks, submit, wait):
    # The installed service queues have depth10; draining eight avoids burst loss.
    for start in range(0, len(checks), 8):
        futures = [submit(row) for row in checks[start:start+8]]
        for future in futures:
            if not wait(future).valid:
                raise RuntimeError('ACT path fails full-robot/cup state validity; no teacher fallback')
    return len(checks)


def require_fixed_table_mount(urdf, table, table_z):
    joints = {joint.find('child').get('link'): joint for joint in ET.fromstring(urdf).findall('joint')}
    link, visited = 'base_link', set()
    while link != 'world':
        joint = joints.get(link)
        if link in visited or joint is None or joint.get('type') != 'fixed':
            raise RuntimeError('Base/table exception requires a world-fixed base_link')
        visited.add(link);link = joint.find('parent').get('link')
    if (table.id != 'nero_table' or table.header.frame_id != 'world'
            or len(table.primitives) != 1 or len(table.primitive_poses) != 1):
        raise RuntimeError('Known world-fixed table geometry required')
    shape = table.primitives[0];dimensions = np.asarray(shape.dimensions, float)
    poses = [table.pose, table.primitive_poses[0]]
    if (shape.type != 1 or dimensions.shape != (3,) or not np.isfinite(dimensions).all()
            or np.any(dimensions <= 0) or not np.isfinite(table_z)):
        raise RuntimeError('Finite fixed table box required')
    for pose in poses:
        q = pose.orientation;p = pose.position
        if (not np.isfinite([p.x,p.y,p.z,q.x,q.y,q.z,q.w]).all()
                or not np.allclose([q.x,q.y,q.z,abs(q.w)],[0,0,0,1],rtol=0,atol=1e-9)):
            raise RuntimeError('Table must match the accepted horizontal model')
    if abs(sum(p.position.z for p in poses)+dimensions[2]/2-table_z) > 1e-6:
        raise RuntimeError('Table top differs from accepted scene')


class RobotGeometry:
    def __init__(self, config):
        self.model = build_scene(config)
        self.data = mujoco.MjData(self.model)
        self.link = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')

    def pose(self, state):
        state = np.asarray(state, float)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError('Finite eight-dimensional actual/ACT state required for FK')
        self.data.qpos[:7] = state[:7]
        self.data.qpos[7:9] = [state[7]/2, -state[7]/2]
        mujoco.mj_kinematics(self.model, self.data)
        return dict(tcp=self.data.site_xpos[self.site].copy(),
                    position=self.data.xpos[self.link].copy(),
                    rotation=self.data.xmat[self.link].reshape(3, 3).copy())


class CupSceneMixin:
    def configure_initial_scene(self):
        tables = [obj for obj in self.scene().world.collision_objects if obj.id == 'nero_table']
        if len(tables) != 1:
            raise RuntimeError('Unique known static table required')
        require_fixed_table_mount((self.output/'robot.urdf').read_text(), tables[0],
                                 self.scene_config['table_top_z_m'])
        self.apply_scene([self.cup_object()], fingers_allowed=False, fixed_table_mount=True)

    def apply_scene(self, objects=(), attached=(), *, fingers_allowed=None, fixed_table_mount=False):
        from moveit_msgs.msg import PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = list(objects)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = list(attached)
        if fingers_allowed is not None or fixed_table_mount:
            from moveit_msgs.msg import AllowedCollisionEntry
            acm = copy.deepcopy(self.scene().allowed_collision_matrix)
            names = list(acm.entry_names)
            matrix = [list(row.enabled) for row in acm.entry_values]
            if len(matrix) != len(names) or any(len(row) != len(names) for row in matrix):
                raise RuntimeError('Malformed collision matrix; refusing cup contact permission')
            required = ['act_task_cup', 'gripper_link1', 'gripper_link2']
            if fixed_table_mount:required += ['base_link','nero_table']
            for name in required:
                if name not in names:
                    names.append(name)
                    for row in matrix:row.append(False)
                    matrix.append([False]*len(names))
            cup = names.index('act_task_cup')
            # Clear every cup exception, then permit exactly the two intended fingers.
            for index in range(len(names)):
                matrix[cup][index] = matrix[index][cup] = False
            if fingers_allowed:
                for name in ['gripper_link1', 'gripper_link2']:
                    finger = names.index(name)
                    matrix[cup][finger] = matrix[finger][cup] = True
            if fixed_table_mount:
                base, table = names.index('base_link'), names.index('nero_table')
                matrix[base][table] = matrix[table][base] = True
            acm.entry_names = names
            acm.entry_values = [AllowedCollisionEntry(enabled=row) for row in matrix]
            defaults = list(acm.default_entry_values)
            if 'act_task_cup' in acm.default_entry_names:
                defaults[list(acm.default_entry_names).index('act_task_cup')] = False
            else:
                acm.default_entry_names = list(acm.default_entry_names)+['act_task_cup']
                defaults.append(False)
            acm.default_entry_values = defaults
            scene.allowed_collision_matrix = acm
        client = self.cached_client('/apply_planning_scene', ApplyPlanningScene)
        if not self.wait(client.call_async(ApplyPlanningScene.Request(scene=scene))).success:
            raise RuntimeError('Collision scene update rejected')

    def cup_object(self, xy=None):
        from moveit_msgs.msg import CollisionObject
        from geometry_msgs.msg import Pose
        from shape_msgs.msg import SolidPrimitive
        obj = CollisionObject(id='act_task_cup', operation=CollisionObject.ADD)
        obj.header.frame_id = 'world'
        shape = SolidPrimitive(type=SolidPrimitive.CYLINDER,
                               dimensions=[self.scene_config['cup']['height_m'],
                                           self.scene_config['cup']['outer_diameter_m']/2])
        pose = Pose()
        pose.orientation.w = 1.
        pose.position.x, pose.position.y = map(float, xy if xy is not None else self.scene_config['pick_xy_m'])
        pose.position.z = self.scene_config['table_top_z_m']+self.scene_config['cup']['height_m']/2
        obj.primitives, obj.primitive_poses = [shape], [pose]
        return obj

    def allow_finger_contact(self, enabled):
        self.apply_scene(fingers_allowed=enabled)

    def attach_actual_source(self):
        from moveit_msgs.msg import AttachedCollisionObject
        obj = self.cup_object()
        actual = self.geometry.pose(self.actual())
        center = np.r_[self.scene_config['pick_xy_m'], self.scene_config['table_top_z_m']+.0475]
        self.held_center = actual['rotation'].T@(center-actual['position'])
        self.held_rotation = actual['rotation'].T
        pose = obj.primitive_poses[0]
        pose.position.x, pose.position.y, pose.position.z = self.held_center.tolist()
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, self.held_rotation.ravel())
        pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quaternion.tolist()
        obj.header.frame_id = 'link7'
        # MoveIt atomically removes the world object when attaching the same ID.
        self.apply_scene(attached=[AttachedCollisionObject(link_name='link7', object=obj,
                      touch_links=['gripper_link1', 'gripper_link2'])], fingers_allowed=True)

    def detach_for_supported_release(self, observed_xy):
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
        self.apply_scene([self.cup_object(observed_xy)], attached=[AttachedCollisionObject(link_name='link7',
            object=CollisionObject(id='act_task_cup', operation=CollisionObject.REMOVE))], fingers_allowed=True)
        self.held_center = self.held_rotation = None

    def finish_supported_release(self, observed_xy):
        self.apply_scene([self.cup_object(observed_xy)], fingers_allowed=False)



def session_class():
    sys.path.insert(0, str(AIR))
    sys.path.insert(0, str(REPLAY))
    from full_runtime import FullSession
    from task_replay import task_session_class
    from std_msgs.msg import Empty
    from std_srvs.srv import SetBool, Trigger

    class ActSession(CupSceneMixin, task_session_class(FullSession)):
        def __init__(self, output, report, observer, config, contact):
            self.observer, self.contact, self.scene_config = observer, contact, config
            self.monitoring, self.cancelling, self.awaiting_ack = False, False, False
            self.last_health_time = 0.
            self.last_target = None
            self.health_trace = []
            self.geometry = RobotGeometry(config)
            self.held_center = None
            self.held_rotation = None
            self.contact_hold_pose = None
            self.clients_by_service = {}
            super().__init__(output, report)
            from rclpy.signals import uninstall_signal_handlers
            # KeyboardInterrupt must reach task finally before ROS is shut down.
            uninstall_signal_handlers()
            self.gate_closer = self.node.create_client(SetBool, '/control_enable')
            self.contact_client = self.node.create_client(SetBool, '/act_contact_permission')
            self.stop_client = self.node.create_client(Trigger, '/act_task_stop')
            self.heartbeat = self.node.create_publisher(Empty, '/act_full_heartbeat', 1)

        def enable_monitoring(self):
            self.last_target = np.asarray(self.actual())
            self.monitoring = True
            self.guard()

        def guard(self):
            if not self.monitoring or self.cancelling:
                return
            if time.monotonic()-self.last_health_time < .01:
                return
            measured = self.observer.read_health(require_enabled=True)
            actual = np.asarray(self.actual())
            sdk = np.asarray(measured['state'])
            contract = FullTaskContract(dict(task_scope='complete_pick_carry_place_retreat',
                                             chunk_size=20, state_dim=8, action_dim=8))
            if (np.any(sdk[:7] < contract.lower) or np.any(sdk[:7] > contract.upper)
                    or np.max(abs(actual[:7]-sdk[:7])) > .003 or abs(actual[7]-sdk[7]) > .002):
                raise RuntimeError('Actual SDK/ROS feedback or original physical limits disagree')
            grip = measured['gripper_details']
            self.contact.update(width=grip['width_m'], force=grip['force_n'],
                                target=float(self.last_target[7]), stamp=grip['timestamp'])
            if self.contact_hold_pose is not None and contact_hold_required(self.contact):
                require_support_pose(self.geometry.pose(actual)['tcp'],self.contact_hold_pose)
            self.last_health_time = time.monotonic()
            self.health_trace.append(measured)
            self.heartbeat.publish(Empty())

        def spin(self, seconds):
            deadline = time.monotonic()+seconds
            while time.monotonic() < deadline:
                super().spin(min(.005, max(0., deadline-time.monotonic())))
                self.guard()

        def wait(self, future, timeout=10):
            deadline = time.monotonic()+timeout
            if self.awaiting_ack and not self.cancelling:
                deadline = min(deadline, self.scheduled_monotonic-.005)
            while not future.done():
                if time.monotonic() >= deadline:
                    raise TimeoutError('ROS/trajectory acknowledgement missed its deadline')
                self.spin(.002)
            if future.exception():
                raise future.exception()
            if self.awaiting_ack and time.monotonic() >= deadline:
                raise TimeoutError('Late trajectory acknowledgement')
            return future.result()

        def cached_client(self, service, kind):
            if service not in self.clients_by_service:
                client = self.node.create_client(kind, service)
                if not client.wait_for_service(timeout_sec=3):
                    raise RuntimeError('Required service unavailable: '+service)
                self.clients_by_service[service] = client
            return self.clients_by_service[service]

        def warm_services(self):
            from moveit_msgs.srv import GetStateValidity, ApplyPlanningScene
            for name, kind in [('/check_state_validity', GetStateValidity),
                               ('/apply_planning_scene', ApplyPlanningScene)]:
                self.cached_client(name, kind)
            for client in [self.gate_closer, self.contact_client, self.stop_client]:
                if not client.wait_for_service(timeout_sec=3):
                    raise RuntimeError('Required control/contact gate service unavailable')

        def close_control_gate(self):
            # Stop the arbiter before closing the driver; it must never re-enable on a late status.
            try:
                if not self.stop_client.wait_for_service(timeout_sec=.2):
                    raise RuntimeError('Permanent stop service unavailable')
                reply = self.wait(self.stop_client.call_async(Trigger.Request()), timeout=.5)
                if not reply.success:
                    raise RuntimeError('Permanent stop was not latched')
                self.report['act_stop_latched'] = True
            except Exception as exc:
                self.report['act_stop_latch_error'] = str(exc)
            return super().close_control_gate()

        def ready(self, planned_start):
            super().ready(planned_start)
            active = {c['name'] for c in self.report['controllers'] if c['state'] == 'active'}
            if not {'arm_controller', 'gripper_controller'} <= active:
                raise RuntimeError('Both trajectory controllers must be active')
            if self.node.count_publishers('/act_full_heartbeat') != 1:
                raise RuntimeError('ACT heartbeat publisher ownership is ambiguous')

        def contact_permission(self, enabled):
            result = self.wait(self.contact_client.call_async(SetBool.Request(data=bool(enabled))))
            if not result.success:
                raise RuntimeError('Contact gate rejected permission state')

        def validate_rows(self, rows):
            from moveit_msgs.srv import GetStateValidity
            state = copy.deepcopy(self.scene().robot_state)
            state.is_diff = False
            state.joint_state.name = NAMES
            state.joint_state.velocity, state.joint_state.effort = [], []
            client = self.cached_client('/check_state_validity', GetStateValidity)
            checks = checked_interpolation(self.actual(), rows)
            def submit(row):
                request = GetStateValidity.Request()
                # Empty group checks the entire robot, including gripper and attached cup.
                request.group_name = ''
                request.robot_state = copy.deepcopy(state)
                request.robot_state.joint_state.position = row.tolist()
                return client.call_async(request)
            validate_in_batches(checks, submit, self.wait)
            self.report.setdefault('collision_check_counts', []).append(len(checks))

        def cup_bottom(self, state):
            if self.held_center is None:
                return None
            pose = self.geometry.pose(state)
            center = pose['position']+pose['rotation']@self.held_center
            axis = pose['rotation']@self.held_rotation[:, 2]
            extent = .0475*abs(axis[2])+.0375*np.sqrt(max(0., 1-axis[2]**2))
            return float(center[2]-extent-self.scene_config['table_top_z_m'])

        def send_block(self, clients, rows):
            # One ROS start; both controllers reach their first point50ms later.
            self.scheduled_monotonic = time.monotonic()+.05
            stamp = self.node.get_clock().now().nanoseconds
            session = self
            class CommonStart:
                def __init__(self, client):self.client = client
                def send_goal_async(self, goal):
                    goal.trajectory.header.stamp.sec, goal.trajectory.header.stamp.nanosec = divmod(stamp, 1000000000)
                    sent = time.monotonic()
                    return track_ack(self.client.send_goal_async(goal),goal.trajectory.joint_names,
                                     sent,session.scheduled_monotonic-.005,
                                     session.report.setdefault('controller_acknowledgements',[]))
            self.awaiting_ack = True
            self.last_target = np.asarray(rows[-1])
            try:
                return FullSession.send_block(self, [CommonStart(c) for c in clients], rows)
            finally:
                self.awaiting_ack = False
    return ActSession
