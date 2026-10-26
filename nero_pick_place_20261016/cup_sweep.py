"""Check the carried 95x75mm cylinder against the historical table plane."""
import math
import numpy as np


def rotate_vector(vector, quaternion):
    q = np.asarray(quaternion, float)
    v = np.asarray(vector, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-6:
        raise ValueError('Invalid unit FK quaternion')
    return v + 2*np.cross(q[:3], np.cross(q[:3], v) + q[3]*v)


def cylinder_bounds(center, axis):
    center, axis = np.asarray(center, float), np.asarray(axis, float)
    if (center.shape != (3,) or axis.shape != (3,)
            or not np.isfinite(np.r_[center, axis]).all() or np.linalg.norm(axis) < .5):
        raise ValueError('Invalid finite cup geometry')
    axis = axis / np.linalg.norm(axis)
    extent = .0475 * np.abs(axis) + .0375 * np.sqrt(np.maximum(0, 1-axis**2))
    return center-extent, center+extent


def check_cup_floor(center, axis):
    lower, _ = cylinder_bounds(center, axis)
    # One micrometre accounts only for IK/FK floating point precision at contact.
    if lower[2] < -1e-6:
        raise ValueError(f'Carried cup intersects historical table: bottom={lower[2]*1000:.4f}mm')
    return float(lower[2])


def validate_cup_trajectory(session, trajectory, start, *, tcp_offset=(.135, 0, 0)):
    from copy import deepcopy
    from moveit_msgs.srv import GetPositionFK
    client = session.node.create_client(GetPositionFK, '/compute_fk')
    if not client.wait_for_service(timeout_sec=5):
        raise TimeoutError('Cup sweep requires /compute_fk')
    state = deepcopy(start)
    initial = dict(zip(state.name, state.position))
    previous = np.array([initial[name] for name in trajectory.joint_names])
    minimum = math.inf
    samples = 0
    try:
        for point in trajectory.points:
            target = np.asarray(point.positions)
            count = max(1, math.ceil(float(np.max(np.abs(target-previous)))/.001))
            for alpha in np.linspace(0, 1, count+1):
                mapping = dict(initial)
                mapping.update(zip(trajectory.joint_names, previous+(target-previous)*alpha))
                state.position = [float(mapping[name]) for name in state.name]
                state.velocity = []
                state.effort = []
                request = GetPositionFK.Request()
                request.header.frame_id = session.frame_id
                request.fk_link_names = ['link7']
                request.robot_state.joint_state = state
                response = session._wait(client.call_async(request), '杯体扫掠FK')
                if response.error_code.val != 1 or len(response.pose_stamped) != 1:
                    raise RuntimeError('Cup sweep FK failed')
                pose = response.pose_stamped[0].pose
                quaternion = [
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w]
                position = np.array([pose.position.x, pose.position.y, pose.position.z])
                # 55mm grasp height and 47.5mm cup half-height at the fixed side grasp RPY.
                center_local = np.asarray(tcp_offset, float) + np.array([0, .0075, 0])
                center = position + rotate_vector(center_local, quaternion)
                axis = rotate_vector([0, -1., 0], quaternion)
                minimum = min(minimum, check_cup_floor(center, axis))
                samples += 1
            previous = target
    finally:
        session.node.destroy_client(client)
    return dict(fk_samples=samples, minimum_cup_bottom_mm=minimum*1000)
