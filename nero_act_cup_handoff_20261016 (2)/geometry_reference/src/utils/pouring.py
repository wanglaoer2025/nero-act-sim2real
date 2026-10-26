"""透明直立器皿定位与倾倒轨迹的纯几何工具。"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from src.utils.vision import (
    pose_to_transform,
    rotated_to_raw_pixel,
    transform_points,
    validate_rigid_transform,
)


@dataclass(frozen=True)
class MaskGeometry:
    center_pixel: tuple
    base_pixel: tuple
    rim_pixel: tuple
    width_px: int
    height_px: int
    axis: tuple
    tilt_deg: float
    area_px: int


@dataclass(frozen=True)
class VerticalObjectPose:
    base_camera_mm: np.ndarray
    base_robot_mm: np.ndarray
    top_robot_mm: np.ndarray
    grasp_robot_mm: np.ndarray
    vertical_robot: np.ndarray


def extract_mask_geometry(mask):
    """Extract the largest mask component and its vertical geometry."""
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    contours, _hierarchy = cv2.findContours(
        (binary > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("mask 为空")
    contour = max(contours, key=cv2.contourArea)
    area = int(round(cv2.contourArea(contour)))
    if area < 20:
        raise ValueError("mask 面积过小")

    component = np.zeros_like(binary)
    cv2.drawContours(component, [contour], -1, 1, thickness=cv2.FILLED)
    ys, xs = np.nonzero(component)
    x, y, width, height = cv2.boundingRect(contour)
    points = np.column_stack((xs, ys)).astype(float)
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    _values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    if axis[1] < 0:
        axis = -axis
    tilt = math.degrees(math.acos(np.clip(axis[1], -1.0, 1.0)))

    top_y = int(ys.min())
    bottom_y = int(ys.max())
    rim_x = float(np.median(xs[ys <= top_y + max(1, height // 20)]))
    base_x = float(np.median(xs[ys >= bottom_y - max(1, height // 20)]))
    return MaskGeometry(
        center_pixel=(float(center[0]), float(center[1])),
        base_pixel=(base_x, float(bottom_y)),
        rim_pixel=(rim_x, float(top_y)),
        width_px=width,
        height_px=height,
        axis=(float(axis[0]), float(axis[1])),
        tilt_deg=float(tilt),
        area_px=area,
    )


def fit_table_plane(points, threshold_mm=5.0, iterations=200, seed=0):
    """Fit ``ax+by+cz+d=0`` with deterministic RANSAC."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是 N×3 数组")
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    if len(points) < 30 or threshold_mm <= 0 or iterations < 1:
        raise ValueError("桌面有效点不足或 RANSAC 参数无效")
    if len(points) > 6000:
        indices = np.linspace(0, len(points) - 1, 6000, dtype=int)
        points = points[indices]

    rng = np.random.default_rng(seed)
    best = None
    best_count = 0
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) <= threshold_mm
        count = int(inliers.sum())
        if count > best_count:
            best = inliers
            best_count = count
    if best is None or best_count < max(30, int(0.25 * len(points))):
        raise ValueError("没有找到稳定桌面平面")

    inlier_points = points[best]
    center = inlier_points.mean(axis=0)
    _u, _values, vt = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ center)
    residuals = np.abs(inlier_points @ normal + offset)
    return np.array([*normal, offset]), residuals


def pixel_ray(pixel, camera_matrix):
    """Return a normalized camera ray for one raw image pixel."""
    matrix = np.asarray(camera_matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera_matrix 必须是有限的 3×3 矩阵")
    u, v = map(float, pixel)
    ray = np.linalg.solve(matrix, [u, v, 1.0])
    length = np.linalg.norm(ray)
    if length < 1e-12:
        raise ValueError("像素射线无效")
    return ray / length


def camera_pixel_to_base_z(pixel, camera_matrix, base_from_camera, z_base_mm):
    """Intersect one camera pixel ray with a horizontal plane in base coordinates."""
    transform = validate_rigid_transform(base_from_camera)
    ray_camera = pixel_ray(pixel, camera_matrix)
    origin_base = transform[:3, 3]
    ray_base = transform[:3, :3] @ ray_camera
    if abs(ray_base[2]) < 1e-9:
        raise ValueError("像素射线与 base 水平面平行")
    distance = (float(z_base_mm) - origin_base[2]) / ray_base[2]
    if distance <= 0 or not math.isfinite(distance):
        raise ValueError("base 水平面交点不在相机前方")
    return origin_base + distance * ray_base


def recover_vertical_object_from_axis_pixels(
    base_pixel,
    rim_pixel,
    image_shape,
    camera_matrix,
    base_from_camera,
    table_z_base_mm,
    height_mm,
    grasp_height_ratio=0.6,
    position_offset_mm=(0.0, 0.0, 0.0),
):
    """Recover an upright object's base/grasp points without using depth."""
    if height_mm <= 0 or not 0 < grasp_height_ratio < 1:
        raise ValueError("物体高度或抓取比例无效")
    height, width = image_shape[:2]
    base = np.asarray(base_pixel, dtype=float)
    rim = np.asarray(rim_pixel, dtype=float)
    offset = np.asarray(position_offset_mm, dtype=float)
    if base.shape != (2,) or rim.shape != (2,) or offset.shape != (3,):
        raise ValueError("轴线像素或位置补偿格式无效")
    if not np.isfinite([*base, *rim, *offset, table_z_base_mm]).all():
        raise ValueError("无深度定位输入必须是有限数值")

    grasp_pixel = base + grasp_height_ratio * (rim - base)
    raw_pixel = rotated_to_raw_pixel(*grasp_pixel, width, height)
    grasp_z = float(table_z_base_mm) + height_mm * grasp_height_ratio
    grasp_robot = camera_pixel_to_base_z(
        raw_pixel, camera_matrix, base_from_camera, grasp_z
    ) + offset
    base_robot = grasp_robot.copy()
    base_robot[2] = float(table_z_base_mm) + offset[2]
    top_robot = base_robot + [0.0, 0.0, height_mm]
    camera_from_base = np.linalg.inv(validate_rigid_transform(base_from_camera))
    base_camera = transform_points([base_robot], camera_from_base)[0]
    return VerticalObjectPose(
        base_camera_mm=base_camera,
        base_robot_mm=base_robot,
        top_robot_mm=top_robot,
        grasp_robot_mm=grasp_robot,
        vertical_robot=np.array([0.0, 0.0, 1.0]),
    )


def intersect_ray_with_plane(ray, plane):
    """Intersect a camera-origin ray with a plane in the same coordinates."""
    ray = np.asarray(ray, dtype=float)
    plane = np.asarray(plane, dtype=float)
    if ray.shape != (3,) or plane.shape != (4,):
        raise ValueError("ray 必须为 3 元，plane 必须为 4 元")
    denominator = float(plane[:3] @ ray)
    if abs(denominator) < 1e-9:
        raise ValueError("像素射线与桌面平面平行")
    distance = -float(plane[3]) / denominator
    if distance <= 0 or not math.isfinite(distance):
        raise ValueError("桌面交点不在相机前方")
    return distance * ray


def recover_vertical_object(
    base_camera_mm,
    table_plane_camera,
    base_from_camera,
    height_mm,
    grasp_height_ratio=0.6,
    max_table_tilt_deg=20.0,
):
    """Recover top and grasp points from a known upright object height."""
    base_camera = np.asarray(base_camera_mm, dtype=float)
    plane = np.asarray(table_plane_camera, dtype=float)
    transform = validate_rigid_transform(base_from_camera)
    if base_camera.shape != (3,) or plane.shape != (4,):
        raise ValueError("底部点或桌面平面格式无效")
    if height_mm <= 0 or not 0 < grasp_height_ratio < 1:
        raise ValueError("物体高度或抓取比例无效")

    vertical = transform[:3, :3] @ (plane[:3] / np.linalg.norm(plane[:3]))
    if vertical[2] < 0:
        vertical = -vertical
    vertical /= np.linalg.norm(vertical)
    tilt = math.degrees(math.acos(np.clip(vertical[2], -1.0, 1.0)))
    if tilt > max_table_tilt_deg:
        raise ValueError(f"桌面法向与 base +Z 偏差过大：{tilt:.1f}°")

    base_robot = transform_points([base_camera], transform)[0]
    top_robot = base_robot + height_mm * vertical
    grasp_robot = base_robot + height_mm * grasp_height_ratio * vertical
    return VerticalObjectPose(
        base_camera_mm=base_camera,
        base_robot_mm=base_robot,
        top_robot_mm=top_robot,
        grasp_robot_mm=grasp_robot,
        vertical_robot=vertical,
    )


def rotation_matrix_to_rpy(rotation):
    """Convert a rotation matrix to NERO fixed-frame XYZ roll/pitch/yaw."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation 必须是 3×3 矩阵")
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return np.array([roll, pitch, yaw])


def generate_pour_waypoints(
    initial_pose,
    pivot,
    target_direction,
    max_angle_deg=90.0,
    step_angle_deg=5.0,
    tilt_translation=None,
):
    """Generate pour/return poses with optional translation during tilt."""
    pose = np.asarray(initial_pose, dtype=float)
    pivot = np.asarray(pivot, dtype=float)
    direction = np.array(target_direction, dtype=float, copy=True)
    translation = np.asarray(
        [0.0, 0.0, 0.0] if tilt_translation is None else tilt_translation,
        dtype=float,
    )
    if pose.shape != (6,) or pivot.shape != (3,) or direction.shape != (3,):
        raise ValueError("位姿、pivot 或倾倒方向格式无效")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("倾倒旋转补偿必须是有限的 base XYZ 三元向量")
    direction[2] = 0.0
    if np.linalg.norm(direction) < 1e-9:
        raise ValueError("源器皿与目标器皿的水平位置重合")
    if not 0 < max_angle_deg <= 120 or not 0 < step_angle_deg <= max_angle_deg:
        raise ValueError("倾倒角度必须满足 0 < step <= max <= 120")
    direction /= np.linalg.norm(direction)
    axis = np.cross([0.0, 0.0, 1.0], direction)
    axis /= np.linalg.norm(axis)

    angles = list(np.arange(0.0, max_angle_deg, step_angle_deg))
    if not angles or not math.isclose(angles[-1], max_angle_deg):
        angles.append(float(max_angle_deg))
    initial_rotation = pose_to_transform(pose)[:3, :3]
    relative = pose[:3] - pivot
    forward = []
    for angle in angles:
        rotation, _jacobian = cv2.Rodrigues(axis * math.radians(angle))
        position = (
            pivot + rotation @ relative
            + translation * (angle / max_angle_deg)
        )
        orientation = rotation @ initial_rotation
        forward.append(np.concatenate((position, rotation_matrix_to_rpy(orientation))))
    return forward, list(reversed(forward[:-1]))


def build_dry_run_plan(source_base_mm, target_base_mm, vertical_robot, config):
    """Build a config-driven grasp/lift/pour plan without controlling hardware."""
    source = np.asarray(source_base_mm, dtype=float) / 1000.0
    target = np.asarray(target_base_mm, dtype=float) / 1000.0
    vertical = np.array(vertical_robot, dtype=float, copy=True)
    if source.shape != (3,) or target.shape != (3,) or vertical.shape != (3,):
        raise ValueError("源点、目标点和竖直方向必须是三元坐标")
    if not np.isfinite([*source, *target, *vertical]).all():
        raise ValueError("规划输入必须是有限数值")
    vertical_norm = np.linalg.norm(vertical)
    if vertical_norm < 1e-9:
        raise ValueError("桌面法向量无效")
    vertical /= vertical_norm
    if vertical[2] < 0:
        vertical = -vertical

    source_cfg = config["source"]
    target_cfg = config["target"]
    grasp_cfg = config["grasp"]
    pour_cfg = config["pour"]
    source_height = float(source_cfg["height_mm"]) / 1000.0
    source_radius = float(source_cfg["diameter_mm"]) / 2000.0
    target_height = float(target_cfg["height_mm"]) / 1000.0
    grasp_ratio = float(source_cfg["grasp_height_ratio"])
    if source_height <= 0 or source_radius <= 0 or target_height <= 0:
        raise ValueError("器皿尺寸必须大于零")
    if not 0 < grasp_ratio < 1:
        raise ValueError("抓取高度比例必须在 (0, 1) 内")

    direction = np.asarray(
        pour_cfg.get("direction_base", target - source), dtype=float
    ).copy()
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("倾倒方向必须是有限的三元 base 向量")
    direction -= vertical * float(direction @ vertical)
    if np.linalg.norm(direction) < 1e-9:
        raise ValueError("倾倒方向不能与桌面法向平行")
    direction /= np.linalg.norm(direction)
    approach = np.asarray(
        grasp_cfg.get("approach_direction_base", [1.0, 0.0, 0.0]), dtype=float
    )
    approach -= vertical * float(approach @ vertical)
    if np.linalg.norm(approach) < 1e-9:
        raise ValueError("接近方向不能与桌面法向平行")
    approach /= np.linalg.norm(approach)

    grasp_height, pregrasp_clearance = _grasp_path_heights(
        source_height, grasp_ratio, grasp_cfg
    )
    grasp_xyz = source + grasp_height * vertical
    pregrasp_xyz = (
        grasp_xyz
        + float(grasp_cfg["pregrasp_distance_mm"]) / 1000.0 * approach
    )
    pregrasp_high_xyz = pregrasp_xyz + pregrasp_clearance * vertical
    lift_xyz = grasp_xyz + float(grasp_cfg["lift_height_mm"]) / 1000.0 * vertical
    target_top = target + target_height * vertical
    lip_standoff = float(pour_cfg.get("lip_standoff_mm", 0.0)) / 1000.0
    if lip_standoff < 0:
        raise ValueError("杯口水平退让距离不能小于零")
    pivot = (
        target_top
        - lip_standoff * direction
        + float(pour_cfg["vertical_offset_mm"]) / 1000.0 * vertical
    )
    # The beaker stays diagonally above -direction; its near lip pours toward center.
    rim_start = pivot - source_radius * direction
    pour_start_xyz = rim_start - (source_height - grasp_height) * vertical
    vertical_raise = max(
        0.0, float((lift_xyz - pour_start_xyz) @ vertical)
    )
    pivot += vertical_raise * vertical
    pour_start_xyz += vertical_raise * vertical
    pour_position_offset = np.asarray(
        pour_cfg.get("pour_position_offset_base_mm", [0.0, 0.0, 0.0]),
        dtype=float,
    ) / 1000.0
    if (
        pour_position_offset.shape != (3,)
        or not np.isfinite(pour_position_offset).all()
    ):
        raise ValueError("倾倒准备位偏移必须是有限的 base XYZ 三元向量")
    pivot += pour_position_offset
    pour_start_xyz += pour_position_offset
    carry_xyz = pour_start_xyz.copy()
    pre_pour_descent = float(
        pour_cfg.get("pre_pour_descent_mm", 0.0)
    ) / 1000.0
    if not math.isfinite(pre_pour_descent) or not 0 <= pre_pour_descent <= 0.1:
        raise ValueError("倾倒前下降距离必须在 0..100 mm 范围内")
    pivot -= pre_pour_descent * vertical
    pour_start_xyz -= pre_pour_descent * vertical
    upright_bottom_clearance = float(
        (pour_start_xyz - grasp_height * vertical - target_top) @ vertical
    )
    if upright_bottom_clearance < 0.01:
        raise ValueError(
            "倾倒前下降过低：竖直源杯底部距目标杯口必须至少保留 10 mm"
        )
    rpy = np.radians(np.asarray(grasp_cfg["flange_rpy_deg"], dtype=float))
    carry_pose = np.concatenate((carry_xyz, rpy))
    pour_start_pose = np.concatenate((pour_start_xyz, rpy))
    tilt_translation = np.asarray(
        pour_cfg.get("tilt_translation_base_mm", [0.0, 0.0, 0.0]),
        dtype=float,
    ) / 1000.0
    if tilt_translation.shape != (3,) or not np.isfinite(tilt_translation).all():
        raise ValueError("倾倒旋转补偿必须是有限的 base XYZ 三元向量")
    final_lip_clearance = float(
        (pivot + tilt_translation - target_top) @ vertical
    )
    if final_lip_clearance < 0.02:
        raise ValueError("倾倒终点杯口距目标杯口必须至少保留 20 mm")
    forward, return_path = generate_pour_waypoints(
        pour_start_pose, pivot, direction,
        max_angle_deg=float(pour_cfg["max_angle_deg"]),
        step_angle_deg=float(pour_cfg["angle_step_deg"]),
        tilt_translation=tilt_translation,
    )
    return {
        "pregrasp_high_pose": np.concatenate((pregrasp_high_xyz, rpy)),
        "pregrasp_pose": np.concatenate((pregrasp_xyz, rpy)),
        "grasp_pose": np.concatenate((grasp_xyz, rpy)),
        "lift_pose": np.concatenate((lift_xyz, rpy)),
        "carry_pose": carry_pose,
        "pour_start_pose": pour_start_pose,
        "pivot": pivot,
        "pour_direction": direction,
        "upright_bottom_clearance_mm": upright_bottom_clearance * 1000.0,
        "final_lip_clearance_mm": final_lip_clearance * 1000.0,
        "pour_waypoints": forward,
        "return_waypoints": return_path,
}


def _grasp_path_heights(source_height, grasp_ratio, grasp_cfg):
    """Return safe TCP grasp height and vertical pregrasp clearance in metres."""
    minimum = float(grasp_cfg.get("min_tcp_height_above_table_mm", 55.0)) / 1000.0
    clearance = float(grasp_cfg.get(
        "pregrasp_vertical_clearance_mm", grasp_cfg["lift_height_mm"]
    )) / 1000.0
    if not 0 < minimum < source_height:
        raise ValueError("最低 TCP 离桌高度必须大于 0 且小于烧杯高度")
    if clearance <= 0:
        raise ValueError("预抓取垂直净空必须大于 0")
    return max(grasp_ratio * source_height, minimum), clearance


def build_source_grasp_plan(source_base_mm, vertical_robot, config):
    """Build only the source pregrasp, grasp and lift poses."""
    source = np.asarray(source_base_mm, dtype=float) / 1000.0
    vertical = np.asarray(vertical_robot, dtype=float).copy()
    if source.shape != (3,) or vertical.shape != (3,):
        raise ValueError("源点和竖直方向必须是三元坐标")
    if not np.isfinite([*source, *vertical]).all():
        raise ValueError("规划输入必须是有限数值")
    vertical_norm = np.linalg.norm(vertical)
    if vertical_norm < 1e-9:
        raise ValueError("桌面法向量无效")
    vertical /= vertical_norm
    if vertical[2] < 0:
        vertical = -vertical

    source_cfg = config["source"]
    grasp_cfg = config["grasp"]
    source_height = float(source_cfg["height_mm"]) / 1000.0
    grasp_ratio = float(source_cfg["grasp_height_ratio"])
    if source_height <= 0 or not 0 < grasp_ratio < 1:
        raise ValueError("源器皿尺寸和抓取高度比例无效")
    approach = np.asarray(
        grasp_cfg.get("approach_direction_base", [1.0, 0.0, 0.0]), dtype=float
    )
    approach -= vertical * float(approach @ vertical)
    if np.linalg.norm(approach) < 1e-9:
        raise ValueError("接近方向不能与桌面法向平行")
    approach /= np.linalg.norm(approach)

    grasp_height, pregrasp_clearance = _grasp_path_heights(
        source_height, grasp_ratio, grasp_cfg
    )
    grasp_xyz = source + grasp_height * vertical
    pregrasp_xyz = (
        grasp_xyz
        + float(grasp_cfg["pregrasp_distance_mm"]) / 1000.0 * approach
    )
    rpy = np.radians(np.asarray(grasp_cfg["flange_rpy_deg"], dtype=float))
    return {
        "pregrasp_high_pose": np.concatenate((
            pregrasp_xyz + pregrasp_clearance * vertical,
            rpy,
        )),
        "pregrasp_pose": np.concatenate((
            pregrasp_xyz,
            rpy,
        )),
        "grasp_pose": np.concatenate((grasp_xyz, rpy)),
        "lift_pose": np.concatenate((
            grasp_xyz
            + float(grasp_cfg["lift_height_mm"]) / 1000.0 * vertical,
            rpy,
        )),
    }
