"""视觉坐标、检测和标定共用几何函数。"""

import math
import os
from pathlib import Path

import cv2
import numpy as np


SYSTEM_QT_FONT_DIRS = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation2",
    "/usr/share/fonts/opentype/noto",
)


def use_system_qt_fonts(candidates=SYSTEM_QT_FONT_DIRS):
    """Replace OpenCV wheel's missing font path with installed system fonts."""
    font_dir = os.environ.get("QT_QPA_FONTDIR")
    if not font_dir or not Path(font_dir).is_dir():
        os.environ.pop("QT_QPA_FONTDIR", None)
        for candidate in map(Path, candidates):
            if candidate.is_dir() and any(candidate.glob("*.ttf")):
                os.environ["QT_QPA_FONTDIR"] = str(candidate)
                break


use_system_qt_fonts()


class ArucoMarkerDetector:
    """依次尝试一个或多个 ArUco 字典，失败时回退到特征匹配。"""

    def __init__(self, dictionary_id, marker_id):
        self.marker_id = marker_id
        parameters = cv2.aruco.DetectorParameters()
        parameters.errorCorrectionRate = 1.0
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.features = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        dictionaries = (
            dictionary_id.items()
            if isinstance(dictionary_id, dict)
            else [(str(dictionary_id), dictionary_id)]
        )
        self.detectors = []
        for name, predefined_id in dictionaries:
            dictionary = cv2.aruco.getPredefinedDictionary(predefined_id)
            template = cv2.aruco.generateImageMarker(dictionary, marker_id, 700)
            template_points, template_descriptors = self.features.detectAndCompute(
                template, None
            )
            self.detectors.append(
                (
                    name,
                    cv2.aruco.ArucoDetector(dictionary, parameters),
                    template_points,
                    template_descriptors,
                )
            )
        self.last_dictionary_name = None

    def detect(self, gray):
        """返回规范方向的四角和检测方式；未找到时返回 ``(None, None)``。"""
        gray = np.asarray(gray)
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("gray 必须是 uint8 灰度图")

        self.last_dictionary_name = None
        for name, aruco, _template_points, _template_descriptors in self.detectors:
            corners, ids, _rejected = aruco.detectMarkers(gray)
            if ids is not None:
                matches = np.flatnonzero(ids.reshape(-1) == self.marker_id)
                if len(matches):
                    self.last_dictionary_name = name
                    return corners[int(matches[0])].reshape(4, 2), "aruco"

        scene_points, scene_descriptors = self.features.detectAndCompute(gray, None)
        if scene_descriptors is None:
            return None, None
        for name, _aruco, template_points, template_descriptors in self.detectors:
            pairs = self.matcher.knnMatch(template_descriptors, scene_descriptors, k=2)
            good = [
                first
                for pair in pairs
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < 0.75 * second.distance
            ]
            if len(good) < 8:
                continue

            source = np.float32([template_points[item.queryIdx].pt for item in good])
            target = np.float32([scene_points[item.trainIdx].pt for item in good])
            homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
            if homography is None or inliers is None or int(inliers.sum()) < 8:
                continue
            marker_corners = cv2.perspectiveTransform(
                np.float32([[[0, 0], [699, 0], [699, 699], [0, 699]]]),
                homography,
            )[0]
            height, width = gray.shape
            if (
                not np.isfinite(marker_corners).all()
                or not cv2.isContourConvex(marker_corners.astype(np.float32))
                or cv2.contourArea(marker_corners.astype(np.float32)) < 1000
                or np.any(marker_corners[:, 0] < 0)
                or np.any(marker_corners[:, 0] >= width)
                or np.any(marker_corners[:, 1] < 0)
                or np.any(marker_corners[:, 1] >= height)
            ):
                continue
            self.last_dictionary_name = name
            return marker_corners, "feature"
        return None, None


def rotate_image_180(image):
    """将倒装相机的图像旋转 180°，恢复为正向画面。"""
    return cv2.rotate(image, cv2.ROTATE_180)


def rotated_to_raw_pixel(x, y, width, height):
    """把旋转后画面中的像素坐标映射回 ZED 原始图像。"""
    return width - 1 - x, height - 1 - y


def transform_points(points, transform):
    """使用 4×4 齐次矩阵转换一组三维点。"""
    points = np.asarray(points, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是 N×3 数组")
    if transform.shape != (4, 4):
        raise ValueError("transform 必须是 4×4 矩阵")
    if not np.isfinite(points).all() or not np.isfinite(transform).all():
        raise ValueError("坐标和变换矩阵必须是有限数值")
    return points @ transform[:3, :3].T + transform[:3, 3]


def median_xyz_in_box(xyz_image, box, center_fraction=0.4, min_points=12):
    """返回检测框中心区域的稳健三维坐标，自动忽略无效深度。"""
    xyz = np.asarray(xyz_image, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] < 3:
        raise ValueError("xyz_image 必须是 H×W×3（或更多通道）数组")
    if not 0 < center_fraction <= 1:
        raise ValueError("center_fraction 必须在 (0, 1] 范围内")

    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("检测框坐标无效")
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_width = (x2 - x1) * center_fraction / 2
    half_height = (y2 - y1) * center_fraction / 2
    height, width = xyz.shape[:2]
    left = max(0, int(math.floor(center_x - half_width)))
    right = min(width, int(math.ceil(center_x + half_width)))
    top = max(0, int(math.floor(center_y - half_height)))
    bottom = min(height, int(math.ceil(center_y + half_height)))
    points = xyz[top:bottom, left:right, :3].reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    if len(points) < min_points:
        return None

    # 只按深度剔除桌面/边缘混合点，保留物体表面的横向尺寸。
    depth_median = np.median(points[:, 2])
    depth_mad = np.median(np.abs(points[:, 2] - depth_median))
    if depth_mad > 0:
        filtered = points[
            np.abs(points[:, 2] - depth_median) <= 3 * 1.4826 * depth_mad
        ]
        if len(filtered) >= min_points:
            points = filtered
    return np.median(points, axis=0)


def stable_point(points, max_spread_mm):
    """若连续三维点都位于中位点门限内，返回中位点，否则返回 ``None``。"""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return None
    if not np.isfinite(points).all() or max_spread_mm <= 0:
        return None
    center = np.median(points, axis=0)
    if np.linalg.norm(points - center, axis=1).max() > max_spread_mm:
        return None
    return center


def cylinder_center_from_surface(surface_base, camera_point, base_from_camera, radius_mm):
    """沿相机视线的桌面投影，把圆柱可见表面点修正到中心轴。"""
    surface = np.asarray(surface_base, dtype=float)
    camera_point = np.asarray(camera_point, dtype=float)
    transform = validate_rigid_transform(base_from_camera)
    if surface.shape != (3,) or camera_point.shape != (3,):
        raise ValueError("表面点和相机点必须是 3 元坐标")
    if not np.isfinite([*surface, *camera_point, radius_mm]).all() or radius_mm <= 0:
        raise ValueError("圆柱坐标和半径必须是有效正数")
    ray = transform[:3, :3] @ (camera_point / np.linalg.norm(camera_point))
    ray[2] = 0.0
    length = np.linalg.norm(ray)
    if length < 1e-6:
        raise ValueError("相机视线在桌面上的投影过小")
    return surface + radius_mm * ray / length


def validate_rigid_transform(transform, atol=1e-3):
    """验证 4×4 矩阵只包含旋转和平移，不接受缩放或镜像。"""
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("外参必须是包含有限数值的 4×4 矩阵")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=atol):
        raise ValueError("外参最后一行必须是 [0, 0, 0, 1]")

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("外参旋转部分不是正交矩阵")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("外参旋转部分包含缩放或镜像")
    return transform


def validate_calibration_quality(
    calibration, max_translation_rmse_mm=10.0, max_rotation_rmse_deg=2.0
):
    """拒绝拟合一致性超过项目安全门限的相机外参。"""
    try:
        translation = float(calibration["fit"]["translation_rmse_mm"])
        rotation = float(calibration["fit"]["rotation_rmse_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("外参缺少手眼标定质量指标") from exc
    if not np.isfinite([translation, rotation]).all():
        raise ValueError("外参质量指标不是有限数值")
    if translation > max_translation_rmse_mm or rotation > max_rotation_rmse_deg:
        raise ValueError(
            f"外参一致性不合格：{translation:.3f} mm, {rotation:.3f} deg"
        )
    return calibration


def pose_to_transform(pose):
    """将 NERO 的 ``[x, y, z, roll, pitch, yaw]`` 转为齐次矩阵。"""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("pose 必须是包含有限数值的 6 元数组")

    x, y, z, roll, pitch, yaw = pose
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4)
    # NERO 使用固定坐标系 XYZ 外旋：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。
    transform[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    transform[:3, 3] = [x, y, z]
    return transform


def flange_pose_for_tcp(tcp_xyz_mm, tcp_offset_mm, flange_rpy_deg):
    """由目标 TCP 点、法兰坐标系 TCP 偏移和法兰姿态计算 SDK 法兰位姿。"""
    rpy = np.radians(flange_rpy_deg)
    rotation = pose_to_transform([0, 0, 0, *rpy])[:3, :3]
    flange_xyz_mm = np.asarray(tcp_xyz_mm) - rotation @ np.asarray(tcp_offset_mm)
    return [*(flange_xyz_mm / 1000.0), *rpy]


def invert_transform(transform):
    """对刚体变换求逆，保留旋转正交性。"""
    transform = validate_rigid_transform(transform)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _project_rotation(rotation):
    """把数值求解得到的矩阵投影到最近的旋转矩阵。"""
    u, _singular_values, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def _rotation_error_degrees(first, second):
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def estimate_eye_to_hand(base_from_flange, camera_from_marker):
    """由多组法兰/标记位姿求固定外置相机外参，长度单位须一致。"""
    if len(base_from_flange) != len(camera_from_marker):
        raise ValueError("法兰位姿和标记位姿数量必须相同")
    if len(base_from_flange) < 4:
        raise ValueError("手眼标定至少需要 4 组位姿，实机建议采集 10 组以上")

    flange_poses = [validate_rigid_transform(item) for item in base_from_flange]
    marker_poses = [validate_rigid_transform(item) for item in camera_from_marker]
    flange_motions = []
    marker_motions = []
    for i in range(len(flange_poses)):
        for j in range(i + 1, len(flange_poses)):
            flange_motions.append(invert_transform(flange_poses[j]) @ flange_poses[i])
            marker_motions.append(invert_transform(marker_poses[j]) @ marker_poses[i])

    # A * X = X * B，其中 X 是 flange <- marker。
    rotation_system = np.vstack(
        [
            np.kron(np.eye(3), a[:3, :3])
            - np.kron(b[:3, :3].T, np.eye(3))
            for a, b in zip(flange_motions, marker_motions)
        ]
    )
    if np.linalg.matrix_rank(rotation_system, tol=1e-8) < 8:
        raise ValueError("采样姿态旋转变化不足，请明显改变 Roll、Pitch、Yaw")
    _u, _singular_values, vt = np.linalg.svd(rotation_system)
    raw = vt[-1].reshape(3, 3, order="F")
    candidates = [_project_rotation(raw), _project_rotation(-raw)]
    rotation = min(
        candidates,
        key=lambda candidate: sum(
            np.linalg.norm(a[:3, :3] @ candidate - candidate @ b[:3, :3]) ** 2
            for a, b in zip(flange_motions, marker_motions)
        ),
    )

    translation_system = np.vstack(
        [motion[:3, :3] - np.eye(3) for motion in flange_motions]
    )
    if np.linalg.matrix_rank(translation_system, tol=1e-8) < 3:
        raise ValueError("采样姿态无法确定平移，请增加不同旋转轴的位姿")
    translation_targets = np.hstack(
        [
            rotation @ marker[:3, 3] - flange[:3, 3]
            for flange, marker in zip(flange_motions, marker_motions)
        ]
    )
    translation = np.linalg.lstsq(
        translation_system, translation_targets, rcond=None
    )[0]
    flange_from_marker = np.eye(4)
    flange_from_marker[:3, :3] = rotation
    flange_from_marker[:3, 3] = translation

    estimates = [
        flange @ flange_from_marker @ invert_transform(marker)
        for flange, marker in zip(flange_poses, marker_poses)
    ]
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = _project_rotation(
        sum(item[:3, :3] for item in estimates)
    )
    base_from_camera[:3, 3] = np.mean(
        [item[:3, 3] for item in estimates], axis=0
    )
    translation_errors = np.array(
        [
            np.linalg.norm(item[:3, 3] - base_from_camera[:3, 3])
            for item in estimates
        ]
    )
    rotation_errors = np.array(
        [
            _rotation_error_degrees(base_from_camera[:3, :3], item[:3, :3])
            for item in estimates
        ]
    )
    return base_from_camera, flange_from_marker, translation_errors, rotation_errors


def estimate_eye_to_hand_robust(
    base_from_flange,
    camera_from_marker,
    min_inliers=10,
    max_outlier_fraction=0.25,
    mad_scale=2.5,
):
    """迭代剔除明显离群位姿，并返回求解结果及原始样本下标。"""
    sample_count = len(base_from_flange)
    if sample_count != len(camera_from_marker):
        raise ValueError("法兰位姿和标记位姿数量必须相同")
    if sample_count < min_inliers:
        raise ValueError(f"稳健手眼标定至少需要 {min_inliers} 组样本")

    flange = np.asarray(base_from_flange, dtype=float)
    marker = np.asarray(camera_from_marker, dtype=float)
    inliers = np.arange(sample_count)
    max_outliers = int(sample_count * max_outlier_fraction)
    for _ in range(5):
        result = estimate_eye_to_hand(flange[inliers], marker[inliers])
        translation_errors, rotation_errors = result[2:]
        translation_median = np.median(translation_errors)
        rotation_median = np.median(rotation_errors)
        translation_mad = np.median(
            np.abs(translation_errors - translation_median)
        )
        rotation_mad = np.median(np.abs(rotation_errors - rotation_median))
        translation_limit = max(
            0.005,
            translation_median + mad_scale * 1.4826 * translation_mad,
        )
        rotation_limit = max(
            1.0,
            rotation_median + mad_scale * 1.4826 * rotation_mad,
        )
        rejected = (translation_errors > translation_limit) | (
            rotation_errors > rotation_limit
        )
        if not rejected.any():
            return (*result, inliers)
        remaining = inliers[~rejected]
        if (
            len(remaining) < min_inliers
            or sample_count - len(remaining) > max_outliers
        ):
            raise ValueError("离群样本过多，请检查标定板、工作区覆盖和机械臂反馈")
        inliers = remaining

    result = estimate_eye_to_hand(flange[inliers], marker[inliers])
    return (*result, inliers)


def estimate_rigid_transform(source_points, target_points):
    """拟合 source 到 target 的无缩放刚体变换，并返回逐点误差。"""
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points 和 target_points 必须是同形状的 N×3 数组")
    if len(source) < 3:
        raise ValueError("至少需要 3 组对应点")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("对应点必须是有限数值")

    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    if np.linalg.matrix_rank(source_centered) < 2:
        raise ValueError("相机坐标点不能全部重合或共线")
    if np.linalg.matrix_rank(target_centered) < 2:
        raise ValueError("机械臂基坐标点不能全部重合或共线")

    covariance = source_centered.T @ target_centered
    u, _singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T

    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    validate_rigid_transform(transform)
    errors = np.linalg.norm(transform_points(source, transform) - target, axis=1)
    return transform, errors


def estimate_tcp_pivot(flange_poses):
    """由多组法兰位姿接触同一固定点，求法兰坐标系中的 TCP 偏移。"""
    if len(flange_poses) < 4:
        raise ValueError("TCP 枢轴标定至少需要 4 组位姿，建议采集 8 组以上")
    transforms = [pose_to_transform(pose) for pose in flange_poses]
    system = np.vstack(
        [np.hstack((transform[:3, :3], -np.eye(3))) for transform in transforms]
    )
    targets = np.hstack([-transform[:3, 3] for transform in transforms])
    if np.linalg.matrix_rank(system) < 6:
        raise ValueError("采样姿态变化不足，请明显改变 Roll、Pitch、Yaw")
    solution = np.linalg.lstsq(system, targets, rcond=None)[0]
    tcp_offset, fixed_point = solution[:3], solution[3:]
    errors = np.array(
        [
            np.linalg.norm(
                transform[:3, :3] @ tcp_offset
                + transform[:3, 3]
                - fixed_point
            )
            for transform in transforms
        ]
    )
    return tcp_offset, fixed_point, errors
