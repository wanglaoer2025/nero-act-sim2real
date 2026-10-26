"""Unchanged localization functions used by the methylene-blue demo."""

import numpy as np

from src.utils.pouring import recover_vertical_object_from_axis_pixels, camera_pixel_to_base_z, build_source_grasp_plan
from src.utils.vision import invert_transform, transform_points, flange_pose_for_tcp

def box_axis_pixels(box, image_shape):
    """Return the visible top and bottom centers of a clipped detection box."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = map(float, box)
    left = float(np.clip(x1, 0, width - 1))
    right = float(np.clip(x2, 0, width - 1))
    top = float(np.clip(y1, 0, height - 1))
    bottom = float(np.clip(y2, 0, height - 1))
    if right <= left or bottom <= top:
        raise ValueError("检测框裁剪后无效")
    center_x = (left + right) / 2.0
    return (center_x, bottom), (center_x, top)


def project_cylinder_box(
    center_xy,
    image_shape,
    camera_matrix,
    base_from_camera,
    table_z_base_mm,
    beaker_height_mm,
    beaker_diameter_mm,
    samples=96,
):
    """Project a vertical base-frame cylinder and return its rotated-image box."""
    center = np.asarray(center_xy, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("圆柱中心必须是有限的二维坐标")
    if beaker_height_mm <= 0 or beaker_diameter_mm <= 0 or samples < 12:
        raise ValueError("圆柱尺寸或采样数无效")

    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    radius = beaker_diameter_mm / 2.0
    ring_xy = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    base_ring = np.column_stack(
        (ring_xy, np.full(samples, float(table_z_base_mm)))
    )
    top_ring = base_ring.copy()
    top_ring[:, 2] += beaker_height_mm
    camera_points = transform_points(
        np.vstack((base_ring, top_ring)), invert_transform(base_from_camera)
    )
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("圆柱投影位于相机后方")

    normalized = camera_points[:, :2] / camera_points[:, 2, None]
    raw_u = camera_matrix[0, 0] * normalized[:, 0] + camera_matrix[0, 2]
    raw_v = camera_matrix[1, 1] * normalized[:, 1] + camera_matrix[1, 2]
    height, width = image_shape[:2]
    rotated_u = width - 1 - raw_u
    rotated_v = height - 1 - raw_v
    return np.array(
        [
            rotated_u.min(),
            rotated_v.min(),
            rotated_u.max(),
            rotated_v.max(),
        ],
        dtype=float,
    )


def fit_cylinder_center(
    observed_box,
    initial_xy,
    image_shape,
    camera_matrix,
    base_from_camera,
    table_z_base_mm,
    beaker_height_mm,
    beaker_diameter_mm,
):
    """Fit cylinder center XY to all four observed box edges with Gauss-Newton."""
    observed = np.asarray(observed_box, dtype=float)
    center = np.asarray(initial_xy, dtype=float).copy()
    if observed.shape != (4,) or center.shape != (2,):
        raise ValueError("检测框或圆柱初值格式无效")

    def projected(value):
        return project_cylinder_box(
            value,
            image_shape,
            camera_matrix,
            base_from_camera,
            table_z_base_mm,
            beaker_height_mm,
            beaker_diameter_mm,
        )

    for _iteration in range(12):
        prediction = projected(center)
        residual = prediction - observed
        epsilon = 0.5
        jacobian = np.column_stack(
            [
                (projected(center + offset) - projected(center - offset))
                / (2.0 * epsilon)
                for offset in (
                    np.array([epsilon, 0.0]),
                    np.array([0.0, epsilon]),
                )
            ]
        )
        step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        length = float(np.linalg.norm(step))
        if length > 50.0:
            step *= 50.0 / length
        center += step
        if np.linalg.norm(step) < 0.01:
            break

    residual = projected(center) - observed
    return center, float(np.sqrt(np.mean(residual**2)))


def localize_boxes(
    boxes,
    image_shape,
    camera_matrix,
    base_from_camera,
    table_z_base_mm,
    beaker_height_mm,
    beaker_diameter_mm,
):
    """Fit left-to-right beaker boxes as vertical cylinders in the base frame."""
    # ponytail: left-to-right IDs assume static beakers; add tracking if they move.
    ordered = sorted(boxes, key=lambda box: float(box[0] + box[2]))
    points = []
    for box in ordered:
        base_pixel, rim_pixel = box_axis_pixels(box, image_shape)
        pose = recover_vertical_object_from_axis_pixels(
            base_pixel,
            rim_pixel,
            image_shape,
            camera_matrix,
            base_from_camera,
            table_z_base_mm,
            beaker_height_mm,
            grasp_height_ratio=0.4,
        )
        center_xy, fit_rmse_px = fit_cylinder_center(
            box,
            pose.base_robot_mm[:2],
            image_shape,
            camera_matrix,
            base_from_camera,
            table_z_base_mm,
            beaker_height_mm,
            beaker_diameter_mm,
        )
        bottom_center = np.array([*center_xy, float(table_z_base_mm)])
        fitted_box = project_cylinder_box(
            center_xy,
            image_shape,
            camera_matrix,
            base_from_camera,
            table_z_base_mm,
            beaker_height_mm,
            beaker_diameter_mm,
        )
        points.append(
            (np.asarray(box, dtype=float), fitted_box, bottom_center, fit_rmse_px)
        )
    return points
