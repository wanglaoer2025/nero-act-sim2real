"""Isolated read-only tabletop selector; retained calibration remains unqualified."""
import numpy as np


def select_blue(rgb, config, strict_selector):
    calibration = config['camera_calibration']
    width, height = calibration['resolution_wh']
    if rgb.shape != (height, width, 3) or calibration['image_rotation_degrees'] != 180:
        raise ValueError('Selector requires the retained upright180 camera profile')
    matrix = np.asarray(calibration['intrinsic_matrix'], float)
    transform = np.asarray(calibration['transform_base_from_raw_optical_m'], float)
    center = np.asarray(config['table_center_xy_m'], float)
    half = np.asarray(config['table_half_size_xy_m'], float)
    if (matrix.shape != (3, 3) or transform.shape != (4, 4)
            or center.shape != (2,) or half.shape != (2,)
            or not np.isfinite(np.r_[matrix.ravel(), transform.ravel(), center, half,
                                     config['table_top_z_m']]).all() or np.any(half <= 0)):
        raise ValueError('Invalid retained table/camera geometry')
    rows, columns = np.indices((height, width))
    pixels = np.stack([width-1-columns, height-1-rows, np.ones_like(rows)], axis=-1)
    directions = pixels @ np.linalg.inv(matrix).T @ transform[:3, :3].T
    scale = np.zeros((height, width), float)
    np.divide(config['table_top_z_m']-transform[2, 3], directions[:, :, 2],
              out=scale, where=np.abs(directions[:, :, 2]) > 1e-12)
    points = transform[:3, 3]+scale[:, :, None]*directions
    mask = ((scale > 0) & np.isfinite(points).all(axis=2)
            & (np.abs(points[:, :, :2]-center) < half).all(axis=2))
    filtered = np.where(mask[:, :, None], rgb, 0).astype(rgb.dtype)
    box = strict_selector(filtered)
    x1, y1 = np.maximum(0, np.floor(box[:2]).astype(int)-2)
    x2, y2 = np.minimum([width, height], np.ceil(box[2:]).astype(int)+3)
    if not mask[y1:y2, x1:x2].all():
        raise ValueError('Blue target touches projected table boundary')
    return box
