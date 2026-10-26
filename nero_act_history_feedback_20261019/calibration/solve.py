"""Solve recorded TCP/camera samples offline: python -m calibration.solve --help."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.bundle import SCHEMA, rigid_transform, save_calibration, validate_geometry


def _sample_ids(rows):
    ids = [row['sample_id'] for row in rows]
    if not all(isinstance(x, str) and x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique nonempty sample IDs required')
    return ids


def solve_tcp(samples):
    if len(samples) < 6:
        raise ValueError('At least six independent TCP orientations required')
    _sample_ids(samples)
    poses = [rigid_transform(row['base_from_flange_4x4']) for row in samples]
    a = np.concatenate([np.c_[t[:3,:3], -np.eye(3)] for t in poses])
    b = np.concatenate([-t[:3,3] for t in poses])
    solution, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank != 6 or singular[0]/singular[-1] > 1e6:
        raise ValueError('TCP orientation coverage is rank deficient or ill conditioned')
    residual = np.linalg.norm((a@solution-b).reshape(-1, 3), axis=1)
    return dict(tcp_offset_flange_xyz_m=solution[:3].tolist(), fixed_point_base_xyz_m=solution[3:].tolist(),
                rank=int(rank), residual_m=residual.tolist(), sample_ids=_sample_ids(samples))


def solve_camera(base_points_m, pixels_rectified, K):
    points, pixels, k = [np.asarray(x, np.float64) for x in (base_points_m, pixels_rectified, K)]
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 6
            or pixels.shape != (len(points), 2) or k.shape != (3,3)
            or not all(np.isfinite(x).all() for x in [points, pixels, k])
            or np.linalg.matrix_rank(points-points.mean(0)) != 3):
        raise ValueError('At least six finite noncoplanar control points (rank3) required')
    if (k[0,0] <= 0 or k[1,1] <= 0 or not np.allclose(k[2], [0,0,1])
            or not np.isclose(k[0,1], 0) or not np.isclose(k[1,0], 0)):
        raise ValueError('Rectified pinhole intrinsics required')
    ok, rvec, tvec = cv2.solvePnP(points, pixels, k, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise ValueError('Camera PnP failed')
    rvec, tvec = cv2.solvePnPRefineLM(points, pixels, k, None, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    optical = points@rotation.T+tvec.ravel()
    if np.any(optical[:,2] <= 0):
        raise ValueError('Camera control points must have positive optical depth')
    camera_from_base = np.eye(4);camera_from_base[:3,:3] = rotation;camera_from_base[:3,3] = tvec.ravel()
    transform = rigid_transform(np.linalg.inv(camera_from_base))
    projected = cv2.projectPoints(points, rvec, tvec, k, None)[0].reshape(-1,2)
    return dict(transform_base_from_camera=transform.tolist(), K=k.tolist(),
                reprojection_rmse_px=float(np.sqrt(np.mean((projected-pixels)**2))))


def validate_calibration(solution, holdout):
    tcp_rows, camera_rows = holdout['tcp'], holdout['camera']
    if len(tcp_rows) < 3 or len(camera_rows) < 9:
        raise ValueError('At least3 independent TCP and9 camera holdout points required')
    tcp_ids, camera_ids = _sample_ids(tcp_rows), _sample_ids(camera_rows)
    if (set(tcp_ids) & set(solution['tcp']['sample_ids'])
            or set(camera_ids) & set(solution['camera']['sample_ids'])):
        raise ValueError('Fit and holdout sample IDs must be disjoint; overlap found')
    uncertainty = float(holdout['measurement_uncertainty_m'])
    if not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError('Finite independent measurement uncertainty required')
    tcp = np.r_[solution['tcp']['tcp_offset_flange_xyz_m'], 1]
    tcp_errors = []
    for row in tcp_rows:
        reference = np.asarray(row['reference_base_xyz_m'], float)
        if reference.shape != (3,) or not np.isfinite(reference).all():
            raise ValueError('Independent metric reference point required')
        measured = (rigid_transform(row['base_from_flange_4x4'])@tcp)[:3]
        tcp_errors.append(float(np.linalg.norm(measured-reference)))
    transform = rigid_transform(solution['camera']['transform_base_from_camera'])
    camera_errors = []
    for row in camera_rows:
        base, camera = [np.asarray(row[k], float) for k in ['base_xyz_m', 'camera_xyz_m']]
        if base.shape != (3,) or camera.shape != (3,) or not np.isfinite(np.r_[base,camera]).all() or camera[2] <= 0:
            raise ValueError('Independent base/camera3D correspondence required')
        camera_errors.append(float(np.linalg.norm((transform@np.r_[camera,1])[:3]-base)))
    return dict(tcp_max_error_m=max(tcp_errors), camera_max_error_m=max(camera_errors),
                tcp_rmse_m=float(np.sqrt(np.mean(np.square(tcp_errors)))),
                camera_rmse_m=float(np.sqrt(np.mean(np.square(camera_errors)))),
                measurement_uncertainty_m=uncertainty,
                bounds_m=dict(tcp_m=max(tcp_errors)+uncertainty, vision_m=max(camera_errors)+uncertainty),
                tcp_sample_ids=tcp_ids, camera_sample_ids=camera_ids,
                accuracy_target_met=max(tcp_errors)+uncertainty <= .001 and max(camera_errors)+uncertainty <= .002)


def solve_samples(samples):
    raw = copy.deepcopy(samples)
    if raw.get('translation_unit') != 'meter':
        raise ValueError('Samples must explicitly use meter units')
    metadata = raw['camera_metadata']
    if raw.get('pixel_space') != 'rectified_left' or metadata.get('pixel_space') != 'rectified_left':
        raise ValueError('Samples must match rectified left optical pixels')
    size = np.asarray(metadata.get('image_size_wh'), float)
    if (not metadata.get('serial') or size.shape != (2,) or not np.isfinite(size).all()
            or np.any(size <= 0) or metadata.get('image_rotation_degrees') != 180):
        raise ValueError('Camera identity, image size and explicit180-degree display mapping required')
    camera = solve_camera(**{k: raw['camera'][k] for k in ['base_points_m', 'pixels_rectified', 'K']})
    if 'K' in metadata and not np.array_equal(np.asarray(metadata['K'],float),np.asarray(camera['K'])):
        raise ValueError('Camera metadata intrinsics differ from the rectified control-point camera')
    metadata['K'] = camera['K']
    camera['sample_ids'] = list(raw['camera']['sample_ids'])
    if len(camera['sample_ids']) != len(raw['camera']['base_points_m']):
        raise ValueError('One ID per camera control point required')
    _sample_ids([dict(sample_id=x) for x in camera['sample_ids']])
    solution = dict(tcp=solve_tcp(raw['tcp']), camera=camera)
    validation = validate_calibration(solution, raw['holdout'])
    geometry = validate_geometry(raw['geometry'])
    synthetic = raw.get('synthetic') is not False
    return dict(schema=SCHEMA, translation_unit='meter', pixel_space='rectified_left',
                synthetic=synthetic, verified=not synthetic,
                accuracy_target_met=validation['accuracy_target_met'],
                solution=solution, validation=validation, geometry=geometry,
                camera_metadata=metadata, samples=raw, motion_sent=False,
                hardware_deployment_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = solve_samples(json.loads(args.samples.read_text()))
    save_calibration(args.output, report)
    print(json.dumps(dict(output=str(args.output), synthetic=report['synthetic'],
                          verified=report['verified'], accuracy_target_met=report['accuracy_target_met'], motion_sent=False)))


if __name__ == '__main__':
    main()
