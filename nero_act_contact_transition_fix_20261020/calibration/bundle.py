"""Versioned calibration evidence; loading never connects to hardware."""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'nero_visual_calibration_v1'


def camera_metadata_hash(metadata):
    return hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def rigid_transform(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(t[:3,:3].T@t[:3,:3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(t[:3,:3]), 1, atol=1e-7)):
        raise ValueError('Finite proper rigid transform required')
    return t


def calibration_hash(document):
    content = {k: v for k, v in document.items() if k != 'calibration_sha256'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_geometry(geometry):
    try:
        table = float(geometry['table_top_z_base_m'])
        rotation = np.asarray(geometry['tcp_rotation_flange'], float)
        fingers, cup = geometry['fingers'], geometry['cup']
        dimensions = [cup[k] for k in ['height_m', 'body_diameter_m', 'rim_diameter_m']]
        bounds = [geometry['uncertainty'][k] for k in ['shape_m', 'tracking_m']]
        yaw = np.asarray(cup['handle_yaw_range_rad'], float)
        boxes = fingers['boxes'] + cup['handle_boxes']
        sizes = np.asarray([b['size_m'] for b in boxes], float)
        centers = np.asarray([b.get('center_tcp_m', b.get('center_bottom_m')) for b in boxes], float)
        valid = (np.isfinite([table, *dimensions, *bounds, fingers['maximum_width_m']]).all()
                 and all(x > 0 for x in dimensions) and all(x >= 0 for x in bounds)
                 and 0 < fingers['maximum_width_m'] <= .1
                 and cup['rim_diameter_m'] >= cup['body_diameter_m']
                 and yaw.shape == (2,) and np.isfinite(yaw).all() and yaw[0] < yaw[1]
                 and len(fingers['boxes']) >= 2 and len(cup['handle_boxes']) >= 1
                 and sizes.shape == (len(boxes), 3) and centers.shape == sizes.shape
                 and np.isfinite(sizes).all() and np.isfinite(centers).all() and np.all(sizes > 0)
                 and rotation.shape == (3, 3)
                 and np.allclose(rotation.T@rotation, np.eye(3), atol=1e-7)
                 and np.isclose(np.linalg.det(rotation), 1, atol=1e-7)
                 and fingers.get('measured') is True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Complete measured table/finger/cup geometry required') from exc
    if not valid:
        raise ValueError('Invalid measured table/finger/cup geometry')
    return geometry


def save_calibration(path, document):
    document = dict(document)
    document['calibration_sha256'] = calibration_hash(document)
    with Path(path).open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_calibration(path, *, require_verified):
    document = json.loads(Path(path).read_text())
    if document.get('schema') != SCHEMA or document.get('translation_unit') != 'meter':
        raise ValueError('Explicit metric calibration v1 required; nominal/approximate TCP is not calibration')
    if document.get('calibration_sha256') != calibration_hash(document):
        raise ValueError('Calibration hash changed')
    validate_geometry(document['geometry'])
    # Recompute from the bound raw samples instead of trusting edited verification flags.
    from calibration.solve import solve_samples
    actual = solve_samples(document['samples'])
    if calibration_hash(actual) != calibration_hash(document):
        raise ValueError('Calibration solution or verification changed from its raw samples')
    if require_verified and (document.get('verified') is not True or document.get('synthetic') is not False):
        raise ValueError('Independent real calibration is required; synthetic or approximate evidence cannot deploy')
    return document
