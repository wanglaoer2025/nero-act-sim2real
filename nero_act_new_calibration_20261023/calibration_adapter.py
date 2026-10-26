"""Import this dated hand-eye candidate for offline ACT validation, with explicit frames."""
import copy
import importlib.util
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parents[1]/'nero_act_visual_feedback_20261019/calibration/bundle.py'
_spec = importlib.util.spec_from_file_location('_retained_calibration_bundle', _path)
_bundle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bundle)


def candidate_to_metres(candidate):
    if (candidate.get('method')!='eye_to_hand_aruco_ax_xb'
            or candidate.get('source_coordinate_system')!='zed_camera_image'
            or candidate.get('target_coordinate_system')!='nero_base'
            or candidate.get('translation_unit')!='millimeter'):
        raise ValueError('Explicit raw ZED-camera to NERO-base millimetre candidate required')
    transform = _bundle.rigid_transform(candidate['transform_base_from_camera']).copy()
    transform[:3,3] *= .001
    return _bundle.rigid_transform(transform)


def verify_handeye_fit(candidate, samples):
    transform = candidate_to_metres(candidate)
    if samples.get('translation_unit')!='meter':
        raise ValueError('Raw pose pairs must explicitly use metres')
    rows = samples['samples']
    excluded = candidate['fit']['excluded_sample_indices']
    if (len(set(excluded))!=len(excluded) or
            any(type(i) is not int or not 1<=i<=len(rows) for i in excluded)):
        raise ValueError('This source uses unique one-based excluded sample indices')
    marker = _bundle.rigid_transform(candidate['flange_from_marker_m'])
    translations, rotations = [], []
    for index, row in enumerate(rows, 1):
        if index in excluded:
            continue
        camera_marker = _bundle.rigid_transform(row['transform_camera_from_marker_m'])
        base_flange = _bundle.rigid_transform(row['transform_base_from_flange_m'])
        # Match the source tool's camera-pose consistency residual, not marker-position error.
        estimate = base_flange@marker@np.linalg.inv(camera_marker)
        translations.append(float(np.linalg.norm(estimate[:3,3]-transform[:3,3])*1000))
        cosine = (np.trace(transform[:3,:3].T@estimate[:3,:3])-1)/2
        rotations.append(float(np.degrees(np.arccos(np.clip(cosine,-1,1)))))
    if len(translations)<15 or len(translations)!=candidate['fit']['sample_count']:
        raise ValueError('At least15 matching inlier pairs required')
    t_rmse = float(np.sqrt(np.mean(np.square(translations))))
    r_rmse = float(np.sqrt(np.mean(np.square(rotations))))
    if (not np.isclose(t_rmse,candidate['fit']['translation_rmse_mm'],atol=1e-5)
            or not np.isclose(r_rmse,candidate['fit']['rotation_rmse_deg'],atol=1e-5)
            or t_rmse>10 or r_rmse>2):
        raise ValueError('Raw-pair fit differs from the recorded candidate or original fit limits')
    return dict(inlier_count=len(translations), total_samples=len(rows), translation_rmse_mm=t_rmse,
        rotation_rmse_deg=r_rmse, camera_serial_number=samples['camera_serial_number'],
        fit_reproduced=True, independent_task_space_validation_passed=False)


def apply_to_scene(scene, candidate):
    result = copy.deepcopy(scene)
    result['camera_calibration']['transform_base_from_raw_optical_m'] = candidate_to_metres(candidate).tolist()
    result['camera_calibration']['installation_verified_on_real'] = False
    result['hardware_deployment_allowed'] = False
    result['calibration_experiment_scope'] = 'fixed_world_camera_transform_sensitivity'
    return result
