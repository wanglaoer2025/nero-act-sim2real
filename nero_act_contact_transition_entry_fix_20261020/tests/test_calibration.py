import copy
import importlib
import importlib.util
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def module(name):
    assert importlib.util.find_spec(name) is not None, f'{name}: approved offline calibration interface missing'
    return importlib.import_module(name)


def samples():
    tcp = np.array([.151, .002, -.025])
    pivot = np.array([-.45, .01, .2])
    rotations = Rotation.from_euler('xyz', [[0,0,0],[30,0,0],[0,40,0],[0,0,50],[-25,35,10],[20,-30,-40],[5,15,25],[-10,20,40],[30,15,-20]], degrees=True).as_matrix()
    poses = []
    for i, r in enumerate(rotations):
        t = np.eye(4);t[:3,:3] = r;t[:3,3] = pivot-r@tcp
        poses.append(dict(sample_id=f'tcp{i}', base_from_flange_4x4=t.tolist(), reference_base_xyz_m=pivot.tolist()))
    rng = np.random.default_rng(123)
    base = rng.uniform([-.6,-.2,.0],[-.3,.2,.15],size=(21,3))
    rotation = Rotation.from_euler('xyz',[.15,-.1,.03]).as_matrix()
    camera_from_base = np.eye(4);camera_from_base[:3,:3] = rotation;camera_from_base[:3,3]=[.4,0,1.]
    camera = base@rotation.T+camera_from_base[:3,3]
    k = np.array([[500.,0,320],[0,500.,180],[0,0,1.]])
    pixels = camera[:,:2]/camera[:,2,None]*500 + [320,180]
    return dict(synthetic=True, translation_unit='meter', pixel_space='rectified_left',
        tcp=poses[:6], camera=dict(base_points_m=base[:12].tolist(),pixels_rectified=pixels[:12].tolist(),K=k.tolist(),sample_ids=[f'fit{i}' for i in range(12)]),
        holdout=dict(tcp=poses[6:],camera=[dict(sample_id=f'hold{i}',base_xyz_m=b.tolist(),camera_xyz_m=c.tolist()) for i,(b,c) in enumerate(zip(base[12:],camera[12:]))],measurement_uncertainty_m=.0001),
        camera_metadata=dict(serial='test-only',image_size_wh=[640,360],pixel_space='rectified_left',image_rotation_degrees=180),
        geometry=dict(table_top_z_base_m=0.,tcp_rotation_flange=np.eye(3).tolist(),
          fingers=dict(measured=True,maximum_width_m=.095,boxes=[dict(name='left',size_m=[.01,.01,.06],center_tcp_m=[0,.05,0]),dict(name='right',size_m=[.01,.01,.06],center_tcp_m=[0,-.05,0])]),
          cup=dict(height_m=.095,body_diameter_m=.075,rim_diameter_m=.08,handle_boxes=[dict(size_m=[.02,.03,.04],center_bottom_m=[.04,0,.05])],handle_yaw_range_rad=[-.2,.2]),
          uncertainty=dict(shape_m=.001,tracking_m=.001)))


def solved():
    m = module('calibration.solve');raw = samples()
    return m.solve_samples(raw), raw


def test_tcp_recovery_and_rank_deficiency():
    m = module('calibration.solve');rows = samples()['tcp']
    result = m.solve_tcp(rows)
    np.testing.assert_allclose(result['tcp_offset_flange_xyz_m'],[.151,.002,-.025],atol=1e-8)
    assert result['rank'] == 6
    with pytest.raises(ValueError,match='rank|orientation'):
        m.solve_tcp([dict(rows[0],sample_id=str(i)) for i in range(6)])
    invalid=copy.deepcopy(rows);invalid[0]['base_from_flange_4x4'][0][0]=2
    with pytest.raises(ValueError):m.solve_tcp(invalid)


def test_camera_holdout_is_disjoint_and_metric():
    m=module('calibration.solve');result,raw=solved()
    assert result['validation']['camera_max_error_m'] < 1e-7
    assert result['validation']['tcp_max_error_m'] < 1e-7
    assert result['validation']['accuracy_target_met'] is True
    shared=copy.deepcopy(raw['holdout']);shared['tcp'][0]['sample_id']='tcp0'
    with pytest.raises(ValueError,match='disjoint|overlap'):m.validate_calibration(result['solution'],shared)
    wrong=copy.deepcopy(raw);wrong['translation_unit']='millimeter'
    with pytest.raises(ValueError,match='meter'):m.solve_samples(wrong)
    wrong=copy.deepcopy(raw);wrong['pixel_space']='raw_left'
    with pytest.raises(ValueError,match='rectified'):m.solve_samples(wrong)


def test_camera_rejects_coplanar_or_nonfinite_control_points():
    m=module('calibration.solve');cam=samples()['camera'];points=np.array(cam['base_points_m']);points[:,2]=0
    with pytest.raises(ValueError,match='noncoplanar|rank'):m.solve_camera(points,cam['pixels_rectified'],cam['K'])
    points[0,0]=np.nan
    with pytest.raises(ValueError):m.solve_camera(points,cam['pixels_rectified'],cam['K'])


def test_approximate_bundle_cannot_authorize(tmp_path):
    b=module('calibration.bundle');result,_=solved()
    assert result['synthetic'] is True and result['verified'] is False
    path=tmp_path/'bundle.json';b.save_calibration(path,result)
    assert b.load_calibration(path,require_verified=False)['synthetic']
    with pytest.raises(ValueError):b.load_calibration(path,require_verified=True)
    for approximation in [dict(approximate=True,tcp_offset_flange_xyz_m=[.135,0,0]),dict(verified=True,tcp_link7_xyz_m=[.152,0,-.0235])]:
        path.write_text(json.dumps(approximation))
        with pytest.raises(ValueError):b.load_calibration(path,require_verified=True)


def test_holdout_detects_real_error_and_bundle_tampering(tmp_path):
    m=module('calibration.solve');b=module('calibration.bundle');result,raw=solved()
    hold=copy.deepcopy(raw['holdout']);hold['camera'][0]['base_xyz_m'][0]+=.01
    report=m.validate_calibration(result['solution'],hold)
    assert report['camera_max_error_m'] > .009 and report['accuracy_target_met'] is False
    path=tmp_path/'bundle.json';b.save_calibration(path,result)
    changed=json.loads(path.read_text());changed['geometry']['cup']['height_m']=.2;path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='hash|changed'):b.load_calibration(path,require_verified=False)


def test_calibration_cli_is_offline_and_refuses_overwrite(tmp_path):
    module('calibration.solve')
    source=tmp_path/'samples.json';source.write_text(json.dumps(samples()));output=tmp_path/'calibration.json'
    command=[sys.executable,'-m','calibration.solve','--samples',str(source),'--output',str(output)]
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode == 0,run.stderr
    report=json.loads(output.read_text());assert report['verified'] is False and report['motion_sent'] is False
    again=subprocess.run(command,capture_output=True,text=True)
    assert again.returncode != 0


def test_camera_metadata_binds_current_intrinsics():
    m=module('calibration.solve');raw=samples()
    result=m.solve_samples(raw)
    assert 'K' in result['camera_metadata'], 'Camera identity hash must include actual rectified intrinsics'
    np.testing.assert_array_equal(result['camera_metadata']['K'],raw['camera']['K'])
    wrong=copy.deepcopy(raw);wrong['camera_metadata']['K']=[[600,0,320],[0,600,180],[0,0,1]]
    with pytest.raises(ValueError,match='intrinsic|camera'):m.solve_samples(wrong)
