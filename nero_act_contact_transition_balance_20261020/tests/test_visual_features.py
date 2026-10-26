import copy
import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('executor.visual_observation') is not None, 'Unified visual features and uncertainty checks missing'
    return importlib.import_module('executor.visual_observation')


def test_relative_positions_are_base_metric_and_current_tcp():
    m=api()
    geometry=SimpleNamespace(valid=True,cup_bottom_base_m=[.1,.2,0],target_base_m=[.3,.4,0])
    robot=SimpleNamespace(pose=lambda q:dict(tcp=np.array([q[0],0,.1])))
    value=m.relative_positions(np.r_[.05,np.zeros(6),.095],geometry,robot)
    np.testing.assert_allclose(value,[.05,.2,-.1,.25,.4,-.1],atol=1e-8)
    geometry.valid=False
    with pytest.raises(ValueError):m.relative_positions(np.r_[np.zeros(7),.095],geometry,robot)


def test_geometry_stats_use_train_only():
    m=api();features=np.array([[0.]*6,[.2]*6,[100.]*6])
    stats=m.geometry_statistics(features,['train','train','val'],{'train'})
    np.testing.assert_allclose(stats['geometry_mean'],[.1]*6)
    np.testing.assert_allclose(stats['geometry_std'],[.1]*6)
    with pytest.raises(ValueError):m.geometry_statistics(features,['a','b','c'],{'missing'})


def valid_inputs():
    from vision.synchronization import CameraFrame
    from vision.observation import GeometryObservation
    from tests.test_calibration import samples
    from calibration.solve import solve_samples
    from calibration.bundle import calibration_hash
    m=api();raw=samples();raw['synthetic']=False
    calibration=solve_samples(raw);calibration['calibration_sha256']=calibration_hash(calibration)
    frame=CameraFrame(1,np.zeros((360,640,3),np.uint8),10.,10.01,m.camera_metadata_hash(calibration['camera_metadata']),True)
    observation=GeometryObservation(1,[.1,.2,0],[.3,.4,0],0.,.99,.001,calibration['calibration_sha256'],True)
    paired=dict(state=[0.]*7+[.095],monotonic_s=10.004,skew_s=.004,frame_id=1)
    return frame,observation,paired,calibration


def test_stale_invalid_or_wrong_calibration_is_rejected():
    m=api();frame,geometry,paired,cal=valid_inputs()
    m.require_visual_observation(frame,geometry,paired,now_s=10.02,calibration=cal)
    for change in [dict(valid=False),dict(calibration_sha256='wrong'),dict(frame_id=2),dict(error_bound_m=float('nan'))]:
        bad=copy.copy(geometry)
        for k,v in change.items():setattr(bad,k,v)
        with pytest.raises(ValueError):m.require_visual_observation(frame,bad,paired,now_s=10.02,calibration=cal)
    for now in [9.9,10.3]:
        with pytest.raises(ValueError):m.require_visual_observation(frame,geometry,paired,now_s=now,calibration=cal)
    frame.camera_metadata_hash='old-K'
    with pytest.raises(ValueError):m.require_visual_observation(frame,geometry,paired,now_s=10.02,calibration=cal)


def test_clearance_adds_errors_and_stop_distance():
    m=api();bounds=dict(tcp_m=.001,vision_m=.002,shape_m=.001,tracking_m=.001)
    assert m.required_clearance_m(bounds,speed_m_s=.01,detection_s=.1,stopping_s=.1)==pytest.approx(.009)
    for change in [dict(vision_m=-.001),dict(tcp_m=float('nan')),dict(shape_m=None)]:
        with pytest.raises(ValueError):m.required_clearance_m(dict(bounds,**change),speed_m_s=.01,detection_s=.1,stopping_s=.1)
    with pytest.raises(ValueError):m.required_clearance_m(bounds,speed_m_s=.01,detection_s=.1,stopping_s=None)
