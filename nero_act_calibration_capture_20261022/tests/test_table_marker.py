import importlib

import cv2
import numpy as np
import pytest


def points(rx=-2.):
    K=np.array([[527.4,0,647.2],[0,527.4,359.8],[0,0,1.]])
    h=.095/2;obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
    px=cv2.projectPoints(obj,np.array([rx,0.,0.]),np.array([.1,0.,1.]),K,None)[0].reshape(4,2)
    return px,K


def test_known_board_plane_metric_distance_and_physical_thickness():
    px,K=points();m=importlib.import_module('table_marker_capture')
    value=m.marker_plane(px,K,thickness_mm=2.5)
    assert value['camera_to_board_plane_m']==pytest.approx(.4161468365471424,abs=1e-7)
    assert value['camera_to_table_plane_m']==pytest.approx(.4186468365471424,abs=1e-7)
    assert value['calibration_verified'] is False


def test_unknown_thickness_does_not_invent_table_plane():
    px,K=points();v=importlib.import_module('table_marker_capture').marker_plane(px,K)
    assert v['camera_to_table_plane_m'] is None and v['board_thickness_mm'] is None


def test_ambiguous_frontal_board_cannot_become_verified_plane():
    px,K=points(np.pi-.03)
    with pytest.raises(ValueError,match='ambiguous'):
        importlib.import_module('table_marker_capture').marker_plane(px,K)


def test_unstable_repeated_frames_are_rejected_and_never_independent_poses():
    m=importlib.import_module('table_marker_capture');p,K=points()
    rows=[m.marker_plane(p,K) for _ in range(8)]
    report=m.combine_stationary(rows)
    assert report['distinct_physical_positions']==1 and report['calibration_verified'] is False
    rows[-1]=dict(rows[-1],camera_to_board_plane_m=.425)
    with pytest.raises(ValueError,match='spread'):m.combine_stationary(rows)
