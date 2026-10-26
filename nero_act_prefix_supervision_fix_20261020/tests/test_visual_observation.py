import importlib
import importlib.util

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('vision.observation') is not None, 'Calibrated read-only geometry observer missing'
    return importlib.import_module('vision.observation')


def calibration():
    return dict(calibration_sha256='test',verified=False,synthetic=True,pixel_space='rectified_left',
        camera_metadata=dict(serial='test',image_size_wh=[640,360],pixel_space='rectified_left',image_rotation_degrees=180),
        solution=dict(camera=dict(K=[[500,0,320],[0,500,180],[0,0,1]],transform_base_from_camera=np.eye(4).tolist())),
        validation=dict(bounds_m=dict(vision_m=.002)),
        geometry=dict(table_top_z_base_m=1.,cup=dict(height_m=.1,handle_yaw_range_rad=[-.2,.2])))


def keypoints():
    # Camera looking along+Z in this synthetic fixture: bottom[.1,.02,1], rim[.1,.02,1.1].
    return dict(cup_count=1,target_count=1,occluded=False,quality=.99,handle_yaw_rad=0.,
                bottom_center_px=[370,190],rim_center_px=[365.45454545454544,189.0909090909091],target_center_px=[420,180],
                pixel_space='rectified_left',qualified=True)


def test_unrotation_and_rectified_intrinsics_are_paired():
    m=api()
    assert m.upright_to_optical((0,0),(640,360))==(639,359)
    row=m.triangulate_upright_cup(keypoints(),calibration())
    np.testing.assert_allclose(row['cup_bottom_base_m'],[.1,.02,1.],atol=1e-8)
    np.testing.assert_allclose(row['target_base_m'],[.2,0,1.],atol=1e-8)
    wrong=keypoints();wrong['pixel_space']='raw_left'
    with pytest.raises(ValueError):m.triangulate_upright_cup(wrong,calibration())


def test_cup_handle_and_occlusion_are_not_valid_centres():
    m=api()
    for change in [dict(occluded=True),dict(cup_count=2),dict(target_count=2),dict(handle_yaw_rad=1.),dict(qualified=False),dict(quality=.2)]:
        with pytest.raises(ValueError):m.triangulate_upright_cup(dict(keypoints(),**change),calibration())
    with pytest.raises(ValueError):m.triangulate_upright_cup(dict(keypoints(),rim_center_px=[370,190]),calibration())


def test_observer_records_invalid_frame_instead_of_using_old_geometry():
    m=api()
    from vision.synchronization import CameraFrame
    from executor.visual_observation import camera_metadata_hash
    frame=CameraFrame(1,np.zeros((360,640,3),np.uint8),10.,10.01,camera_metadata_hash(calibration()['camera_metadata']),True)
    observer=m.VisualObserver(calibration(),detector=lambda _:keypoints())
    observation=observer.measure(frame)
    assert observation.valid and observation.frame_id==1
    observer.detector=lambda _:dict(keypoints(),occluded=True)
    invalid=observer.measure(frame)
    assert invalid.valid is False and invalid.cup_bottom_base_m is None
    # Legacy bounding boxes do not certify body/rim/handle semantics.
    assert m.VisualObserver(calibration()).measure(frame).valid is False


def test_observer_refuses_wrong_camera_or_intrinsics_hash():
    m=api()
    from vision.synchronization import CameraFrame
    frame=CameraFrame(1,np.zeros((360,640,3),np.uint8),10.,10.01,'wrong-camera-and-K',True)
    observation=m.VisualObserver(calibration(),detector=lambda _:keypoints()).measure(frame)
    assert observation.valid is False and observation.cup_bottom_base_m is None
