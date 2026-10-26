"""Calibrated visual geometry from qualified semantic keypoints; no control APIs.

The inherited bounding-box detector cannot certify rim/body/handle keypoints.
Until a detector is independently validated, a missing frontend produces invalid
observations instead of reusing a nominal cylinder or a previous cup position.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.bundle import rigid_transform,load_calibration,camera_metadata_hash
from vision.synchronization import CameraFrame


@dataclass
class GeometryObservation:
    frame_id: int
    cup_bottom_base_m: object = None
    target_base_m: object = None
    cup_handle_yaw_rad: object = None
    quality: float = 0.
    error_bound_m: float = float('inf')
    calibration_sha256: str = ''
    valid: bool = False
    invalid_reason: str = ''


def upright_to_optical(pixel_xy, image_size_wh):
    p, size = np.asarray(pixel_xy,float),np.asarray(image_size_wh,float)
    if (p.shape != (2,) or size.shape != (2,) or not np.isfinite(np.r_[p,size]).all()
            or np.any(size <= 0) or np.any(p < 0) or np.any(p >= size)):
        raise ValueError('Valid upright pixel within calibrated image required')
    return tuple(size-1-p)


def triangulate_upright_cup(keypoints, calibration):
    if (keypoints.get('pixel_space') != 'rectified_left' or keypoints.get('qualified') is not True
            or keypoints.get('occluded') is not False
            or keypoints.get('cup_count') != 1 or keypoints.get('target_count') != 1
            or not np.isfinite(keypoints.get('quality',float('nan'))) or keypoints['quality'] < .9):
        raise ValueError('Unique qualified body/rim/handle and target keypoints required; occlusion is invalid')
    yaw = float(keypoints['handle_yaw_rad'])
    low, high = calibration['geometry']['cup']['handle_yaw_range_rad']
    if not np.isfinite(yaw) or not low <= yaw <= high:
        raise ValueError('Cup handle orientation outside measured range')
    camera = calibration['solution']['camera']
    transform = rigid_transform(camera['transform_base_from_camera'])
    k = np.asarray(camera['K'],float)
    size = np.asarray(calibration['camera_metadata']['image_size_wh'],float)
    def ray(name):
        pixel = np.asarray(keypoints[name],float)
        if pixel.shape != (2,) or not np.isfinite(pixel).all() or np.any(pixel < 0) or np.any(pixel >= size):
            raise ValueError('Semantic keypoint outside calibrated image')
        return transform[:3,:3]@np.linalg.solve(k,np.r_[pixel,1])
    bottom_ray, rim_ray = ray('bottom_center_px'),ray('rim_center_px')
    height = float(calibration['geometry']['cup']['height_m'])
    a = np.c_[bottom_ray,-rim_ray]
    distances,_,rank,singular = np.linalg.lstsq(a,[0,0,-height],rcond=None)
    if rank != 2 or singular[0]/singular[-1] > 1e5 or np.any(distances <= 0):
        raise ValueError('Cup height triangulation has degenerate rays or nonpositive depth')
    residual = float(np.linalg.norm(a@distances-[0,0,-height]))
    bound = float(calibration['validation']['bounds_m']['vision_m'])
    if not np.isfinite(bound) or not 0 <= residual <= bound:
        raise ValueError('Cup upright geometry disagrees with independent visual error bound')
    center = transform[:3,3]+distances[0]*bottom_ray
    target_ray = ray('target_center_px')
    if abs(target_ray[2]) < 1e-8:
        raise ValueError('Target ray is parallel to the measured table')
    target_depth = (calibration['geometry']['table_top_z_base_m']-transform[2,3])/target_ray[2]
    if target_depth <= 0:
        raise ValueError('Target behind camera')
    return dict(cup_bottom_base_m=center.tolist(),target_base_m=(transform[:3,3]+target_depth*target_ray).tolist(),
                cup_handle_yaw_rad=yaw,quality=float(keypoints['quality']),error_bound_m=bound)


class VisualObserver:
    def __init__(self, calibration, *, detector=None):
        self.calibration,self.detector = calibration,detector

    def measure(self, frame):
        result = GeometryObservation(frame_id=frame.frame_id,calibration_sha256=self.calibration['calibration_sha256'])
        try:
            width,height = self.calibration['camera_metadata']['image_size_wh']
            if frame.rgb.dtype != np.uint8 or frame.rgb.shape != (height,width,3):
                raise ValueError('RGB does not match calibrated image size')
            if frame.camera_metadata_hash != camera_metadata_hash(self.calibration['camera_metadata']):
                raise ValueError('Camera identity/intrinsics metadata does not match the current calibration')
            if self.detector is None:
                raise ValueError('No independently qualified body/rim/handle keypoint frontend; bounding boxes are insufficient')
            measurement = triangulate_upright_cup(self.detector(frame),self.calibration)
            return GeometryObservation(frame_id=frame.frame_id,calibration_sha256=result.calibration_sha256,
                                       valid=True,**measurement)
        except (KeyError,TypeError,ValueError,np.linalg.LinAlgError) as exc:
            result.invalid_reason = str(exc)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readonly', action='store_true', required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--frames-json',type=Path,required=True,
                        help='Saved frames with independently annotated semantic keypoints; offline diagnostic only')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    calibration = load_calibration(args.calibration,require_verified=False)
    records = []
    for row in json.loads(args.frames_json.read_text()):
        bgr = cv2.imread(row['image_path'])
        if bgr is None:
            raise ValueError('Cannot read saved frame')
        frame = CameraFrame(row['frame_id'],cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),
            row['exposure_monotonic_s'],row['received_monotonic_s'],row['camera_metadata_hash'],row.get('clock_verified',False))
        observer = VisualObserver(calibration,detector=lambda _,r=row:r['keypoints'])
        record = asdict(observer.measure(frame))
        if not np.isfinite(record['error_bound_m']):record['error_bound_m']=None
        records.append(record)
    with args.output.open('x') as stream:
        json.dump(dict(mode='annotated_frames_readonly_geometry_diagnostic',observations=records,
                       online_detector_qualified=False,motion_sent=False,hardware_deployment_allowed=False),stream,indent=2,allow_nan=False)


if __name__ == '__main__':
    main()
