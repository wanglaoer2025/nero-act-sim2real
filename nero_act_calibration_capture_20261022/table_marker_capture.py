"""Camera-only measured-marker/table-plane observations; no robot or ROS control."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from capture import dump,marker_pixels,install_cleanup_signals


def marker_plane(pixels,K,*,thickness_mm=None):
    px,K=np.asarray(pixels,float),np.asarray(K,float)
    if (px.shape!=(4,2) or K.shape!=(3,3) or not np.isfinite(np.r_[px.ravel(),K.ravel()]).all()
            or K[0,0]<=0 or K[1,1]<=0 or abs(cv2.contourArea(px.astype(np.float32)))<16
            or (thickness_mm is not None and (not np.isfinite(thickness_mm) or not 0<=thickness_mm<=30))):
        raise ValueError('Finite full marker, actual rectified K and optional measured thickness required')
    h=.095/2;obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
    found=cv2.solvePnPGeneric(obj,px,K,None,flags=cv2.SOLVEPNP_IPPE_SQUARE);candidates=[]
    if found[0]:
        for rv,tv in zip(found[1],found[2]):
            R=cv2.Rodrigues(rv)[0];t=np.asarray(tv).reshape(3)
            if not np.isfinite(np.r_[R.ravel(),t]).all() or np.any((obj@R.T+t)[:,2]<=0):continue
            projected=cv2.projectPoints(obj,rv,tv,K,None)[0].reshape(4,2)
            candidates.append((float(np.sqrt(np.mean(np.sum((projected-px)**2,axis=1)))),R,t))
    candidates.sort(key=lambda c:c[0])
    if not candidates or candidates[0][0]>1:raise ValueError('Positive-depth/reprojection check failed')
    if len(candidates)>1 and candidates[1][0]<=1:
        a,b=candidates[:2]
        if np.linalg.norm(a[2]-b[2])>.001 or Rotation.from_matrix(a[1].T@b[1]).magnitude()>np.deg2rad(1):
            raise ValueError('Planar pose ambiguous; use an oblique table view')
    error,R,t=candidates[0];n=R[:,2];d=float(-n@t)
    if d<0:n=-n;d=-d
    return dict(camera_to_board_plane_m=d,plane_normal_optical=n.tolist(),marker_center_optical_m=t.tolist(),
        camera_to_table_plane_m=None if thickness_mm is None else d+float(thickness_mm)/1000,
        board_thickness_mm=thickness_mm,reprojection_rmse_px=error,
        calibration_verified=False,hardware_deployment_allowed=False)


def combine_stationary(rows):
    if len(rows)<8:raise ValueError('At least8 valid repeated frames required')
    distance=np.array([v['camera_to_board_plane_m'] for v in rows]);normal=np.array([v['plane_normal_optical'] for v in rows])
    center=np.array([v['marker_center_optical_m'] for v in rows]);n=np.median(normal,axis=0);n/=np.linalg.norm(n)
    angles=np.degrees(np.arccos(np.clip(normal@n,-1,1)))
    if (not np.isfinite(np.r_[distance,normal.ravel(),center.ravel()]).all() or np.ptp(distance)>.005
            or np.max(angles)>2 or np.max(np.linalg.norm(center-np.median(center,axis=0),axis=1))>.005):
        raise ValueError('Stationary plane distance/normal/center spread rejected')
    thickness=rows[0]['board_thickness_mm']
    return dict(valid_frames=len(rows),distinct_physical_positions=1,plane_normal_optical=n.tolist(),
        camera_to_board_plane_m=float(np.median(distance)),board_distance_spread_mm=float(np.ptp(distance)*1000),
        board_thickness_mm=thickness,camera_to_table_plane_m=None if thickness is None else float(np.median(distance))+thickness/1000,
        marker_center_optical_m=np.median(center,axis=0).tolist(),maximum_normal_deviation_deg=float(np.max(angles)),
        independent_absolute_base_validation=False,calibration_verified=False,hardware_deployment_allowed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--thickness-mm',type=float);args=parser.parse_args()
    if args.thickness_mm is not None and (not np.isfinite(args.thickness_mm) or not 0<=args.thickness_mm<=30):parser.error('Measured nonnegative board thickness0..30mm required')
    args.output.mkdir(parents=True,exist_ok=False)
    install_cleanup_signals()
    tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    report=dict(schema='camera_table_plane_observation_v1',frames=[],sample_complete=False,
        native_image_rotation_degrees=0,marker_black_side_m=.095,marker_id=0,board_thickness_mm=args.thickness_mm,
        can_tx_start=tx(),robot_connected=False,robot_motion_sent=False,calibration_verified=False,
        hardware_deployment_allowed=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(),Path(__file__).resolve().with_name('capture.py'),
                      Path(__file__).resolve().parent/'marker_measured_95mm/marker_profile.json']})
    cam=None
    try:
        import pyzed.sl as sl
        cam=sl.Camera();init=sl.InitParameters();init.camera_resolution=sl.RESOLUTION.HD720;init.camera_fps=30
        init.depth_mode=sl.DEPTH_MODE.NONE;init.set_from_serial_number(39602891)
        if cam.open(init)!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('ZED open failed')
        res=sl.Resolution(1280,720);info=cam.get_camera_information(res)
        if info.serial_number!=39602891:raise RuntimeError('Camera identity mismatch')
        intr=info.camera_configuration.calibration_parameters.left_cam
        K=np.array([[intr.fx,0,intr.cx],[0,intr.fy,intr.cy],[0,0,1.]])
        report.update(camera_serial=info.serial_number,image_size_wh=[1280,720],K=K.tolist())
        params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
        detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),params)
        mat=sl.Mat();runtime=sl.RuntimeParameters();last=None;values=[]
        for index in range(24):
            if cam.grab(runtime)!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('New camera frame failed')
            if cam.retrieve_image(mat,sl.VIEW.LEFT,sl.MEM.CPU,res)!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('Native image retrieval failed')
            now=cam.get_camera_information(res).camera_configuration.calibration_parameters.left_cam
            if not np.array_equal(K,[[now.fx,0,now.cx],[0,now.fy,now.cy],[0,0,1.]]):raise RuntimeError('Camera intrinsics changed during observation')
            stamp=int(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if stamp==last:raise RuntimeError('Repeated native camera frame')
            last=stamp;im=np.array(mat.get_data(),copy=True)[...,:3]
            name=args.output/f'optical_{index:03d}.png'
            if not cv2.imwrite(str(name),im):raise OSError('Cannot preserve native calibration image')
            corners,ids,_=detector.detectMarkers(im);px=marker_pixels(corners,ids)
            row=dict(index=index,image_stamp_ns=stamp,receipt_monotonic_s=time.monotonic(),
                     image=str(name),image_sha256=hashlib.sha256(name.read_bytes()).hexdigest(),
                     marker_ids=None if ids is None else ids.ravel().tolist(),accepted=False)
            try:
                if px is None or np.any(px[:,0]<1) or np.any(px[:,0]>1278) or np.any(px[:,1]<1) or np.any(px[:,1]>718):raise ValueError('Unique complete in-frame ID0 required')
                value=marker_plane(px,K,thickness_mm=args.thickness_mm);row.update(accepted=True,corners_optical=px.tolist(),measurement=value);values.append(value)
            except ValueError as exc:row['error']=str(exc)
            report['frames'].append(row)
            if tx()!=report['can_tx_start']:raise RuntimeError('External CAN traffic changed during camera-only observation')
        if len(values)<20:raise ValueError('Fewer than20/24 qualified marker frames')
        if (report['frames'][-1]['image_stamp_ns']-report['frames'][0]['image_stamp_ns'])/1e9<.6:raise ValueError('Native observation span under600ms')
        report['measurement']=combine_stationary(values);report['sample_complete']=True
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        try:
            if cam is not None:cam.close()
        finally:
            report['can_tx_end']=tx();dump(args.output/'report.json',report)
    print(json.dumps(dict(sample_complete=report['sample_complete'],measurement=report['measurement'],CAN=[report['can_tx_start'],report['can_tx_end']]),indent=2))


if __name__=='__main__':main()
