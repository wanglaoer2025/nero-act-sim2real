"""Camera-only raw diagnostic; never a calibration bundle or robot executor."""
from pathlib import Path
import json,time
import cv2
import numpy as np
import pyzed.sl as sl

ROOT=Path(__file__).resolve().parent
out=ROOT/'stereo_table';out.mkdir(exist_ok=False)
scene=json.loads(Path('/home/user/wq/nero_act_transfer_resume_20261021/results/unqualified_nominal_full_development/scene_configuration.json').read_text())
old=np.array(scene['camera_calibration']['transform_base_from_raw_optical_m']);normal=old[2,:3]
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
report=dict(mode='camera_only_stereo_table_diagnostic',depth_mode='ULTRA',coordinates='IMAGE',units='meter',camera_serial_required=39602891,
    can_tx_start=tx(),frames=[],motion_sent=False,calibration_verified=False,hardware_deployment_allowed=False,
    nominal_camera_table_distance_m=float(old[2,3]-scene['table_top_z_m']),old_cup_dimensions_mm=[95,75],operator_reports_dimensions_unchanged=True)
cam=sl.Camera()
try:
 init=sl.InitParameters();init.camera_resolution=sl.RESOLUTION.HD720;init.camera_fps=30
 init.depth_mode=sl.DEPTH_MODE.ULTRA;init.coordinate_units=sl.UNIT.METER;init.coordinate_system=sl.COORDINATE_SYSTEM.IMAGE
 init.set_from_serial_number(39602891)
 status=cam.open(init)
 if status!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('ZED depth open: '+str(status))
 res=sl.Resolution(640,360);info=cam.get_camera_information(res);assert info.serial_number==39602891
 intr=info.camera_configuration.calibration_parameters.left_cam
 report['K']=[[intr.fx,0,intr.cx],[0,intr.fy,intr.cy],[0,0,1]]
 image,cloud,confidence=sl.Mat(),sl.Mat(),sl.Mat();runtime=sl.RuntimeParameters();runtime.confidence_threshold=50;runtime.texture_confidence_threshold=50
 for _ in range(10):
  if cam.grab(runtime)!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('Depth warmup failed')
 rng=np.random.default_rng(221022)
 for index in range(3):
  if cam.grab(runtime)!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('Depth grab failed')
  for result in [cam.retrieve_image(image,sl.VIEW.LEFT,sl.MEM.CPU,res),cam.retrieve_measure(cloud,sl.MEASURE.XYZRGBA,sl.MEM.CPU,res),cam.retrieve_measure(confidence,sl.MEASURE.CONFIDENCE,sl.MEM.CPU,res)]:
   if result!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('Depth retrieve failed: '+str(result))
  stamp=int(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
  native=np.array(image.get_data(),copy=True)[...,:3];xyz=np.array(cloud.get_data(),copy=True)[...,:3];conf=np.array(confidence.get_data(),copy=True)
  cv2.imwrite(str(out/f'optical_{index:03d}.png'),native);np.savez_compressed(out/f'depth_{index:03d}.npz',xyz_optical_m=xyz,confidence=conf,image_stamp_ns=stamp)
  upright=native[::-1,::-1];points=xyz[::-1,::-1];quality=conf[::-1,::-1];hsv=cv2.cvtColor(upright,cv2.COLOR_BGR2HSV)
  mask=(hsv[:,:,1]<40)&(hsv[:,:,2]>90)&(quality<=50)&np.isfinite(points).all(axis=2)&(points[:,:,2]>.3)&(points[:,:,2]<2.)
  area=np.zeros((360,640),bool);area[190:350,180:600]=True;mask&=area
  sample=points[mask].astype(float);row=dict(index=index,image_stamp_ns=stamp,selected_points=len(sample))
  if len(sample)<500:raise RuntimeError('Insufficient reliable white tabletop depth: '+str(len(sample)))
  if len(sample)>6000:sample=sample[rng.choice(len(sample),6000,replace=False)]
  best=np.zeros(len(sample),bool)
  for _ in range(150):
   a,b,c=sample[rng.choice(len(sample),3,replace=False)];n=np.cross(b-a,c-a);size=np.linalg.norm(n)
   if size<1e-8:continue
   n/=size;d=-n@a;inliers=abs(sample@n+d)<.003
   if inliers.sum()>best.sum():best=inliers
  fit=sample[best];center=fit.mean(0);_,_,vt=np.linalg.svd(fit-center,full_matrices=False);n=vt[-1];d=-n@center
  if n@normal<0:n=-n;d=-d
  errors=abs(sample@n+d)
  row.update(plane_normal_optical=n.tolist(),plane_offset_m=float(d),camera_to_plane_distance_m=float(abs(d)),inlier_fraction=float(np.mean(errors<.003)),median_abs_error_mm=float(np.median(errors)*1000),p95_abs_error_mm=float(np.percentile(errors,95)*1000),normal_vs_nominal_deg=float(np.degrees(np.arccos(np.clip(n@normal,-1,1)))),offset_vs_nominal_mm=float((d-report['nominal_camera_table_distance_m'])*1000))
  report['frames'].append(row)
  if tx()!=report['can_tx_start']:raise RuntimeError('External CAN TX changed during camera-only diagnosis')
except BaseException as exc:
 report['error']=repr(exc)
finally:
 cam.close();report['can_tx_end']=tx();(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)
