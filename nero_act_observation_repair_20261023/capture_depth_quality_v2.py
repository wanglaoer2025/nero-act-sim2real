"""Capture ZED RGB/XYZ/confidence without ROS or any robot SDK connection."""
import json
from pathlib import Path
import time
import argparse
import hashlib

import cv2
import numpy as np
import pyzed.sl as sl

from depth_quality_v2 import upright_arrays,surface_quality,sequence_quality,cup_colour_roi,fresh_exposure

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth-mode',choices=['ULTRA','NEURAL_LIGHT'],default='ULTRA')
    parser.add_argument('--output',type=Path,default=ROOT/'depth_capture_v2')
    args=parser.parse_args()
    output=args.output;output.mkdir(exist_ok=False)
    camera=sl.Camera();samples=[];times=[];details=[]
    report=dict(complete=False,readonly=True,hardware_commands_sent=False,
        absolute_accuracy_verified=False,hardware_deployment_allowed=False,
        filled_depth_allowed=False,depth_mode=args.depth_mode,sdk_version=sl.Camera.get_sdk_version(),
        capture_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    local_sources=[Path(__file__),ROOT/'depth_quality_v2.py']
    report['source_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in local_sources}
    report['duplicate_exposures_rejected']=0
    try:
        init=sl.InitParameters()
        init.set_from_serial_number(39602891)
        init.camera_resolution=sl.RESOLUTION.HD720;init.camera_fps=30
        if args.depth_mode=='NEURAL_LIGHT':
            status=sl.check_ai_model_status(sl.AI_MODELS.NEURAL_LIGHT_DEPTH)
            report['local_neural_model_status_before']=status
            if status.get('downloaded') is not True:
                raise RuntimeError('No local neural depth weights; automatic download is forbidden')
        init.depth_mode=getattr(sl.DEPTH_MODE,args.depth_mode)
        init.coordinate_units=sl.UNIT.METER;init.coordinate_system=sl.COORDINATE_SYSTEM.IMAGE
        status=camera.open(init)
        if status!=sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('Read-only ZED open failed: '+str(status))
        info=camera.get_camera_information();configuration=info.camera_configuration
        def intrinsics(p):
            return dict(fx=float(p.fx),fy=float(p.fy),cx=float(p.cx),cy=float(p.cy),distortion=[float(v) for v in p.disto])
        report.update(serial_number=int(info.serial_number),camera_model=str(info.camera_model),
            rectified_left_intrinsics=intrinsics(configuration.calibration_parameters.left_cam),
            raw_left_intrinsics=intrinsics(configuration.calibration_parameters_raw.left_cam),
            image_rotation_degrees=180,xyz_vector_rotation_applied=False,
            xyz_coordinate_frame='SDK_IMAGE_left_optical',coordinate_units='metres',
            camera_robot_clock_alignment_verified=False)
        runtime=sl.RuntimeParameters();runtime.confidence_threshold=100
        runtime.texture_confidence_threshold=100;runtime.enable_fill_mode=False
        image,cloud,confidence=sl.Mat(),sl.Mat(),sl.Mat()
        grabbed=0
        for attempt in range(60):
            if camera.grab(runtime)!=sl.ERROR_CODE.SUCCESS:
                continue
            grabbed+=1
            if grabbed<=5:continue
            for mat,kind,is_image in [(image,sl.VIEW.LEFT,True),(cloud,sl.MEASURE.XYZRGBA,False),(confidence,sl.MEASURE.CONFIDENCE,False)]:
                status=camera.retrieve_image(mat,kind,sl.MEM.CPU) if is_image else camera.retrieve_measure(mat,kind,sl.MEM.CPU)
                if status!=sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError('Pixel-paired ZED retrieval failed: '+str(status))
            received=time.monotonic()
            stamp=int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if not fresh_exposure(stamp,times[-1] if times else None):
                report['duplicate_exposures_rejected']+=1
                continue
            rgba=np.asarray(image.get_data())
            rgb=cv2.cvtColor(rgba,cv2.COLOR_BGRA2RGB if rgba.shape[-1]==4 else cv2.COLOR_BGR2RGB)
            xyz=np.asarray(cloud.get_data())[:,:,:3]
            conf=np.asarray(confidence.get_data())
            if conf.ndim==3 and conf.shape[-1]==1:conf=conf[:,:,0]
            rgb,xyz,conf=upright_arrays(rgb,xyz,conf)
            mask,roi=cup_colour_roi(rgb)
            samples.append(dict(mask=mask,xyz=xyz,confidence=conf));times.append(stamp)
            index=len(samples)-1
            details.append(dict(index=index,device_exposure_ns=stamp,received_monotonic_s=received,
                roi=roi,depth_quality=surface_quality(mask,xyz,conf)))
            if index in [0,9,19]:
                np.savez_compressed(output/f'frame_{index:02d}.npz',rgb=rgb,xyz=xyz,
                    confidence=conf,mask=mask,device_exposure_ns=np.array(stamp))
                if not cv2.imwrite(str(output/f'frame_{index:02d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                    raise RuntimeError('Cannot preserve paired camera image')
            if len(samples)==20:break
        if len(samples)!=20:raise RuntimeError('Fewer than20 fresh depth exposures')
        for p,h in report['source_sha256'].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('Depth capture source changed')
        report.update(complete=True,sequence=sequence_quality(samples),frames=details,
            device_capture_fps=float(1e9/np.median(np.diff(times))),
            actual_robot_feedback_collected=False,object_bottom_estimation_performed=False)
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',frames=details)
    finally:
        camera.close()
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['frames','sequence']}),flush=True)
    if report['complete']:
        print(json.dumps({k:v for k,v in report['sequence'].items() if k!='frame_reports'}),flush=True)


if __name__=='__main__':main()
