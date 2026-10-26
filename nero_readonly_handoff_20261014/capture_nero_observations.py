"""4090 read-only ROS feedback + ZED capture. No CAN commands or robot SDK connection.

Run on the real ROS computer with its existing feedback driver already active.
The saved pairing uses host receipt times; it does not establish hardware synchronization.
"""
import argparse
import json
import threading
import time
from pathlib import Path
import cv2
import numpy as np

JOINT_NAMES=[f'joint{i}' for i in range(1,8)]+['gripper']


def ordered_feedback(names,positions):
    if len(names)!=len(positions) or len(set(names))!=len(names):
        raise ValueError('Feedback has mismatched or duplicate names')
    mapping=dict(zip(names,positions))
    state=np.array([mapping[name] for name in JOINT_NAMES],dtype=np.float32)
    if not np.isfinite(state).all():raise ValueError('Nonfinite joint feedback')
    return state


def upright_rgb(frame):
    rotated=cv2.rotate(frame,cv2.ROTATE_180)
    code=cv2.COLOR_BGRA2RGB if rotated.shape[-1]==4 else cv2.COLOR_BGR2RGB
    return np.ascontiguousarray(cv2.cvtColor(rotated,code),dtype=np.uint8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--sample-hz',type=float,default=20)
    parser.add_argument('--joint-topic',default='/feedback/joint_states')
    parser.add_argument('--max-feedback-age-s',type=float,default=.2)
    args=parser.parse_args()
    if args.seconds<=0 or not 0<args.sample_hz<=30 or args.max_feedback_age_s<=0:
        parser.error('seconds/age must be positive; sample-hz must be in (0,30]')
    # These imports occur only for real capture, never for --help or unit tests.
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    import pyzed.sl as sl
    args.output.mkdir(parents=True,exist_ok=False)
    lock=threading.Lock();latest={};callback_errors=[];feedback_receipts=[];rows=[];frames=[];rejected=[]
    rclpy.init();node=rclpy.create_node('nero_act_readonly_observations')
    executor=SingleThreadedExecutor();executor.add_node(node)
    def callback(message):
        try:state=ordered_feedback(message.name,message.position)
        except (ValueError,KeyError) as exc:
            with lock:callback_errors.append(str(exc))
            return
        with lock:
            receipt=time.monotonic();feedback_receipts.append(receipt)
            latest.update(state=state,receipt=receipt,
                          ros_stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9)
    subscription=node.create_subscription(JointState,args.joint_topic,callback,qos_profile_sensor_data)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    camera=None;failure=None;interrupted=False;camera_metadata=None
    try:
        camera=sl.Camera();image=sl.Mat()
        init=sl.InitParameters();init.camera_resolution=sl.RESOLUTION.HD720
        init.camera_fps=30;init.depth_mode=sl.DEPTH_MODE.NONE
        status=camera.open(init)
        if status>sl.ERROR_CODE.SUCCESS:raise RuntimeError(f'ZED open failed: {status}')
        information=camera.get_camera_information().camera_configuration
        native=information.resolution
        intrinsics=information.calibration_parameters.left_cam
        camera_metadata={'sdk_version':str(sl.Camera.get_sdk_version()),
                         'native_width':int(native.width),'native_height':int(native.height),
                         'rectified_left_intrinsics_native':{'fx':float(intrinsics.fx),'fy':float(intrinsics.fy),
                                                            'cx':float(intrinsics.cx),'cy':float(intrinsics.cy)},
                         'intrinsics_coordinate_frame':'raw ZED optical, before 180 degree image rotation',
                         'view':'LEFT rectified; distortion not reapplied',
                         'capture_width':640,'capture_height':round(640*native.height/native.width)}
        resolution=sl.Resolution(640,round(640*native.height/native.width))
        runtime=sl.RuntimeParameters()
        started=time.monotonic();deadline=started;period=1/args.sample_hz
        while time.monotonic()-started<args.seconds:
            time.sleep(max(0,deadline-time.monotonic()))
            grab_started=time.monotonic()
            status=camera.grab(runtime)
            if status>sl.ERROR_CODE.SUCCESS:raise RuntimeError(f'ZED grab failed: {status}')
            camera.retrieve_image(image,sl.VIEW.LEFT,sl.MEM.CPU,resolution)
            rgb=upright_rgb(np.asarray(image.get_data()).copy())
            camera_received=time.monotonic()
            with lock:feedback=latest.copy();errors=callback_errors.copy()
            if errors:raise RuntimeError(f'Invalid feedback: {errors[-1]}')
            age=camera_received-feedback.get('receipt',float('-inf'))
            if not feedback or age>args.max_feedback_age_s:
                rejected.append({'time_monotonic_s':camera_received,'reason':'missing or stale feedback'})
            else:
                frames.append(rgb)
                rows.append({'state':feedback['state'],'sample_monotonic_s':camera_received,
                             'sample_wall_unix_s':time.time(),'joint_receipt_monotonic_s':feedback['receipt'],
                             'joint_ros_stamp_s':feedback['ros_stamp'],'camera_grab_started_monotonic_s':grab_started,
                             'camera_received_monotonic_s':camera_received,'joint_receipt_age_s':age})
            now=time.monotonic();deadline=max(deadline+period,now)
    except KeyboardInterrupt:interrupted=True
    except Exception as exc:failure=str(exc)
    finally:
        if camera is not None:camera.close()
        executor.shutdown(timeout_sec=2);thread.join(timeout=2)
        node.destroy_subscription(subscription);node.destroy_node();rclpy.shutdown()
    if rows:
        arrays={key:np.array([row[key] for row in rows]) for key in rows[0]}
        arrays['rgb']=np.stack(frames);arrays['joint_names']=np.array(JOINT_NAMES)
        arrays['all_feedback_receipt_monotonic_s']=np.array(feedback_receipts)
        np.savez_compressed(args.output/'observations.npz',**arrays)
        dt=np.diff(arrays['sample_monotonic_s'])
        achieved_hz=float(1/np.mean(dt)) if len(dt) else None
        cv2.imwrite(str(args.output/'camera_0_upright_rgb.png'),cv2.cvtColor(frames[0],cv2.COLOR_RGB2BGR))
    else:achieved_hz=None
    feedback_dt=np.diff(feedback_receipts)
    feedback_hz=float(1/np.mean(feedback_dt)) if len(feedback_dt) else None
    report={'mode':'read_only_live_capture','robot':'Nero single arm AGX gripper','joint_names':JOINT_NAMES,
            'arm_unit':'rad','gripper_unit':'m total width','joint_topic':args.joint_topic,
            'rgb_order':'RGB','image_rotation_applied_degrees':180,'image_already_upright':True,
            'camera_configured_fps':30,'requested_sample_hz':args.sample_hz,
            'camera_metadata':camera_metadata,
            'achieved_sample_hz':achieved_hz,'samples':len(rows),'rejected':rejected,
            'feedback_messages':len(feedback_receipts),'observed_feedback_receipt_hz':feedback_hz,
            'feedback_age_gate':'host message receipt, not measured sensor age',
            'hardware_synchronized':False,'robot_sdk_connected':False,'motion_sent':False,
            'interrupted':interrupted,'failure':failure,'real_transfer_verified':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return int(bool(failure) or not rows)


if __name__=='__main__':raise SystemExit(main())
