"""Receive-only current robot/camera capture; all SDK CAN sends are blocked."""
import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
from sensor_msgs.msg import JointState

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nero_act_transfer_resume_20261021'))
from robot_health import observer


def main():
    output=ROOT/'preflight_01';output.mkdir(exist_ok=False)
    tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    report=dict(time_local=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        receive_only=True,can_tx_before=tx(),arm_targets_sent=0,gripper_targets_sent=0,
        enable_sent=False,reset_sent=False)
    robot=node=camera=None
    try:
        rclpy.init(args=[]);node=rclpy.create_node('nero_current_trial_receive_only')
        states=[]
        def receive(message):
            names=[f'joint{i}' for i in range(1,8)]+['gripper']
            if all(message.name.count(name)==1 for name in names):
                states.append(dict(received_monotonic=time.monotonic(),state=[message.position[message.name.index(n)] for n in names]))
        node.create_subscription(JointState,'/feedback/joint_states',receive,10)
        deadline=time.monotonic()+5
        while len(states)<5:
            if time.monotonic()>deadline:raise TimeoutError('No fresh ROS joint feedback')
            rclpy.spin_once(node,timeout_sec=.02)
        report['ROS_feedback']=states[-1]
        report['nodes']=node.get_node_names_and_namespaces()
        report['control_publishers']={topic:[dict(node=p.node_name,namespace=p.node_namespace) for p in node.get_publishers_info_by_topic(topic)]
            for topic in ['/control/joint_states','/control/gated_joint_states']}
        robot=observer();health=robot.read_health(require_enabled=True)
        report['SDK_health']=health
        rclpy.spin_once(node,timeout_sec=.01)
        report['ROS_feedback']=states[-1]
        report['SDK_ROS_arm_max_difference_rad']=float(np.max(abs(np.asarray(health['state'])[:7]-np.asarray(states[-1]['state'])[:7])))
        camera=sl.Camera();init=sl.InitParameters()
        init.camera_resolution=sl.RESOLUTION.HD720;init.camera_fps=30;init.depth_mode=sl.DEPTH_MODE.NONE
        result=camera.open(init)
        if result!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('ZED open failed: '+str(result))
        runtime=sl.RuntimeParameters();mat=sl.Mat()
        for _ in range(3):
            if camera.grab(runtime)!=sl.ERROR_CODE.SUCCESS:raise RuntimeError('No current ZED frame')
        stamp=camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        if camera.retrieve_image(mat,sl.VIEW.LEFT,sl.MEM.CPU,sl.Resolution(1280,720))!=sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('ZED image retrieval failed')
        image=np.asarray(mat.get_data()).copy()[::-1,::-1,:3]
        native=output/'camera_native.png';small=output/'camera_000.png'
        if not cv2.imwrite(str(native),image) or not cv2.imwrite(str(small),cv2.resize(image,(640,360))):
            raise RuntimeError('Cannot save current scene')
        info=camera.get_camera_information();left=info.camera_configuration.calibration_parameters.left_cam
        report['camera']=dict(native_frame=str(native),frame=str(small),image_timestamp_ns=int(stamp),
            received_monotonic=time.monotonic(),intrinsics=dict(fx=left.fx,fy=left.fy,cx=left.cx,cy=left.cy),
            rotation='180degrees, same as retained pipeline',exposure_monotonic_mapping_verified=False)
        report['final_SDK_health']=robot.read_health(require_enabled=True)
    except Exception as exc:
        report['error']=repr(exc)
    finally:
        if camera is not None:camera.close()
        if robot is not None:robot.disconnect()
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        report['can_tx_after']=tx()
        if report['can_tx_after']!=report['can_tx_before']:report['error']='CAN TX changed during receive-only capture'
        (output/'report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist())+'\n')
        print(json.dumps(report,indent=2,default=lambda x:x.tolist()))
    if report.get('error'):raise SystemExit(1)


if __name__=='__main__':main()
