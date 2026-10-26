"""One camera convention for calibrated scene export and ACT observation rendering."""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np


def add_calibrated_camera(world,config):
    calibration=config.get('camera_calibration')
    if calibration is None:return
    transform=np.asarray(calibration['transform_base_from_raw_optical_m'],dtype=float)
    intrinsic=np.asarray(calibration['intrinsic_matrix'],dtype=float)
    size=np.asarray(calibration['resolution_wh'],dtype=float)
    if (transform.shape!=(4,4) or intrinsic.shape!=(3,3) or size.shape!=(2,)
            or not np.isfinite(np.r_[transform.ravel(),intrinsic.ravel(),size]).all()
            or (size<=0).any() or (np.diag(intrinsic)[:2]<=0).any()
            or not np.allclose(transform[3],[0,0,0,1])
            or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-6)
            or np.linalg.det(transform[:3,:3])<.999):
        raise ValueError('Invalid metric rigid camera transform/intrinsics/resolution')
    if calibration.get('image_rotation_degrees')!=180:
        raise ValueError('Calibrated Nero camera expects already-upright180-degree RGB')
    rotation=transform[:3,:3]@np.diag([-1.,1.,-1.])
    quaternion=np.zeros(4);mujoco.mju_mat2Quat(quaternion,rotation.ravel())
    # MuJoCo principalpixel is an offset subtracted from (resolution-1)/2.
    principal=intrinsic[:2,2]-(size-1)/2
    ET.SubElement(world,'camera',name='calibrated_nero_camera',
                  pos=' '.join(map(str,transform[:3,3])),quat=' '.join(map(str,quaternion)),
                  sensorsize=' '.join(map(str,size)),resolution=' '.join(str(int(v)) for v in size),
                  focalpixel=f'{intrinsic[0,0]} {intrinsic[1,1]}',principalpixel=' '.join(map(str,principal)))


def observation_camera(config):
    if config.get('camera_calibration') is not None:return 'calibrated_nero_camera'
    view=config.get('camera',{'lookat':[.3,0,.2],'distance':1.1,'azimuth':90,'elevation':-45})
    camera=mujoco.MjvCamera();camera.lookat[:]=view['lookat'];camera.distance=view['distance']
    camera.azimuth=view['azimuth'];camera.elevation=view['elevation']
    return camera
