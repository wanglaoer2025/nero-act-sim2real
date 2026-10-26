"""Pixel-aligned upright depth to metric base points; no object state is used."""
import numpy as np


def world_points_from_depth(depth,camera):
    depth=np.asarray(depth,float);width,height=camera['resolution_wh']
    k=np.asarray(camera['intrinsic_matrix'],float)
    transform=np.asarray(camera['transform_base_from_raw_optical_m'],float)
    if (depth.shape!=(height,width) or k.shape!=(3,3) or transform.shape!=(4,4)
            or camera['image_rotation_degrees']!=180):
        raise ValueError('Native pixel-paired depth and declared180-degree camera convention required')
    v,u=np.indices(depth.shape)
    pixels=np.stack([width-1-u,height-1-v,np.ones_like(u)],axis=-1)
    rays=pixels@np.linalg.inv(k).T
    camera_points=rays*depth[:,:,None]
    return camera_points@transform[:3,:3].T+transform[:3,3]
