"""Shared v2 observer adapter for the retained physics runner; no truth features."""
from collections import deque
import copy
import numpy as np
import mujoco
from simulation.scene import build_scene
from observer import UnifiedObserver

class ExposureHistory:
    def __init__(self,delay_steps):
        if delay_steps not in (0,1):raise ValueError('Only0/1 exposure delay supported')
        self.rows=deque(maxlen=delay_steps+1);self.required=delay_steps+1
    def append(self,row):self.rows.append(copy.deepcopy(row))
    def reset(self):self.rows.clear()
    def take(self):
        if len(self.rows)<self.required:raise ValueError('Exposure warmup required')
        return copy.deepcopy(self.rows[0])

class ImageFeatures:
    def __init__(self,config,image_size,**kwargs):
        if kwargs.get('camera_translation_m') is not None:raise ValueError('Camera perturbation needs matching calibration')
        self.config=config;self.observer=None;self.index=0
        self.model=build_scene(config);self.data=mujoco.MjData(self.model)
        self.site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
    def tcp(self,q):
        self.data.qpos[:7]=q;mujoco.mj_forward(self.model,self.data)
        return self.data.site_xpos[self.site].copy()
    def __call__(self,image,state):
        if self.observer is None:
            profile=dict(camera=self.config['camera_calibration'],cup=self.config['cup'],table_top_z_m=self.config['table_top_z_m'])
            self.observer=UnifiedObserver(profile,image.shape[:2],tcp_from_q=self.tcp)
            self.observer.register([image.copy() for _ in range(3)])
        result=self.observer.measure(image,self.index*.05,self.index,state)
        self.index+=1
        record=dict(card=result['card'],cup_bottom_base_m=result['cup_bottom_base_m'].tolist(),
                    source=result['source'],truth_used_as_policy_input=False,real_geometry_qualified=False,
                    registration_temporal_evidence='synthetic_static_initial_snapshot_only')
        return result['relative_m'],record
