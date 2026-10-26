"""Training-only pose supervision; never generates execution targets."""
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'policy_source'))
from src.act.nero_kinematics import SerialArmKinematics


def fk_spec_from_scene(config):
    import mujoco
    from simulation.scene import build_scene
    model=build_scene(config)
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
    if not np.allclose(model.site_quat[site],[1,0,0,0],atol=1e-12):
        raise ValueError('Pose objective requires the original unrotated TCP site')
    chain=[];body=int(model.site_bodyid[site])
    while body:
        chain.append(body);body=int(model.body_parentid[body])
    links=[]
    for body in reversed(chain):
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,model.body_quat[body])
        link=dict(position=model.body_pos[body].tolist(),rotation=rotation.reshape(3,3).tolist())
        count=int(model.body_jntnum[body])
        if count:
            if count!=1:raise ValueError('Expected a serial single-hinge robot chain')
            joint=int(model.body_jntadr[body]);index=int(model.jnt_qposadr[joint])
            if model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_HINGE or not 0<=index<7:
                raise ValueError('Pose objective may only use the seven physical arm joints')
            link.update(joint_index=index,axis=model.jnt_axis[joint].tolist(),
                joint_position=model.jnt_pos[joint].tolist())
        links.append(link)
    specification=dict(links=links,tcp_position=model.site_pos[site].tolist())
    SerialArmKinematics(specification)
    return specification


class CartesianPoseLoss(nn.Module):
    def __init__(self,specification,stats):
        super().__init__()
        self.kinematics=SerialArmKinematics(specification)
        for key in ['action_mean','action_std']:
            value=torch.as_tensor(stats[key],dtype=torch.float64)
            if value.shape!=(8,) or not torch.isfinite(value).all() or (key=='action_std' and not (value>0).all()):
                raise ValueError('Pose objective requires paired finite8D physical action statistics')
            self.register_buffer(key,value,persistent=False)

    def forward(self,prediction,target,padding):
        if (prediction.ndim!=3 or prediction.shape[1:]!=(20,8) or target.shape!=prediction.shape
                or padding.shape!=prediction.shape[:2] or padding.dtype!=torch.bool):
            raise ValueError('Pose loss requires normalized[B,20,8] actions and boolean padding')
        scale=self.action_std.to(prediction);offset=self.action_mean.to(prediction)
        p,r=self.kinematics((prediction*scale+offset)[...,:7],return_rotation=True)
        with torch.no_grad():
            goal_p,goal_r=self.kinematics((target*scale+offset)[...,:7],return_rotation=True)
        position=torch.linalg.vector_norm(p-goal_p,dim=-1)/.01
        rotation=torch.linalg.vector_norm((r-goal_r).flatten(-2),dim=-1)/(math.sqrt(2)*math.radians(1))
        weights=(~padding).to(prediction.dtype);weights[:,:10]*=5
        denominator=weights.sum().clamp_min(1)
        return (position*weights).sum()/denominator,(rotation*weights).sum()/denominator
