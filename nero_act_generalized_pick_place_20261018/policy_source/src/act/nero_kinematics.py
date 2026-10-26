"""Portable differentiable seven-axis FK; no SDK, MuJoCo or motion connection."""
import numpy as np
import torch
from torch import nn


class SerialArmKinematics(nn.Module):
    def __init__(self,specification):
        super().__init__()
        self.indices=[]
        def vector(value,shape):
            array=np.asarray(value,dtype=np.float64)
            if array.shape!=shape or not np.isfinite(array).all():
                raise ValueError('Invalid finite kinematics specification')
            return array
        for i,link in enumerate(specification['links']):
            rotation=vector(link['rotation'],(3,3))
            if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6):
                raise ValueError('Fixed link rotation must be proper SO(3)')
            self.register_buffer(f'rotation_{i}',torch.tensor(rotation),persistent=False)
            self.register_buffer(f'position_{i}',torch.tensor(vector(link['position'],(3,))),persistent=False)
            index=link.get('joint_index')
            if index is not None:
                if not isinstance(index,int) or not 0<=index<7:raise ValueError('Invalid arm joint index')
                axis=vector(link['axis'],(3,))
                if not np.isclose(np.linalg.norm(axis),1.,atol=1e-6):raise ValueError('Joint axis must have unit length')
                x,y,z=axis;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
                self.register_buffer(f'skew_{i}',torch.tensor(skew),persistent=False)
                self.register_buffer(f'joint_position_{i}',torch.tensor(vector(link['joint_position'],(3,))),persistent=False)
            self.indices.append(index)
        if sorted(index for index in self.indices if index is not None)!=list(range(7)):
            raise ValueError('Kinematics must contain each of seven arm joints exactly once')
        self.register_buffer('site_position',torch.tensor(vector(specification['tcp_position'],(3,))),persistent=False)

    def forward(self,joints):
        if joints.shape[-1]!=7:raise ValueError('Expected seven arm joints')
        identity=torch.eye(3,device=joints.device,dtype=joints.dtype)
        world_rotation=identity.expand(*joints.shape[:-1],3,3)
        world_position=torch.zeros_like(joints[...,:3])
        for i,index in enumerate(self.indices):
            rotation=getattr(self,f'rotation_{i}').to(joints)
            position=getattr(self,f'position_{i}').to(joints)
            if index is not None:
                angle=joints[...,index,None,None];skew=getattr(self,f'skew_{i}').to(joints)
                hinge=identity+angle.sin()*skew+(1-angle.cos())*(skew@skew)
                pivot=getattr(self,f'joint_position_{i}').to(joints)
                position=position+(rotation@(pivot-(hinge@pivot[...,None])[...,0])[...,None])[...,0]
                rotation=rotation@hinge
            world_position=world_position+(world_rotation@position[...,None])[...,0]
            world_rotation=world_rotation@rotation
        return world_position+(world_rotation@self.site_position.to(joints)[...,None])[...,0]
