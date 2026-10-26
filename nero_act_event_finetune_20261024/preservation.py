"""Training-only rehearsal of CURRENT-camera successful ACT approach queries."""
import json
from pathlib import Path
import numpy as np
import torch
from executor.model import physical_actions
from training.approach_preservation import prefix_preservation_loss,runtime_predictions


def preservation_loss(prediction,reference,stats):
    return prefix_preservation_loss(prediction,reference,stats,arm_unit_rad=.001,gripper_unit_m=.001)


class CurrentCameraPreservation:
    def __init__(self,path,stats):
        self.stats=stats
        with np.load(path,allow_pickle=False) as f:
            values=[f[k] for k in ['current','image','target']]
        n=len(values[0])
        if (not n or values[0].shape!=(n,15) or values[2].shape!=(n,20,8)
                or values[1].ndim!=5 or values[1].shape[:3]!=(n,1,3)
                or any(not np.isfinite(x).all() for x in values)):
            raise ValueError('Complete finite current-camera preservation tuples required')
        self.current,self.image,self.target=[torch.as_tensor(v,dtype=torch.float32,device='cuda') for v in values]
        self.image=self.image[:,0].contiguous(memory_format=torch.channels_last).unsqueeze(1)

    def loss(self,policy,step):
        indices=(torch.arange(4,device=self.current.device)+step*4)%len(self.current)
        return preservation_loss(runtime_predictions(policy,self.current[indices],self.image[indices]),self.target[indices],self.stats)

    def assess(self,policy):
        maximum=np.zeros(2)
        with torch.no_grad():
            for start in range(0,len(self.current),4):
                predicted=runtime_predictions(policy,self.current[start:start+4],self.image[start:start+4])
                delta=physical_actions(predicted,self.stats,bounded_gripper=True)-physical_actions(self.target[start:start+4],self.stats,bounded_gripper=True)
                maximum=np.maximum(maximum,[float(delta[:,:2,:7].abs().max()),float(delta[:,:2,7].abs().max())])
        return dict(samples=len(self.current),maximum_arm_prefix_change_rad=float(maximum[0]),
            maximum_gripper_prefix_change_m=float(maximum[1]),current_camera_development_rehearsal=True)
