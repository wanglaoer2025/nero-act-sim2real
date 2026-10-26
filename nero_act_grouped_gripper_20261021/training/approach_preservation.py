"""Training-only rehearsal of the frozen baseline's pre-GRASP action prefix."""
import math
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training.integrity import require_unchanged,snapshot
from executor.model import physical_actions


def validate_approach_replay(path):
    path=Path(path);proof_path=path.with_suffix('.json');proof=json.loads(proof_path.read_text())
    if (proof.get('complete') is not True or proof.get('profile')!='baseline_prefix_v1'
            or proof.get('zero_update_reference') is not True or proof.get('reference_optimizer_updates')!=0
            or proof.get('assessment_role')!='nominal_development_rehearsal_not_independent_acceptance'
            or not proof.get('source_sha256')
            or proof.get('replay_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError('Approach replay and zero-update provenance proof differ')
    require_unchanged(proof['source_sha256'])
    summary_path=Path(proof['reference_summary_path']).resolve();trace_path=Path(proof['reference_trace_path']).resolve()
    if not {str(summary_path),str(trace_path)}<=proof['source_sha256'].keys():
        raise ValueError('Reference trace and summary must be bound')
    summary=json.loads(summary_path.read_text())
    grasp=next((e for e in summary['result']['simulated_events'] if e['kind']=='GRASP'),None)
    if (summary['optimizer_updates']!=0 or grasp is None
            or grasp['physics_step']!=proof.get('cutoff_before_GRASP_physics_step')):
        raise ValueError('The approved zero-update pre-GRASP reference is required')
    with np.load(trace_path,allow_pickle=False) as trace:
        expected=np.flatnonzero(trace['query_steps']<grasp['physics_step']).tolist()
    with np.load(path,allow_pickle=False) as replay:n=len(replay['current'])
    if not n or proof.get('samples')!=n or proof.get('query_indices')!=expected or len(expected)!=n:
        raise ValueError('All and only baseline pre-GRASP queries must be rehearsed')
    return dict(proof['source_sha256'],**snapshot([path,proof_path]))


def prefix_preservation_loss(prediction,reference,stats,*,arm_unit_rad,gripper_unit_m):
    units=prediction.new_tensor([arm_unit_rad]*7+[gripper_unit_m])
    deployed=physical_actions(prediction,stats,bounded_gripper=True)
    fixed=physical_actions(reference.detach(),stats,bounded_gripper=True)
    error=(deployed[:,:2]-fixed[:,:2])/units
    return error.square().mean()


def runtime_predictions(policy,current,image):
    # CUDA kernels depend on batch shape; rehearse the deployed single-observation path.
    return torch.cat([policy(current[i:i+1],image[i:i+1]) for i in range(len(current))])


class ApproachPreservation:
    def __init__(self,settings,stats,device):
        if (settings.get('profile')!='baseline_prefix_v1'
                or type(settings.get('batch_size')) is not int or settings['batch_size']<=0
                or any(not math.isfinite(float(settings.get(k,0))) or float(settings.get(k,0))<=0
                       for k in ['weight','arm_unit_rad','gripper_unit_m'])):
            raise ValueError('Finite positive training-only approach preservation settings required')
        scale=np.asarray(stats['action_std'])
        if scale.shape!=(8,) or not np.isfinite(scale).all() or not (scale>0).all():
            raise ValueError('Paired positive action statistics required')
        self.settings=dict(settings);self.stats=stats;self.weight=float(settings['weight'])
        with np.load(Path(settings['replay_path']),allow_pickle=False) as f:
            current,image,target=[f[k] for k in ['current','image','target']]
        n=len(current)
        if (not n or current.shape!=(n,15) or target.shape!=(n,20,8)
                or image.ndim!=5 or image.shape[:3]!=(n,1,3)
                or any(not np.isfinite(v).all() for v in [current,image,target])
                or image.min()<0 or image.max()>1):
            raise ValueError('Finite paired RGB/15D/baseline20x8 approach replay required')
        self.current,self.image,self.target=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in [current,image,target]]
        # Match preprocess_images: identical RGB in a different layout changes CUDA convolution numerics.
        self.image=self.image[:,0].contiguous(memory_format=torch.channels_last).unsqueeze(1)
        self.last_indices=None

    def loss(self,policy,step):
        count=self.settings['batch_size']
        self.last_indices=(torch.arange(count,device=self.current.device)+step*count)%len(self.current)
        i=self.last_indices;training=policy.training
        # Preserve the actual deterministic inference path while retaining head gradients.
        policy.eval()
        try:prediction=runtime_predictions(policy,self.current[i],self.image[i])
        finally:policy.train(training)
        return prefix_preservation_loss(prediction,self.target[i],self.stats,
            arm_unit_rad=self.settings['arm_unit_rad'],gripper_unit_m=self.settings['gripper_unit_m'])

    def assess(self,policy):
        training=policy.training;policy.eval();maximum=np.zeros(2);squared=0.;count=0
        try:
            with torch.no_grad():
                for start in range(0,len(self.current),self.settings['batch_size']):
                    stop=start+self.settings['batch_size']
                    prediction=physical_actions(runtime_predictions(policy,self.current[start:stop],self.image[start:stop]),self.stats,bounded_gripper=True)
                    reference=physical_actions(self.target[start:stop],self.stats,bounded_gripper=True)
                    error=prediction[:,:2]-reference[:,:2]
                    maximum=np.maximum(maximum,[float(error[...,:7].abs().max()),float(error[...,7].abs().max())])
                    squared+=float(error[...,:7].square().sum());count+=error[...,:7].numel()
        finally:policy.train(training)
        return dict(samples=len(self.current),maximum_arm_prefix_change_rad=float(maximum[0]),
            rms_arm_prefix_change_rad=math.sqrt(squared/count),maximum_gripper_prefix_change_m=float(maximum[1]),
            training_replay_not_independent_validation=True)
