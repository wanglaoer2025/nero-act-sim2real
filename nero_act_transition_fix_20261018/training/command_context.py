"""Causal training observations and explicit initialization for the one-command channel."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from executor.model import make_policy_observation
from executor.contract import GRIPPER_CONTEXT
from src.act.nero_dataset import NeroEpisodicDataset


class ReferenceDataset(NeroEpisodicDataset):
    def __getitem__(self,index):
        image,_,actions,padding,start=super().__getitem__(index)
        path=self.paths[index] if self.sample_starts is None else self.paths[self.sample_starts[index][0]]
        with h5py.File(path,'r') as root:
            state=np.r_[root['observations/qpos'][start],root['observations/gripper_state'][start]]
            if start:previous=float(root['action'][start-1,7])
            else:
                config=json.loads(root.attrs.get('scene_config','{}'))
                previous=config.get('initial_simulation_snapshot',{}).get('previous_target',state)[7]
        observed=make_policy_observation(state,self.stats,dict(state_dim=8,physical_state_dim=8,policy_input_dim=9,
            gripper_command_context=GRIPPER_CONTEXT),previous)
        return image,torch.from_numpy(observed),actions,padding,start


def expand_context_state_dict(state):
    result=dict(state)
    for key in ['model.input_proj_robot_state.weight','model.encoder_joint_proj.weight']:
        weight=state[key]
        if weight.ndim!=2 or weight.shape[1]!=8:raise ValueError('Only explicit8→9 initialization is supported')
        expanded=weight.new_zeros((weight.shape[0],9));expanded[:,:8]=weight
        result[key]=expanded
    return result


def prepare_context_initializer(source,config,destination):
    source,destination=Path(source),Path(destination)
    payload=torch.load(source,map_location='cpu',weights_only=False)
    if payload['config'].get('state_dim')!=8 or payload['config'].get('gripper_command_context'):
        raise ValueError('Context expansion requires an explicit legacy8D initializer')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    torch.save(dict(model=expand_context_state_dict(payload['model']),config=config,
        initialization_only=True,hardware_deployment_allowed=False,
        source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),destination)
    return destination
