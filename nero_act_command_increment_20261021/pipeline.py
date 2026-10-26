"""Frozen source loading, causal target conversion, and checked development checkpoints."""
from pathlib import Path
import hashlib,json,pickle,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'nero_act_grouped_gripper_20261021'
sys.path.append(str(PARENT))
from training.history import initialize_history
from training.pose_objective import fk_spec_from_scene
from training.integrity import snapshot,require_unchanged
from executor.model import BoundedActPolicy
from model import IncrementHead,FrozenIncrementPolicy,PROFILE

DATA=(PARENT/'training_data/uniform_complete_v1').resolve()
INITIALIZER=ROOT.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'


def build_policy():
    if hashlib.sha256(INITIALIZER.read_bytes()).hexdigest()!='b252dd578b4bf5626566ec5b60a47be0bdcfee30e9eab990cde05ac122bdea71':
        raise ValueError('Canonical B252 initializer changed')
    settings=json.loads((PARENT/'config/history_experiment.json').read_text())
    scene=json.loads((DATA/'scene_config.json').read_text())
    settings.update(training_budget=dict(kind='updates',limit=1000),num_epochs=0,variant='A',stage='pilot',scene_config=scene,pose_kinematics_spec=fk_spec_from_scene(scene))
    source=torch.load(INITIALIZER,map_location='cpu',weights_only=False)
    stats=pickle.loads((DATA/'dataset_stats.pkl').read_bytes());old=pickle.loads((INITIALIZER.parent/'dataset_stats.pkl').read_bytes())
    config,weights=initialize_history(source,old,stats,settings,variant='A')
    if config['feedback_readout_version']!='feedback_tanh_v2' or config.get('numeric_history_profile','none')!='none':
        raise ValueError('Current15D/v2 source required')
    base=BoundedActPolicy(dict(config,device='cuda',pretrained_backbone=False)).cuda().eval();base.load_state_dict(weights,strict=True);base.config=config
    head=IncrementHead(config['hidden_dim'],stats).cuda()
    return FrozenIncrementPolicy(base,head,stats,config).cuda().eval(),stats


def prefix_targets(current,normalized_actions,padding,stats):
    mean=torch.as_tensor(stats['action_mean'],device=current.device,dtype=current.dtype)
    std=torch.as_tensor(stats['action_std'],device=current.device,dtype=current.dtype)
    qmean=torch.as_tensor(stats['qpos_mean'],device=current.device,dtype=current.dtype)
    qstd=torch.as_tensor(stats['qpos_std'],device=current.device,dtype=current.dtype)
    previous=current[:,7:8]*qstd[7]+qmean[7]+current[:,8:9]*.01
    targets=normalized_actions[:,:2,7]*std[7]+mean[7]
    valid=~padding[:,:2]
    delta=torch.where(valid,(targets-previous)/.001,torch.zeros_like(targets))
    if not torch.isfinite(delta).all() or torch.any(delta[valid].abs()>1.0001):
        raise ValueError('First2 teacher deltas exceed declared one-millimeter range')
    return targets,delta,valid


def validate_order(sample_ids,record):
    path=Path(record['path'])
    if (hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']
            or hashlib.sha256(json.dumps(sample_ids,separators=(',',':')).encode()).hexdigest()!=record['sample_ids_sha256']):
        raise ValueError('Frozen batch bytes or original sample ordering changed')
    order=np.load(path,allow_pickle=False)
    if (order.shape!=(1000,32) or not np.issubdtype(order.dtype,np.integer)
            or np.any(order < -1) or np.any(order >= len(sample_ids)) or np.any((order>=0).sum(1)==0)):
        raise ValueError('Original1000x32 valid batch schedule required')
    return order


def validated_replay(path):
    from training.approach_preservation import validate_approach_replay
    hashes=validate_approach_replay(path)
    with np.load(path,allow_pickle=False) as replay:
        if len(replay['current'])!=423:raise ValueError('All original423 reference states required')
    return hashes


def require_checkpoint(proof,path):
    path=Path(path).resolve()
    if (proof.get('complete') is not True or proof.get('actual_updates')!=1000 or proof.get('formal_epochs')!=0
            or proof.get('hardware_deployment_allowed') is not False or not proof.get('source_sha256')
            or str(path) not in proof.get('checkpoint_sha256',{})):
        raise ValueError('Finalized1000-update development checkpoint required')
    require_unchanged(proof['source_sha256']);require_unchanged(proof['checkpoint_sha256'])


def load_policy():
    path=ROOT/'training/pilot_A/policy_last.ckpt'
    proof=json.loads((path.parent/'integrity.json').read_text());require_checkpoint(proof,path)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('profile')!=PROFILE or payload.get('actual_updates')!=1000:raise ValueError('Wrong increment profile or budget')
    policy,stats=build_policy()
    if payload['config']!=policy.config:raise ValueError('Checkpoint/model source contract differs')
    policy.head.load_state_dict(payload['head'],strict=True);policy.eval()
    return policy,stats


def state_hash(module):
    digest=hashlib.sha256()
    for name,value in module.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
