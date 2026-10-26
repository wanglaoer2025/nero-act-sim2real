"""Declared offline arm-head schedules and read-only optimization diagnostics."""
import json
from pathlib import Path
import torch
from executor.model import physical_actions
from training.approach_preservation import runtime_predictions
from training.integrity import require_unchanged,snapshot


def require_optimizer_diagnostics(path,exclusions):
    path=Path(path);proof=json.loads(path.read_text())
    assessment=proof.get('initial_replay_assessment',{})
    if (proof.get('schema')!='optimizer_diagnostics_v1' or proof.get('complete') is not True
            or proof.get('actual_optimizer_updates')!=0 or proof.get('model_states_unchanged') is not True
            or proof.get('exact_teacher_conflict_groups_after_filter')!=0
            or proof.get('excluded_ambiguous_windows')!=exclusions
            or len(proof.get('gradient_records',[]))!=8 or not proof.get('source_sha256')
            or not 0<=assessment.get('maximum_arm_prefix_change_rad',float('inf'))<=1e-6
            or not 0<=assessment.get('maximum_gripper_prefix_change_m',float('inf'))<=1e-7):
        raise ValueError('Matched zero-update supervision and gradient diagnostics required')
    require_unchanged(proof['source_sha256'])
    return dict(proof['source_sha256'],**snapshot([path]))


def arm_rate(control,update):
    if (not isinstance(control,dict) or set(control)!={'schedule','initial_lr','final_lr','updates'}
            or control.get('schedule') not in ('fixed','linear') or control.get('updates')!=1000
            or control.get('initial_lr')!=1e-6
            or control.get('final_lr')!=(1e-6 if control.get('schedule')=='fixed' else 1e-7)
            or type(update) is not int or not 1<=update<=1000):
        raise ValueError('Declared1000-update fixed or linear arm1e-6 experiment required')
    return float(control['initial_lr']+(control['final_lr']-control['initial_lr'])*(update-1)/999)


def apply_arm_rate(policy,update):
    groups=policy.optimizer.param_groups
    expected=[list(policy.model.action_head.parameters()),list(policy.gripper_adapter.parameters())]
    if (not policy.output_heads_only or len(groups)!=2
            or any([id(p) for p in group['params']]!=[id(p) for p in params] for group,params in zip(groups,expected))):
        raise ValueError('Schedule requires the exact restricted arm and gripper optimizer groups')
    groups[0]['lr']=arm_rate(policy.optimizer_control,update)
    return dict(arm_lr=groups[0]['lr'],gripper_lr=groups[1]['lr'])


def gradient_relation(first,second):
    if first.shape!=second.shape or not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError('Matched finite gradient vectors required')
    one,two=[float(v.norm()) for v in [first,second]]
    cosine=float(torch.dot(first.flatten(),second.flatten())/(one*two)) if one>0 and two>0 else None
    return dict(first_norm=one,second_norm=two,cosine=cosine)


def prefix_probe(policy,keeper):
    indices=keeper.last_indices
    if indices is None:indices=torch.arange(min(8,len(keeper.current)),device=keeper.current.device)
    training=policy.training;policy.eval()
    try:
        with torch.no_grad():
            output=runtime_predictions(policy,keeper.current[indices],keeper.image[indices])
            return physical_actions(output,keeper.stats,bounded_gripper=True)[:,:2].detach().cpu()
    finally:policy.train(training)
