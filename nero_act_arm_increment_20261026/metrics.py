"""Frozen deployment-error diagnosis, never a physical success score."""
import collections
import torch
from torch.utils.data import default_collate
from experiment import BASE
from dataset import V2Dataset


def representative_dataset(manifest,stats):
    paths=[];splits=[]
    for split,count in [('train',12),('dev',6)]:
        for i in range(count):
            row=next(r for r in sorted(manifest['episodes'],key=lambda r:r['path'])
                     if r['position_group']==f'{split}_{i:02d}')
            paths.append(row['path']);splits.append(split)
    return V2Dataset(paths,stats,stride=80),splits


def deployment_metrics(policy,dataset,splits):
    sums=collections.defaultdict(collections.Counter)
    policy.eval()
    for offset in range(0,len(dataset),32):
        batch=default_collate([dataset[i] for i in range(offset,min(offset+32,len(dataset)))])
        t={k:v.to(policy.device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
        with torch.inference_mode():
            pred=policy(t['observation'],t['image'])
            target=t['target'];valid=~t['padding'];valid2=valid[:,:2]
            arm=(pred[:,:,:7]-target[:,:,:7]).abs()
            physical=arm[:,:2]*policy.action_std[:7]
            current=t['observation'][:,:7]*policy.qpos_std[:7]+policy.qpos_mean[:7]
            teacher=target[:,:2,:7]*policy.action_std[:7]+policy.action_mean[:7]
            predicted=pred[:,:2,:7]*policy.action_std[:7]+policy.action_mean[:7]
            move=teacher-current[:,None,:7];chosen=predicted-current[:,None,:7]
            grip=(pred[:,:2,7]-target[:,:2,7]).abs()*policy.action_std[7]
        for j,stage in enumerate(batch['stage']):
            v,v2=valid[j],valid2[j]
            dot=(move[j]*chosen[j]).sum(-1)
            meaningful=(move[j].norm(dim=-1)>1e-5)&v2
            row=dict(samples=1,arm20_sum=float(arm[j][v].sum()),arm20_n=int(v.sum())*7,
                prefix_sum=float(physical[j][v2].sum()),prefix_n=int(v2.sum())*7,
                teacher_sum=float(move[j][v2].abs().sum()),grip_sum=float(grip[j][v2].sum()),
                grip_n=int(v2.sum()),positive_direction=int((dot[meaningful]>0).sum()),
                directional_n=int(meaningful.sum()))
            for group in ['all',splits[dataset.samples[offset+j][0]],stage]:sums[group].update(row)
    report={}
    for group,s in sums.items():
        error=s['prefix_sum']/max(1,s['prefix_n'])
        motion=s['teacher_sum']/max(1,s['prefix_n'])
        report[group]=dict(samples=s['samples'],deployment_arm20_normalized_MAE=s['arm20_sum']/max(1,s['arm20_n']),
            deployment_prefix2_MAE_rad=error,teacher_prefix2_motion_mean_abs_rad=motion,
            prefix_error_to_motion_ratio=error/max(motion,1e-12),
            gripper_prefix2_MAE_m=s['grip_sum']/max(1,s['grip_n']),
            positive_motion_direction_fraction=s['positive_direction']/max(1,s['directional_n']))
    return dict(scope='representative18episodes_stride80_not_full_task',records=report,
                optimizer_updates=0,hardware_commands_sent=False)
