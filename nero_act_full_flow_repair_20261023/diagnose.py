"""Compare the same physical observations across frozen ACT weight revisions, offline only."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nero_act_paired_response_20261023'))
from experiment import load_latest,snapshot
from training.visual_dataset import VisualReferenceDataset
from executor.model import physical_actions
from executor.ros_execution import RobotGeometry


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    policy,stats,_,_=load_latest();policy.requires_grad_(False).eval()
    path=ROOT.parent/'nero_act_position_adaptation_20261022/data/visual/episode_1000.hdf5'
    with h5py.File(path) as f:
        valid=f['observations/visual_valid'][:];stage=f['simulation/stage'].asstr()[:]
        scene=json.loads(f.attrs['scene_config'])
        starts=[];labels=[]
        for name in ['approach','descend','close','lift','carry','place','release','retreat']:
            candidates=np.flatnonzero(valid&(stage==name))
            for i in np.linspace(0,len(candidates)-1,min(4,len(candidates))).astype(int):
                starts.append((0,int(candidates[i])));labels.append(name)
    dataset=VisualReferenceDataset([path],['camera_0'],stats,policy_config=policy.config,
        image_size=tuple(policy.config['image_size']),sample_starts=starts)
    values=[dataset[i] for i in range(len(starts))]
    image,current,target,padding=[torch.stack([r[k] for r in values]).cuda() for k in range(4)]
    scale=torch.as_tensor(stats['action_std'],device='cuda');geometry=RobotGeometry(scene)
    records=[]
    for name,checkpoint in [('projected',ROOT.parent/'nero_act_transfer_resume_20261021/training/projected_A/policy_last.ckpt'),
            ('before_pairs',ROOT.parent/'nero_act_position_response_continuation_20261023/training/pilot_A/policy_last.ckpt'),
            ('latest',ROOT.parent/'nero_act_paired_response_20261023/training/pilot_A/policy_last.ckpt')]:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if name=='projected':
            from checked_runtime import load_projected
            original,original_stats=load_projected()
            payload=dict(model=original.state_dict(),stats=original_stats)
        if any(not np.array_equal(payload['stats'][k],stats[k]) for k in stats):raise ValueError('Statistics mismatch')
        policy.load_state_dict(payload['model'],strict=True)
        with torch.inference_mode():prediction=policy(current,image)
        physical=physical_actions(prediction,stats,bounded_gripper=True).cpu().numpy()
        desired=physical_actions(target,stats,bounded_gripper=True).cpu().numpy()
        rows=[]
        for i,stage in enumerate(labels):
            goals=np.array([geometry.pose(row)['tcp'] for row in desired[i,:2]])
            actual=np.array([geometry.pose(row)['tcp'] for row in physical[i,:2]])
            rows.append(dict(stage=stage,step=starts[i][1],prefix_position_error_mm=float(np.linalg.norm(actual-goals,axis=1).max()*1000),
                prefix_joint_error_mrad=float(np.max(abs(physical[i,:2,:7]-desired[i,:2,:7]))*1000),
                prefix_gripper_error_mm=float(np.max(abs(physical[i,:2,7]-desired[i,:2,7]))*1000)))
        summary={stage:{key:float(np.mean([r[key] for r in rows if r['stage']==stage])) for key in
            ['prefix_position_error_mm','prefix_joint_error_mrad','prefix_gripper_error_mm']} for stage in set(labels)}
        records.append(dict(name=name,checkpoint_sha256=snapshot([checkpoint]),stage_mean_error=summary,rows=rows))
        print(json.dumps(dict(name=name,stages=summary)),flush=True)
    (ROOT/'weight_revision_diagnostics.json').write_text(json.dumps(dict(offline_only=True,training_updates=0,
        same_observations=True,training_scene_diagnostic_not_independent_validation=True,records=records),indent=2)+'\n')


if __name__=='__main__':main()
