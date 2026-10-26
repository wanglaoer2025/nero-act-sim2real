"""Only a pregrasp mechanism gate, never a complete-task success predicate."""
from collections import defaultdict
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from learning import ROOT
from executor.contract import FullTaskContract
from executor.model import physical_actions
from executor.ros_execution import RobotGeometry
from executor.phases import PhaseGuard
from executor.contact import ContactMonitor
from training.visual_dataset import VisualReferenceDataset


def evaluate_response(policy,stats,manifest,*,checkpoint_sha256=None):
    plan=json.loads((ROOT/'plan.json').read_text())['gate']
    rows=[r for r in manifest['episodes'] if r['split']=='val' or
          (r['pose_key']=='actual_low_failure' and r['offset_xy_m']==[0.,0.])]
    records=[]
    for row in rows:
        path=Path(row['path'])
        with h5py.File(path) as f:
            scene=json.loads(f.attrs['scene_config'])
            if not f['observations/visual_valid'][0]:
                raise ValueError('Position response gate requires the actual first observation')
            state=np.r_[f['observations/qpos'][0],f['observations/gripper_state'][0]]
            previous=np.r_[state[:7],float(f.attrs['initial_previous_gripper_target_m'])]
        dataset=VisualReferenceDataset([path],['camera_0'],stats,policy_config=policy.config,
            image_size=tuple(policy.config['image_size']),sample_starts=[(0,0)])
        image,current,target,padding,_=dataset[0]
        with torch.inference_mode():
            raw=physical_actions(policy(current[None].cuda(),image[None].cuda()),stats,bounded_gripper=True)[0].cpu().numpy()
        label=target.numpy()*stats['action_std']+stats['action_mean']
        kin=RobotGeometry(scene);origin=kin.pose(state)['tcp'];error=None
        try:
            sent=FullTaskContract(policy.config).validate_prefix(raw,dict(state=state,age_s=.05),previous)[:2]
            poses=np.array([kin.pose(v)['tcp'] for v in sent])
            event=PhaseGuard(scene['pick_xy_m'],scene['place_xy_m']).inspect(origin,poses,sent[:,7],previous[7],ContactMonitor())
            if event is not None:raise ValueError('Pregrasp position gate requested '+event)
        except (ValueError,RuntimeError) as exc:
            error=str(exc);sent=raw[:2];poses=np.array([kin.pose(v)['tcp'] for v in sent])
        goals=np.array([kin.pose(v)['tcp'] for v in label[:2]])
        actor_delta=poses[0]-origin;teacher_delta=goals[0]-origin
        clearance=(row['role']!='clearance' or (error is None and actor_delta[2]>0
            and actor_delta[2]>=plan['minimum_teacher_response_fraction']*teacher_delta[2]
            and sent[:,7].min()>=previous[7]-1e-6))
        records.append(dict(id=row['id'],split=row['split'],pose_key=row['pose_key'],role=row['role'],
            offset_xy_m=row['offset_xy_m'],maximum_prefix_position_error_m=float(np.linalg.norm(poses-goals,axis=1).max()),
            actor_delta_m=actor_delta.tolist(),teacher_delta_m=teacher_delta.tolist(),
            clearance_passed=bool(clearance),guard_error=error))
    grouped=defaultdict(list)
    for row in records:
        if row['split']=='val':grouped[row['pose_key']].append(row)
    responses=[]
    for key,group in grouped.items():
        if len(group)!=4 or any(r['role']!='align' for r in group):continue
        for axis in [0,1]:
            high=[r for r in group if r['offset_xy_m'][axis]>0];low=[r for r in group if r['offset_xy_m'][axis]<0]
            expected=np.mean([r['teacher_delta_m'][axis] for r in high])-np.mean([r['teacher_delta_m'][axis] for r in low])
            actual=np.mean([r['actor_delta_m'][axis] for r in high])-np.mean([r['actor_delta_m'][axis] for r in low])
            if expected<=.00005:continue
            responses.append(dict(pose_key=key,axis=axis,teacher_response_m=float(expected),
                actor_response_m=float(actual),passed=bool(actual>0 and actual>=plan['minimum_teacher_response_fraction']*expected)))
    fraction=sum(r['passed'] for r in responses)/max(1,len(responses))
    maximum=max(r['maximum_prefix_position_error_m'] for r in records)
    passed=(len(responses)>=4 and fraction>=plan['minimum_correct_direction_fraction']
            and maximum<=plan['maximum_prefix_position_error_m']
            and all(r['clearance_passed'] and r['guard_error'] is None for r in records))
    return dict(passed=bool(passed),records=records,responses=responses,
        direction_pass_fraction=fraction,maximum_prefix_position_error_m=maximum,
        checkpoint_sha256=checkpoint_sha256,full_task_passed=False,
        assessment='Unseen offset pregrasp mechanism plus exact development failure; not full task qualification',
        hardware_commands_sent=False)
