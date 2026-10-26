"""Discarded-weight learning probe on training groups, using frozen decoder features."""
import json
import pickle
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from executor.model import physical_actions, feedback_readout
from training.command_context import ReferenceDataset
from training.train import FullTaskPolicy, weighted_l1

SOURCE=ROOT.parent/'nero_act_gripper_contact_focus_20261018/model/run26_current_fixed_scene'


def cached_readout(policy,feature,q):
    raw=policy.absolute_prediction(policy.model.action_head(feature),q)
    paired={key:getattr(policy,'feedback_'+key) for key in ['qpos_mean','qpos_std','action_mean','action_std']}
    bounded=feedback_readout(raw,q[:,:8],paired,policy.feedback_lower,policy.feedback_upper,version=policy.feedback_version)
    return raw,bounded


def configuration(device='cuda'):
    config=json.loads((SOURCE/'training_config.json').read_text())
    config.update(device=device, pretrained_backbone=False,
        train_nonlinear_gripper_only=False, train_arm_output_and_gripper_only=True,
        train_gripper_head_only=False, freeze_backbone=True, reset_action_head=False,
        arm_output_lr=.0001, gripper_adapter_lr=.0003, prior_gripper_loss_weight=.005,
        training_observation_noise_normalized=[0.]*8,
        balance_gripper_transitions=False, focus_grasp_recoveries=False,
        gripper_sampling='critical_windows_complete_stride10_plus_first20_close_lift_release',
        critical_input_sampling=dict(stages=['close','lift','release'], first_points=20,
            lift_first5_repeats=6, lift_later_repeats=2, other_added_repeats=1,
            original_wait_filters_preserved=True, descend_recovery_all_inputs_20hz=False),
        visual_learning_control='visual_CVAE_decoder_frozen; seven_arm_output_rows_and_existing_gripper_adapter_adapt')
    for key in ['correction_manifest','additional_correction_manifests','correction_role',
                'prior_evaluation_scope','correction_source_checkpoint_sha256']:
        config.pop(key,None)
    return config


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    config=configuration()
    with (SOURCE/'dataset_stats.pkl').open('rb') as stream:stats=pickle.load(stream)
    manifest=json.loads((ROOT/'training_data/fixed_v3/manifest.json').read_text())
    assert manifest['complete']
    records=[]
    for row in manifest['episodes']:
        if row['split']!='train' or row['id'] not in [5000,5003,5006,5009,5100]:continue
        with h5py.File(row['path'],'r') as stream:stages=stream['simulation/stage'].asstr()[:]
        for stage in ['approach','descend','close','lift','carry','place','release','retreat']:
            indices=np.flatnonzero(stages==stage)
            if not len(indices):continue
            starts=[int(indices[0])]
            if stage in ['close','release']:starts.append(int(indices[min(10,len(indices)-1)]))
            records.extend(dict(path=row['path'],start=t,stage=stage,id=row['id']) for t in starts)
    policy=FullTaskPolicy(config).to(config['device']).eval()
    payload=torch.load(SOURCE/'policy.ckpt',map_location='cpu',weights_only=False)
    policy.load_state_dict(payload['model'],strict=True)
    feature=[];captured=[];observed=[];targets=[];padding=[]
    handle=policy.model.action_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    for row in records:
        image,q,target,pad,_=ReferenceDataset([Path(row['path'])],config['camera_names'],stats,
            fixed_start_ts=row['start'])[0]
        with torch.no_grad():policy(q[None].cuda(),image[None].cuda())
        feature.append(captured.pop());observed.append(q);targets.append(target[:20]);padding.append(pad[:20])
    handle.remove()
    feature=torch.cat(feature);q=torch.stack(observed).cuda();target=torch.stack(targets).cuda();pad=torch.stack(padding).cuda()
    before={key:value.clone() for key,value in policy.state_dict().items()}
    def prediction():
        return cached_readout(policy,feature,q)
    with torch.no_grad():initial=float(weighted_l1(prediction()[1],target,pad))
    for _ in range(4000):
        raw,bounded=prediction()
        loss=weighted_l1(bounded,target,pad)+config['prior_gripper_loss_weight']*weighted_l1(raw[...,7:8],target[...,7:8],pad)
        policy.optimizer.zero_grad(set_to_none=True);loss.backward();policy.optimizer.step()
    with torch.no_grad():
        raw,bounded=prediction();final=float(weighted_l1(bounded,target,pad))
        physical=physical_actions(bounded,stats,bounded_gripper=True)
        desired=physical_actions(target,stats,bounded_gripper=True)
        difference=(physical[:,:10]-desired[:,:10]).abs().mean(dim=1)
    changed=[key for key,value in policy.state_dict().items() if not torch.equal(value,before[key])]
    allowed={'model.action_head.weight','model.action_head.bias'}
    frozen=all(key in allowed or key.startswith('gripper_adapter.') for key in changed)
    frozen &= all(torch.equal(policy.state_dict()[key][7],before[key][7]) for key in allowed)
    errors=[dict(**row,arm_mae_rad=float(error[:7].mean()),gripper_mae_mm=float(error[7])*1000)
            for row,error in zip(records,difference)]
    passed=bool(frozen and final<initial and max(e['arm_mae_rad'] for e in errors)<.003
                and max(e['gripper_mae_mm'] for e in errors)<1.)
    report=dict(passed=passed,probe_only_not_full_task_success=True,weights_saved=False,
        initial_l1=initial,final_l1=final,steps=4000,changed_tensors=changed,
        frozen_base_and_legacy_gripper_row_passed=bool(frozen),training_groups_only=True,
        cached_frozen_zero_latent_decoder=True,hardware_connected=False,records=errors)
    (ROOT/'evidence/fixed_heads_probe_v3.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
    if not passed:raise RuntimeError('Head learning probe failed; no full training started')


if __name__=='__main__':main()
