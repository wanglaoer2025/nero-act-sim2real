"""Paired conditional-response learning, entirely offline within the existing ACT."""
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from paired import paired_response_loss,worst_prefix_position_loss

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'nero_act_position_response_continuation_20261023'
sys.path.insert(0,str(SOURCE))
from resume import load_initial as load_previous_initial,prior,require_unchanged,snapshot
from objective import executed_objective,preservation_scale
from response import evaluate_response
from training.pose_objective import CartesianPoseLoss,fk_spec_from_scene
from training.approach_preservation import ApproachPreservation,validate_approach_replay

INITIALIZER=SOURCE/'training/pilot_A/policy_last.ckpt'
PROFILE='paired_position_response_v1'


def load_latest():
    proof=json.loads((INITIALIZER.parent/'integrity.json').read_text())
    if (proof.get('complete') is not True or proof.get('actual_updates')!=3000
            or proof.get('hardware_deployment_allowed') is not False):
        raise ValueError('Completed offline continuation required')
    require_unchanged(proof['source_sha256']);require_unchanged(proof['checkpoint_sha256'])
    if snapshot([INITIALIZER])[str(INITIALIZER.resolve())]!=json.loads((ROOT/'plan.json').read_text())['initializer_sha256']:
        raise ValueError('Wrong fixed-last initializer')
    payload=torch.load(INITIALIZER,map_location='cpu',weights_only=False)
    if payload.get('profile')!='position_response_continuation_v1' or payload.get('actual_updates')!=3000:
        raise ValueError('Wrong initializer profile or budget')
    policy,stats,groups=load_previous_initial()
    if payload['config']!=policy.config or any(not np.array_equal(payload['stats'][k],stats[k]) for k in stats):
        raise ValueError('Unchanged paired ACT configuration/statistics required')
    policy.load_state_dict(payload['model'],strict=True)
    return policy,stats,groups,payload['optimizer']


def position_pair(prediction,target,pose_loss):
    scale=pose_loss.action_std.to(prediction);mean=pose_loss.action_mean.to(prediction)
    positions=pose_loss.kinematics((prediction[:,:2]*scale+mean)[...,:7],return_rotation=True)[0]
    with torch.no_grad():
        labels=pose_loss.kinematics((target[:,:2]*scale+mean)[...,:7],return_rotation=True)[0]
    return positions,labels


def prepare_pairs(dataset,manifest):
    rows=[r for r in manifest['episodes'] if r['split']=='train' and r['role']=='align']
    keyed={(r['pose_key'],tuple(r['offset_xy_m'])):r for r in rows}
    samples={(str(dataset.paths[i].resolve()),t):index for index,(i,t) in enumerate(dataset.sample_starts)}
    records=[];indices=[]
    for pose in sorted({r['pose_key'] for r in rows}):
        for axis in [0,1]:
            for other in [-.03,0.,.03]:
                low=[0.,0.];high=[0.,0.]
                low[axis]=-.03;high[axis]=.03;low[1-axis]=high[1-axis]=other
                left=keyed.get((pose,tuple(low)));right=keyed.get((pose,tuple(high)))
                if left is None or right is None:continue
                with h5py.File(left['path']) as a,h5py.File(right['path']) as b:
                    if (not np.array_equal(a['observations/qpos'][0],b['observations/qpos'][0])
                            or not np.array_equal(a['observations/gripper_state'][0],b['observations/gripper_state'][0])
                            or a.attrs['initial_previous_gripper_target_m']!=b.attrs['initial_previous_gripper_target_m']):
                        raise ValueError('Counterfactual pair must share exact initial robot state and command')
                    sa,sb=json.loads(a.attrs['scene_config']),json.loads(b.attrs['scene_config'])
                    delta=np.asarray(sb['pick_xy_m'])-sa['pick_xy_m']
                    expected=np.zeros(2);expected[axis]=.06
                    if not np.allclose(delta,expected,atol=1e-10,rtol=0) or sa['place_xy_m']!=sb['place_xy_m']:
                        raise ValueError('Pair must change only the declared cup-position axis, with fixed target')
                pair=[samples[(str(Path(r['path']).resolve()),0)] for r in [left,right]]
                records.append(dict(pose_key=pose,axis=axis,other_offset_m=other,
                    ids=[left['id'],right['id']],sample_indices=pair,split='train'))
                indices.extend(pair)
    if len(records)!=24:raise ValueError('All24 training-only same-pose intervention pairs required')
    values=[dataset[i] for i in indices]
    cache=tuple(torch.stack([v[k] for v in values]).cuda() for k in range(4))
    if cache[3][:,:2].any():raise ValueError('All paired first2 labels must be executable')
    return records,cache


def choose_pairs(cache,ids):
    indices=torch.as_tensor(np.array([[2*i,2*i+1] for i in ids]).ravel(),device='cuda')
    return tuple(value[indices] for value in cache)


def pair_metrics(policy,cache,records,ids,pose_loss):
    image,current,target,padding=choose_pairs(cache,ids)
    with torch.inference_mode():
        prediction=policy(current,image);positions,labels=position_pair(prediction,target,pose_loss)
    paired=positions.reshape(-1,2,2,3);expected=labels.reshape(-1,2,2,3)
    axes=[]
    for index,pair in enumerate(ids):
        axis=records[pair]['axis'];teacher=float(expected[index,1,0,axis]-expected[index,0,0,axis])
        actual=float(paired[index,1,0,axis]-paired[index,0,0,axis])
        axes.append(dict(pose_key=records[pair]['pose_key'],axis=axis,teacher_response_m=teacher,
            actor_response_m=actual,passed=teacher>0 and actual>=.25*teacher))
    error=float(torch.linalg.vector_norm(positions-labels,dim=-1).max())
    return dict(passed=all(r['passed'] for r in axes) and error<=.0005,axes=axes,
        maximum_prefix_error_m=error,full_task_passed=False,assessment='Raw training-pair learnability only; not protected rollout or task qualification')


def assert_frozen(policy,frozen):
    if any(not torch.equal(p.detach().cpu(),frozen[n]) for n,p in policy.named_parameters() if n in frozen):
        raise ValueError('Frozen visual/CVAE parameters changed')


def main():
    torch.set_num_threads(2);torch.manual_seed(44);np.random.seed(44)
    policy,stats,groups,_=load_latest()
    dataset,old_batches,manifest,scene,files=prior.prepare_data(policy,stats)
    records,cache=prepare_pairs(dataset,manifest)
    (ROOT/'pairs.json').write_text(json.dumps(records,indent=2)+'\n')
    rng=np.random.default_rng(44)
    batches=np.concatenate([old_batches[rng.permutation(len(old_batches))] for _ in range(2)])
    prefix_paths={str(Path(r['path']).resolve()) for r in manifest['episodes'] if r['split']=='train'}
    is_prefix=np.array([str(dataset.paths[i].resolve()) in prefix_paths for i,_ in dataset.sample_starts])
    if batches.shape!=(2000,32) or not is_prefix[batches[:,:16]].all():
        raise ValueError('All2000 pointwise batches must start with16 audited correction prefixes')
    pair_order=rng.integers(0,len(records),size=(2000,8))
    np.save(ROOT/'batch_order.npy',batches,allow_pickle=False);np.save(ROOT/'pair_order.npy',pair_order,allow_pickle=False)
    files+=list(ROOT.glob('*.py'))+[ROOT/'plan.json',ROOT/'pairs.json',ROOT/'batch_order.npy',ROOT/'pair_order.npy',
        ROOT/'data/scene_protocol.json',INITIALIZER,INITIALIZER.parent/'integrity.json',
        ROOT.parent/'nero_act_position_response_repair_20261023/plan.json',
        ROOT.parent/'nero_act_position_response_repair_20261023/data/pose_protocol.json',
        ROOT.parent/'nero_act_position_recovery_20261022/evaluate.py',
        ROOT.parent/'nero_act_position_adaptation_20261022/data/scene_protocol.json',
        ROOT.parent/'nero_act_command_increment_20261021/run_qualification.py']
    files += [Path(m.__file__) for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
        and str(m.__file__).startswith(str(ROOT.parent)+'/') and str(m.__file__).endswith('.py')]
    hashes=snapshot(files)
    hashes.update(validate_approach_replay(policy.config['approach_preservation']['replay_path']))
    pose_loss=CartesianPoseLoss(fk_spec_from_scene(scene),stats).cuda()
    probe_ids=[next(i for i,r in enumerate(records) if r['pose_key']==pose and r['axis']==axis and r['other_offset_m']==0)
        for pose,axis in [('current_0',1),('current_800',0)]]
    probe=ROOT/'probe';probe.mkdir(exist_ok=False)
    before=pair_metrics(policy,cache,records,probe_ids,pose_loss)
    frozen={n:p.detach().cpu().clone() for n,p in policy.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW(groups,weight_decay=0.)
    image,current,target,padding=choose_pairs(cache,probe_ids)
    for step in range(200):
        optimizer.zero_grad(set_to_none=True);prediction=policy(current,image)
        base,_=executed_objective(prediction,target,padding,stats,pose_loss)
        positions,labels=position_pair(prediction,target,pose_loss)
        response=paired_response_loss(positions.reshape(-1,2,2,3),labels.reshape(-1,2,2,3))
        loss=base+5*response+worst_prefix_position_loss(positions,labels)
        if not torch.isfinite(loss):raise ValueError('Nonfinite probe loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
    assert_frozen(policy,frozen);require_unchanged(hashes)
    after=pair_metrics(policy,cache,records,probe_ids,pose_loss)
    torch.save(dict(profile='disposable_paired_probe',model=policy.state_dict(),stats=stats,config=policy.config,
        actual_updates=200,hardware_deployment_allowed=False),probe/'policy_last.ckpt')
    probe_result=dict(updates=200,before=before,after=after,passed=after['passed'],training_pair_ids=probe_ids,
        checkpoint_sha256=snapshot([probe/'policy_last.ckpt']),source_sha256=hashes,
        weights_discarded_for_candidate=True,hardware_commands_sent=False)
    (probe/'result.json').write_text(json.dumps(probe_result,indent=2)+'\n')
    print(json.dumps(dict(stage='probe_complete',passed=after['passed'],before=before,after=after)),flush=True)
    if not after['passed']:
        (ROOT/'scope.json').write_text(json.dumps(dict(candidate_trained=False,full_task_passed=False,
            reason='The two training-pair learnability test did not pass; no larger training launched',hardware_commands_sent=False),indent=2)+'\n')
        return
    del policy,optimizer;torch.cuda.empty_cache()
    policy,stats,groups,optimizer_state=load_latest()
    frozen={n:p.detach().cpu().clone() for n,p in policy.named_parameters() if not p.requires_grad}
    if [(g['name'],len(g['params'])) for g in groups]!=[(g['name'],len(g['params'])) for g in optimizer_state['param_groups']]:
        raise ValueError('Optimizer parameter groups differ from the declared initializer')
    rates=[g['lr'] for g in groups];optimizer=torch.optim.AdamW(groups,weight_decay=0.);optimizer.load_state_dict(optimizer_state)
    if {float(s['step']) for s in optimizer.state.values()}!={3000.}:raise ValueError('Expected restored3000-update AdamW moments')
    hashes.update(snapshot([probe/'result.json',probe/'policy_last.ckpt']))
    output=ROOT/'training/pilot_A';output.mkdir(parents=True,exist_ok=False)
    keeper=ApproachPreservation(dict(policy.config['approach_preservation'],batch_size=4),stats,'cuda')
    loader=DataLoader(dataset,batch_sampler=batches.tolist(),num_workers=2,multiprocessing_context='spawn',pin_memory=True)
    started=time.monotonic()
    for step,(image,current,target,padding,_) in enumerate(loader,1):
        optimizer.zero_grad(set_to_none=True)
        image,current,target,padding=image.cuda(),current.cuda(),target.cuda(),padding.cuda()
        prediction=policy(current,image);base,parts=executed_objective(prediction,target,padding,stats,pose_loss)
        positions,labels=position_pair(prediction[:16],target[:16],pose_loss)
        worst=worst_prefix_position_loss(positions,labels)
        pair_image,pair_current,pair_target,_=choose_pairs(cache,pair_order[step-1])
        pair_prediction=policy(pair_current,pair_image);pair_positions,pair_labels=position_pair(pair_prediction,pair_target,pose_loss)
        response=paired_response_loss(pair_positions.reshape(-1,2,2,3),pair_labels.reshape(-1,2,2,3))
        primary=base+5*response+worst;preservation=keeper.loss(policy,4000+step-1)
        weight,norms=preservation_scale(primary,preservation,policy.base.model.input_proj_robot_state.parameters())
        loss=primary+weight*preservation
        if not torch.isfinite(loss):raise ValueError('Nonfinite paired-response training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*(1-.9*(step-1)/1999)
        optimizer.step()
        if step==1 or step%100==0:
            status=dict(updates=step,limit=2000,primary=float(primary.detach()),paired_response=float(response.detach()),
                worst_prefix=float(worst.detach()),base_terms={k:float(v.detach()) for k,v in parts.items()},
                preservation_weight=weight,weighted_preservation_gradient_fraction=weight*norms['preservation_gradient_norm']/max(norms['task_gradient_norm'],1e-12),
                elapsed_s=time.monotonic()-started)
            (output/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
    if step!=2000:raise ValueError('Exactly2000 candidate updates required')
    assert_frozen(policy,frozen);require_unchanged(hashes)
    checkpoint=output/'policy_last.ckpt'
    torch.save(dict(profile=PROFILE,model=policy.state_dict(),stats=stats,config=policy.config,actual_updates=2000,
        cumulative_lineage_updates=8000,optimizer=optimizer.state_dict(),optimizer_restored=True,
        hardware_deployment_allowed=False),checkpoint)
    policy.requires_grad_(False).eval()
    gate=evaluate_response(policy,stats,manifest,checkpoint_sha256=snapshot([checkpoint])[str(checkpoint.resolve())])
    (output/'response_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    proof=dict(complete=True,actual_updates=2000,diagnostic_updates=200,cumulative_lineage_updates=8000,
        formal_epochs=0,source_sha256=hashes,checkpoint_sha256=snapshot([checkpoint,output/'response_gate.json']),
        frozen_parameters_unchanged=True,optimizer_restored=True,position_response_gate_passed=gate['passed'],
        full_task_passed=False,hardware_commands_sent=False,hardware_deployment_allowed=False)
    (output/'integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(stage='candidate_complete',updates=2000,gate_passed=gate['passed'],
        direction=gate['direction_pass_fraction'],maximum_prefix_error_m=gate['maximum_prefix_position_error_m'])),flush=True)


if __name__=='__main__':main()
