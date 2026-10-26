"""One predeclared1000-update correction; no hardware or checkpoint selection."""
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from sampling import STAGES,make_batches

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'nero_act_paired_response_20261023'
sys.path.insert(0,str(SOURCE))
from experiment import load_latest,prior,require_unchanged,snapshot
from objective import executed_objective,preservation_scale
from response import evaluate_response
from training.visual_dataset import VisualReferenceDataset,audit_visual_episode
from training.approach_preservation import ApproachPreservation,validate_approach_replay
from training.pose_objective import CartesianPoseLoss,fk_spec_from_scene


def load_initializer():
    checkpoint=SOURCE/'training/pilot_A/policy_last.ckpt'
    proof=json.loads((checkpoint.parent/'integrity.json').read_text())
    require_unchanged(proof['source_sha256']);require_unchanged(proof['checkpoint_sha256'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['profile']!='paired_position_response_v1' or payload['actual_updates']!=2000:
        raise ValueError('Declared latest fixed-last initializer required')
    policy,stats,groups,_=load_latest();policy.load_state_dict(payload['model'],strict=True)
    return policy,stats,groups


def prepare_data(policy,stats):
    old,_,validation,scene,files=prior.prepare_data(policy,stats)
    manifest_path=ROOT/'data/manifest.json';manifest=json.loads(manifest_path.read_text())
    if (not manifest['complete'] or manifest['failed_ACT_actions_used_as_labels'] is not False
            or not any(r['full_task_verified'] for r in manifest['episodes'])):
        raise ValueError('Audited corrections and a complete physical teacher required')
    rows=manifest['episodes'];paths=list(old.paths);new_start=len(paths)
    require_unchanged({r['path']:r['sha256'] for r in rows})
    require_unchanged({r['physical_path']:r['physical_sha256'] for r in rows})
    for row in rows:
        if row['split']!='train' or not row['replay']['passed'] or not row['prefix_contract_audit']['passed']:
            raise ValueError('No validation, failed replay or invalid labels in optimization')
        audit_visual_episode(row['path']);paths.append(Path(row['path']))
    samples=[];pools={'correction':[],**{f'{kind}:{stage}':[] for stage in STAGES for kind in ['new','old']}}
    old_prefix=json.loads((prior.ROOT/'data/manifest.json').read_text())
    prefix_paths={Path(r['path']).resolve() for r in old_prefix['episodes']}
    # Prior datasets already enforce source-group train/validation separation.
    for i,t in old.sample_starts:
        with h5py.File(paths[i]) as f:
            stage=f['simulation/stage'].asstr()[t]
            if f.attrs['split']!='train':raise ValueError('Retained validation leakage')
        if stage not in STAGES:continue
        if paths[i].resolve() in prefix_paths:continue
        index=len(samples);samples.append((i,t));pools['old:'+stage].append(index)
    for offset,row in enumerate(rows):
        i=new_start+offset
        with h5py.File(paths[i]) as f:
            valid=np.flatnonzero(f['observations/visual_valid'][:]);stages=f['simulation/stage'].asstr()[:]
        for t in valid:
            if row['role']!='correction' and stages[t] not in STAGES:continue
            index=len(samples);samples.append((i,int(t)))
            pools['correction' if row['role']=='correction' else 'new:'+stages[t]].append(index)
    dataset=VisualReferenceDataset(paths,['camera_0'],stats,policy_config=policy.config,
        image_size=tuple(policy.config['image_size']),sample_starts=samples)
    batches=make_batches(pools,steps=1000,seed=231019)
    (ROOT/'sampling_audit.json').write_text(json.dumps(dict(pool_sizes={k:len(v) for k,v in pools.items()},
        batch_size=32,updates=1000,validation_rows_in_training=0,paths=[str(p) for p in paths]),indent=2)+'\n')
    files+=[manifest_path,ROOT/'data/protocol.json',ROOT/'current_scene.json']
    files+=[Path(r[k]) for r in rows for k in ['path','physical_path']]
    return dataset,batches,validation,scene,files


def main():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    output=ROOT/'training/pilot_A';output.mkdir(parents=True,exist_ok=False)
    policy,stats,groups=load_initializer();dataset,batches,manifest,scene,files=prepare_data(policy,stats)
    np.save(output/'batch_order.npy',batches,allow_pickle=False)
    files+=list(ROOT.glob('*.py'))+[ROOT/'repair_plan.json',output/'batch_order.npy',
        SOURCE/'training/pilot_A/policy_last.ckpt',SOURCE/'training/pilot_A/integrity.json']
    files += [Path(m.__file__) for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
        and str(m.__file__).startswith(str(ROOT.parent)+'/') and str(m.__file__).endswith('.py')]
    hashes=snapshot(files);hashes.update(validate_approach_replay(policy.config['approach_preservation']['replay_path']))
    keeper=ApproachPreservation(dict(policy.config['approach_preservation'],batch_size=4),stats,'cuda')
    pose_loss=CartesianPoseLoss(fk_spec_from_scene(scene),stats).cuda()
    before=evaluate_response(policy,stats,manifest)
    (output/'response_before.json').write_text(json.dumps(before,indent=2)+'\n')
    frozen={n:p.detach().cpu().clone() for n,p in policy.named_parameters() if not p.requires_grad}
    allowed=[n for n,p in policy.named_parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(groups,weight_decay=0.);rates=[g['lr'] for g in groups]
    loader=DataLoader(dataset,batch_sampler=batches.tolist(),num_workers=2,multiprocessing_context='spawn',pin_memory=True)
    started=time.monotonic();updates=0
    for updates,(image,current,target,padding,_) in enumerate(loader,1):
        optimizer.zero_grad(set_to_none=True)
        prediction=policy(current.cuda(),image.cuda())
        task,parts=executed_objective(prediction,target.cuda(),padding.cuda(),stats,pose_loss)
        preservation=keeper.loss(policy,updates-1)
        weight,norms=preservation_scale(task,preservation,policy.base.model.input_proj_robot_state.parameters())
        loss=task+weight*preservation
        if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*(1-.9*(updates-1)/999)
        optimizer.step()
        if updates==1 or updates%100==0:
            status=dict(updates=updates,limit=1000,task=float(task.detach()),preservation=float(preservation.detach()),
                preservation_weight=weight,terms={k:float(v.detach()) for k,v in parts.items()},elapsed_s=time.monotonic()-started)
            (output/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
    if updates!=1000:raise ValueError('Exactly1000 updates required')
    if any(not torch.equal(p.detach().cpu(),frozen[n]) for n,p in policy.named_parameters() if n in frozen):
        raise ValueError('Frozen visual/CVAE parameters changed')
    require_unchanged(hashes)
    checkpoint=output/'policy_last.ckpt'
    torch.save(dict(profile='full_flow_recovery_v1',model=policy.state_dict(),config=policy.config,stats=stats,
        actual_updates=1000,optimizer=optimizer.state_dict(),hardware_deployment_allowed=False),checkpoint)
    policy.requires_grad_(False).eval()
    after=evaluate_response(policy,stats,manifest,checkpoint_sha256=snapshot([checkpoint])[str(checkpoint.resolve())])
    (output/'response_gate.json').write_text(json.dumps(after,indent=2)+'\n')
    proof=dict(complete=True,actual_updates=1000,formal_epochs=0,optimizer_initialization='Fresh AdamW, new recovery data/objective declared before training',
        source_sha256=hashes,checkpoint_sha256=snapshot([checkpoint,output/'response_gate.json']),
        trainable_parameters=allowed,frozen_parameters_unchanged=True,position_response_gate_passed=after['passed'],
        full_task_passed=False,hardware_commands_sent=False,hardware_deployment_allowed=False)
    (output/'integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(training_complete=True,updates=1000,response_passed=after['passed'],
        maximum_response_error_m=after['maximum_prefix_position_error_m'])),flush=True)


if __name__=='__main__':main()
