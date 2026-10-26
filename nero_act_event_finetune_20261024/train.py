"""One1000-update offline ACT pilot with current-camera approach preservation."""
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import ROOT,CHECKPOINT,CHECKPOINT_SHA,load_initial,historical,base
from preservation import CurrentCameraPreservation
from training.visual_dataset import VisualReferenceDataset,audit_visual_episode
from training.pose_objective import CartesianPoseLoss,fk_spec_from_scene
from executor.model import physical_actions
from executor.ros_execution import RobotGeometry

PROFILE='event_terminated_expert_current_camera_preservation_v1'


def prepare_dataset(policy,stats):
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    selection=json.loads((ROOT/'supervision_audit.json').read_text())
    proof=json.loads((ROOT/'preservation/proof.json').read_text())
    if (not manifest['complete'] or manifest['rejected'] or not selection['complete']
            or selection['retained_ambiguous_close_query_starts']!=0
            or proof['original_query_reproduction_max_abs_rad_m']>1e-6):
        raise ValueError('Expert, supervision and current-camera preservation gates required')
    hashes=dict(manifest['source_sha256']);hashes.update(selection['source_sha256'])
    hashes.update(proof['source_sha256']);hashes.update(proof['replay_sha256'])
    historical.require_unchanged(hashes)
    rows=manifest['episodes'];samples=[]
    pools={f'{kind}:{stage}':[] for kind in ['center','other'] for stage in base.protocol.STAGES}
    for i,row in enumerate(rows):
        if row['split']!='train' or not row['full_task_verified']:raise ValueError('Only complete training experts permitted')
        if audit_visual_episode(row['path'])['split']!='train':raise ValueError('Training split required')
        historical.require_unchanged({row['path']:row['sha256'],row['physical_path']:row['physical_sha256']})
        choice=next(x for x in selection['episodes'] if x['id']==row['id'])
        with h5py.File(row['path']) as f:stages=f['simulation/stage'].asstr()[:]
        for t in choice['training_query_starts']:
            if stages[t] not in base.protocol.STAGES:continue
            key=f'{"center" if row["role"]=="center" else "other"}:{stages[t]}'
            pools[key].append(len(samples));samples.append((i,t))
    dataset=VisualReferenceDataset([Path(r['path']) for r in rows],['camera_0'],stats,
        policy_config=policy.config,image_size=tuple(policy.config['image_size']),sample_starts=samples)
    batches=base.protocol.balanced_batches(pools,steps=1000)
    (ROOT/'sampling_audit.json').write_text(json.dumps(dict(pool_sizes={k:len(v) for k,v in pools.items()},
        samples=len(samples),updates=1000,batch_size=32,validation_rows_in_training=0,
        statistics_recomputed=False,selection='audited_nonconflicting_query_starts'),indent=2)+'\n')
    return dataset,batches,rows,selection,hashes


def lift_mechanism(policy,stats,dataset,rows,selection):
    records=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            t=next(r for r in selection['episodes'] if r['id']==row['id'])['first_lift_index']
            index=dataset.sample_starts.index((i,t));image,current,_,_,_=dataset[index]
            prediction=policy(current[None].cuda(),image[None].cuda())
            action=physical_actions(prediction,stats,bounded_gripper=True)[0,0].cpu().numpy()
            with h5py.File(row['path']) as f:
                scene=json.loads(f.attrs['scene_config'])
                state=np.r_[f['observations/qpos'][t],f['observations/gripper_state'][t]]
            geometry=RobotGeometry(scene)
            dz=float(geometry.pose(action)['tcp'][2]-geometry.pose(state)['tcp'][2]);gap=float(state[7]-action[7])
            records.append(dict(id=row['id'],first_target_tcp_dz_m=dz,gripper_feedback_gap_m=gap,
                lift_and_hold=bool(dz>1e-5 and .003<=gap<=.012)))
    return dict(records=records,passed=sum(r['lift_and_hold'] for r in records),total=len(records),
        training_probe_only=True,full_task_passed=False)


def main():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    output=ROOT/'training/pilot_A';output.mkdir(parents=True,exist_ok=False)
    policy,stats,groups=load_initial();dataset,batches,rows,selection,hashes=prepare_dataset(policy,stats)
    np.save(output/'batch_order.npy',batches,allow_pickle=False)
    scene=json.loads((ROOT/'data/protocol.json').read_text())[0]['scene']
    keeper=CurrentCameraPreservation(ROOT/'preservation/current_camera.npz',stats)
    pose_loss=CartesianPoseLoss(fk_spec_from_scene(scene),stats).cuda()
    before={n:p.detach().cpu().clone() for n,p in policy.state_dict().items()}
    allowed=[n for n,p in policy.named_parameters() if p.requires_grad]
    assert sum(len(g['params']) for g in groups)==len(allowed)
    before_probe=lift_mechanism(policy,stats,dataset,rows,selection)
    (output/'mechanism_before.json').write_text(json.dumps(before_probe,indent=2)+'\n')
    for group in groups:group['lr']*=.25
    rates=[g['lr'] for g in groups]
    optimizer=torch.optim.AdamW(groups,weight_decay=0.)
    files=[*ROOT.glob('*.py'),ROOT/'data/manifest.json',ROOT/'data/protocol.json',ROOT/'supervision_audit.json',
        ROOT/'sampling_audit.json',ROOT/'preservation/proof.json',output/'batch_order.npy',CHECKPOINT,
        ROOT.parent/'docs/superpowers/specs/2026-10-24-nero-act-event-finetune-design.md',
        ROOT.parent/'docs/superpowers/plans/2026-10-24-nero-act-event-finetune.md']
    files+=[Path(m.__file__) for m in tuple(__import__('sys').modules.values()) if getattr(m,'__file__',None)
        and str(m.__file__).startswith(str(ROOT.parent)+'/') and str(m.__file__).endswith('.py')]
    hashes.update(historical.snapshot(files))
    loader=DataLoader(dataset,batch_sampler=batches.tolist(),num_workers=2,multiprocessing_context='spawn',pin_memory=True)
    audits=[];started=time.monotonic()
    for updates,(image,current,target,padding,_) in enumerate(loader,1):
        optimizer.zero_grad(set_to_none=True)
        prediction=policy(current.cuda(),image.cuda())
        task,parts=historical.executed_objective(prediction,target.cuda(),padding.cuda(),stats,pose_loss)
        preservation=keeper.loss(policy,updates-1)
        coefficient,audit=base._scope.preservation_scale(task,preservation,groups)
        loss=task+coefficient*preservation
        if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*(1-.9*(updates-1)/999)
        optimizer.step()
        if updates==1 or updates%100==0:
            status=dict(updates=updates,limit=1000,task=float(task.detach()),preservation=float(preservation.detach()),
                preservation_coefficient=coefficient,terms={k:float(v.detach()) for k,v in parts.items()},elapsed_s=time.monotonic()-started)
            audits.append(dict(update=updates,coefficient=coefficient,**audit))
            (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            (output/'gradient_scope_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
            print(json.dumps(status),flush=True)
    assert updates==1000
    changed=[n for n,p in policy.state_dict().items() if not torch.equal(p.cpu(),before[n])]
    if not set(changed)<=set(allowed):raise ValueError('Frozen parameter or buffer changed')
    historical.require_unchanged(hashes)
    checkpoint=output/'policy_last.ckpt'
    torch.save(dict(profile=PROFILE,model=policy.state_dict(),stats=stats,config=policy.config,
        actual_updates=1000,optimizer=optimizer.state_dict(),hardware_deployment_allowed=False),checkpoint)
    assessment=keeper.assess(policy)
    (output/'preservation.json').write_text(json.dumps(assessment,indent=2)+'\n')
    after_probe=lift_mechanism(policy,stats,dataset,rows,selection)
    (output/'mechanism_after.json').write_text(json.dumps(after_probe,indent=2)+'\n')
    proof=dict(complete=True,actual_updates=1000,initializer_sha256=CHECKPOINT_SHA,
        trainable_parameters=allowed,changed_tensors=changed,frozen_parameters_unchanged=True,
        optimizer_initialization='Fresh AdamW for declared event data, same scopes, quarter prior initial learning rates',
        initial_learning_rates=rates,source_sha256=hashes,checkpoint_sha256=historical.snapshot([checkpoint]),
        interface_and_statistics_unchanged=True,hardware_commands_sent=False,hardware_deployment_allowed=False)
    (output/'integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(training_complete=True,updates=updates,changed_tensors=len(changed),
        lift_probe_before=before_probe['passed'],lift_probe_after=after_probe['passed'],preservation=assessment)),flush=True)


if __name__=='__main__':main()
