"""Read-only gradient comparison on four fixed training batches and two frozen policies."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np
import torch
from torch.utils.data import default_collate

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from training.history import initialize_history,build_history_policy,_batch_to_device
from training.history_dataset import HistoryReferenceDataset
from training.approach_preservation import ApproachPreservation
from training.optimizer_control import gradient_relation
from training.integrity import snapshot,require_unchanged
from training.pose_objective import fk_spec_from_scene
from training.rebase_initializer import rebase_initializer
from training.group_loss import group_sample_weights
from training.grouped_gripper import gripper_contributions
from executor.model import BoundedActPolicy,physical_actions
from training.history_seal import runtime_sources


def state_hash(policy):
    digest=hashlib.sha256()
    for name,value in policy.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    if not torch.cuda.is_available():raise RuntimeError('Read-only host CUDA diagnostics required')
    data=(ROOT/'training_data/uniform_complete_v1').resolve();manifest=json.loads((data/'manifest.json').read_text())
    init=BASE/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
    failed=BASE/'nero_act_contact_prefix_repair_20261021/training/pilot_A/checkpoints/policy_last.ckpt'
    settings=json.loads((ROOT/'config/history_experiment.json').read_text())
    settings.update(training_budget=dict(kind='updates',limit=1000),num_epochs=0,variant='A',stage='pilot')
    scene=json.loads((data/'scene_config.json').read_text())
    settings.update(scene_config=scene,pose_kinematics_spec=fk_spec_from_scene(scene))
    source=torch.load(init,map_location='cpu',weights_only=False)
    stats=pickle.loads((data/'dataset_stats.pkl').read_bytes());old=pickle.loads((init.parent/'dataset_stats.pkl').read_bytes())
    config,initial=initialize_history(source,old,stats,settings,variant='A')
    dataset=HistoryReferenceDataset([Path(r['path']) for r in manifest['episodes'] if r['split']=='train'],stats,config)
    order=np.load(data/'batch_order_pilot.npy',allow_pickle=False)
    calibrated=np.load(OUT/'group_vectors.npz',allow_pickle=False)
    calibration=json.loads((OUT/'all_group_diagnostics.json').read_text())
    probes=[next(i for i,sample_id in enumerate(dataset.sample_ids) if tuple(sample_id)==tuple(probe))
        for probe in calibration['probe_samples']]
    sources=[Path(__file__),init,failed,init.parent/'dataset_stats.pkl',data/'dataset_stats.pkl',
             data/'manifest.json',data/'batch_order_pilot.npy',data/'batch_order_manifest.json',data/'sampling_resolution.json',
             ROOT/'config/history_experiment.json',ROOT/'training_inputs/approach_replay.npz',ROOT/'training_inputs/approach_replay.json']
    sources += runtime_sources(ROOT)
    sources += [failed.parent/'dataset_stats.pkl',OUT/'qualify_pilot.py',ROOT/'approved_scope.md',
        OUT/'all_group_weights.json',OUT/'all_group_diagnostics.json',OUT/'group_vectors.npz',OUT/'solve_all_group_weights.py',OUT/'readonly_gate_scope.md']
    hashes=snapshot(sources)
    scan=json.loads((data/'sampling_resolution.json').read_text())
    if scan['exact_teacher_conflict_groups_after']!=0:raise ValueError('Resolve exact conflicts before diagnostics')
    records=[];physical_records=[];states=[];assessment=None;max_difference=0.;max_calibration_difference=0.
    failed_payload=torch.load(failed,map_location='cpu',weights_only=False)
    failed_stats=pickle.loads((failed.parent/'dataset_stats.pkl').read_bytes())
    failed_weights=rebase_initializer(failed_payload['model'],failed_stats,stats,config)
    for name,weights in [('zero_update_baseline',initial),('failed_preservation_1000',failed_weights)]:
        policy=build_history_policy(config).cuda();policy.load_state_dict(weights,strict=True);policy.train()
        keeper=ApproachPreservation(config['approach_preservation'],stats,'cuda');before=state_hash(policy)
        if name=='zero_update_baseline':
            assessment=keeper.assess(policy)
            if assessment['maximum_arm_prefix_change_rad']>1e-6 or assessment['maximum_gripper_prefix_change_m']>1e-7:
                raise ValueError('Host replay differs from the zero-update baseline')
        groups=[list(policy.model.action_head.parameters()),list(policy.gripper_adapter.parameters())]
        params=sum(groups,[])
        probe_gradients=[]
        for index in probes:
            sample=dataset[index];q=sample['current'][None].cuda();image=sample['image'][None].cuda()
            prediction=physical_actions(policy(q,image),stats,bounded_gripper=True)[0,:2,7]
            target=sample['actions'][:2,7].cuda()*float(stats['action_std'][7])+float(stats['action_mean'][7])
            probe_gradients.append(torch.cat([v.flatten() for v in torch.autograd.grad((prediction-target).abs().mean(),groups[1])]))
        for ordinal in [0,24,99,499,149,249,749,999,74,349,649,899]:
            torch.manual_seed(42+ordinal)
            batch=default_collate([dataset[int(i)] for i in order[ordinal] if i>=0])
            values,kwargs=_batch_to_device(batch,'cuda',with_history=False)
            losses=policy(*values,**kwargs);keep=keeper.loss(policy,ordinal)
            first=torch.autograd.grad(losses['loss'],params,allow_unused=False)
            second=torch.autograd.grad(keeper.weight*keep,params,allow_unused=False)
            prior=BoundedActPolicy.forward(policy,values[0],values[1],return_unbounded=True)
            terms=gripper_contributions(policy,prior,values[0],values[2],values[3])
            sample_weights=group_sample_weights(kwargs['gripper_groups'],config['gripper_group_loss'])
            expected=torch.autograd.grad((terms*sample_weights).sum(),groups[1],retain_graph=True)
            actual=torch.cat([v.flatten() for v in first[2:]])
            expected=torch.cat([v.flatten() for v in expected])
            difference=float((actual-expected).abs().max());max_difference=max(max_difference,difference)
            if difference>1e-7:raise RuntimeError('Actual grouped loss differs from calibrated counterfactual')
            means=[]
            for group in range(4):
                mask=kwargs['gripper_groups']==group
                if mask.any():
                    gradient=torch.cat([v.flatten() for v in torch.autograd.grad(terms[mask].sum(),groups[1],retain_graph=True)])
                    means.append(gradient*len(mask)/int(mask.sum()))
                else:means.append(torch.zeros_like(actual))
            means=torch.stack(means)
            keep_gradient=torch.cat([v.flatten() for v in second[2:]])
            saved_name='B252' if name=='zero_update_baseline' else 'previous1000'
            saved=int(np.flatnonzero((calibrated['policies']==saved_name)&(calibrated['batches']==ordinal))[0])
            for current,reference in [(means,calibrated['mean'][saved]),(keep_gradient,calibrated['keep'][saved]),
                    (torch.stack(probe_gradients),calibrated['probes'][saved])]:
                delta=float(np.abs(current.detach().cpu().double().numpy()-reference).max())
                max_calibration_difference=max(max_calibration_difference,delta)
                if delta>1e-7:raise RuntimeError('Bound production gradients differ from saved calibration')
            scale=float(means.norm(dim=1)[means.norm(dim=1)>1e-12].mean())
            ratio=float(actual.norm())/scale
            cosines=[gradient_relation(actual+keep_gradient,g)['cosine'] for g in probe_gradients]
            passed=bool(ratio>=1e-3 and all(c is not None and c>=.01 for c in cosines[:2])
                and all(c is None or c>=-1e-5 for c in cosines[2:]))
            physical_records.append(dict(policy=name,batch=ordinal,passed=passed,cosines=cosines,noncollapse_ratio=ratio))
            if not passed:raise RuntimeError('Current production physical direction gate failed')
            offset=0;relations={}
            for label,group in zip(['arm_output','gripper_adapter'],groups):
                a=torch.cat([v.flatten() for v in first[offset:offset+len(group)]])
                b=torch.cat([v.flatten() for v in second[offset:offset+len(group)]])
                relations[label]=gradient_relation(a,b);offset+=len(group)
            total_norm=float(torch.cat([(a+b).flatten() for a,b in zip(first,second)]).norm())
            record=dict(policy=name,batch_ordinal_zero_based=ordinal,ordinary_loss=float(losses['loss'].detach()),
                preservation_loss=float(keep.detach()),gradient_groups=relations,total_gradient_norm=total_norm,
                global_clip_multiplier=min(1.,.1/(total_norm+1e-6)),optimizer_steps=0)
            if ordinal in [0,24,99,499]:records.append(record)
            print(json.dumps(dict(record=record,physical_gate=physical_records[-1])),flush=True)
            del first,second,losses,keep,values,batch
        after=state_hash(policy)
        if before!=after or len(policy.optimizer.state)!=0:raise RuntimeError('Read-only diagnostics mutated model/optimizer')
        states.append(dict(policy=name,before_sha256=before,after_sha256=after,optimizer_state_entries=0))
        del policy,keeper;torch.cuda.empty_cache()
    require_unchanged(hashes)
    proof=dict(schema='optimizer_diagnostics_v1',complete=True,actual_optimizer_updates=0,model_states_unchanged=True,
        exact_teacher_conflict_groups_after_filter=scan['exact_teacher_conflict_groups_after'],
        excluded_ambiguous_windows=settings['excluded_ambiguous_windows'],gradient_records=records,
        initial_replay_assessment=assessment,model_states=states,source_sha256=hashes,hardware_commands_sent=False,
        grouped_loss=dict(settings=settings['gripper_group_loss'],calibration_path=str((OUT/'all_group_weights.json').resolve()),
            actual_loss_verified=True,maximum_gripper_gradient_difference=max_difference,
            physical_directions_revalidated=True,physical_records=physical_records,
            maximum_calibration_gradient_difference=max_calibration_difference))
    (OUT/'optimizer_diagnostics.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(complete=True,gradient_records=len(records),actual_optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
