"""Frozen-policy label fitting, replay equivalence, input sensitivity and gradient probes."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np
import h5py
import torch
import mujoco
from torch.utils.data import default_collate

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'nero_act_optimizer_fixed_20261021'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from training.history import initialize_history,build_history_policy,_batch_to_device
from training.history_dataset import HistoryReferenceDataset
from training.approach_preservation import ApproachPreservation
from training.optimizer_control import gradient_relation
from training.integrity import snapshot,require_unchanged
from executor.model import make_policy_observation,physical_actions,feedback_readout,feedback_preimage
from simulation.scene import build_scene
from src.act.nero_dataset import preprocess_images


def state_hash(policy):
    digest=hashlib.sha256()
    for name,value in policy.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reconstruct(arm,case,query,config,stats):
    root=BASE/f'nero_act_optimizer_{arm}_20261021';folder=root/'results/mechanism'/case
    scene=json.loads((folder/'scene_configuration.json').read_text());model=build_scene(scene);data=mujoco.MjData(model)
    records=json.loads((folder/'visual_observations.json').read_text());record=records[query]
    snapshots=json.loads((folder/'physical_snapshots.json').read_text());saved=snapshots['snapshots'][query]
    # The nominal exposures in these traces were consumed at the same physics time.
    assert abs(record['exposure_simulation_s']-record['consumed_simulation_s'])<1e-9
    data.qpos[:]=saved['qpos'];data.qvel[:]=saved['qvel'];data.time=record['exposure_simulation_s'];mujoco.mj_forward(model,data)
    with mujoco.Renderer(model,height=180,width=320) as renderer:
        renderer.update_scene(data,camera='calibrated_nero_camera');renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        rgb=renderer.render().copy()
    features=np.r_[np.asarray(record['cup_bottom_base_m'])-record['tcp_base_m'],np.asarray(record['target_base_m'])-record['tcp_base_m']]
    current=make_policy_observation(record['paired_feedback_state'],stats,config,record['causal_previous_gripper_target_m'],geometry_features_m=features)
    with np.load(folder/'rollout.npz',allow_pickle=False) as trace:expected=trace['raw_chunks'][query].copy()
    return dict(name=f'{arm}_{case}_{query}',arm=arm,case=case,query=query,current=current,rgb=rgb,
        image=preprocess_images(rgb[None],[180,320])[None].cuda(),expected=expected,record=record)


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    if not torch.cuda.is_available():raise RuntimeError('Host CUDA read-only policy probes required')
    data=(ROOT/'training_data/filtered_v1').resolve();manifest=json.loads((data/'manifest.json').read_text())
    config=json.loads((ROOT/'training/pilot_A/config.json').read_text());stats=pickle.loads((data/'dataset_stats.pkl').read_bytes())
    paths=[Path(r['path']) for r in manifest['episodes'] if r['split']=='train']
    dataset=HistoryReferenceDataset(paths,stats,config);order=np.load(data/'batch_order_pilot.npy',allow_pickle=False)
    audit=json.loads((OUT/'training_audit.json').read_text());require_unchanged(audit['source_sha256'])
    selected=[r for r in audit['selected_teacher_windows'] if r['episode'] in ('episode_2016','episode_2017','episode_2018')]
    # Include actual held-out starts as controls; they are not used for gradient training batches.
    val_paths=[Path(r['path']) for r in manifest['episodes'] if r['split']=='val']
    val=HistoryReferenceDataset(val_paths,stats,config,weighted=False)
    examples=[(f"{r['episode']}:{r['start']}",dataset[r['example_index']],r) for r in selected]
    for eid in ['episode_2012','episode_2013']:
        seen=set()
        for i,(name,t,view) in enumerate(val.sample_ids):
            if name==eid and t in (5,6,7,8,9,10,20) and t not in seen:
                seen.add(t);examples.append((f'{name}:{t}',val[i],dict(episode=name,start=t,split='val')))
    init=BASE/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
    source=torch.load(init,map_location='cpu',weights_only=False);old=pickle.loads((init.parent/'dataset_stats.pkl').read_bytes())
    _,initial=initialize_history(source,old,stats,config,variant='A')
    checkpoints=dict(restricted=BASE/'nero_act_restricted_heads_20261020/training/pilot_A/checkpoints/policy_last.ckpt',
        old_preservation=BASE/'nero_act_approach_preservation_20261021/training/pilot_A/checkpoints/policy_last.ckpt',
        fixed=ROOT/'training/pilot_A/checkpoints/policy_last.ckpt',
        decay=BASE/'nero_act_optimizer_decay_20261021/training/pilot_A/checkpoints/policy_last.ckpt')
    sources=list(checkpoints.values())+[init,init.parent/'dataset_stats.pkl',Path(__file__),OUT/'training_audit.json']
    sources += [p for folder in ['training','executor','simulation'] for p in (ROOT/folder).rglob('*.py')]
    queries=[]
    for arm in ['fixed','decay']:
        for case in ['nominal_pre_close','nominal_hold_to_lift']:
            folder=BASE/f'nero_act_optimizer_{arm}_20261021/results/mechanism'/case
            records=json.loads((folder/'visual_observations.json').read_text())
            for qi in [0,len(records)-1]:queries.append(reconstruct(arm,case,qi,config,stats))
            sources += [folder/name for name in ['rollout.npz','scene_configuration.json','physical_snapshots.json','visual_observations.json']]
    hashes=snapshot(sources);fit=[];reproduction=[];sensitivities=[];gradients=[];states=[];roundtrip_max=0.
    for name in ['baseline','restricted','old_preservation','fixed','decay']:
        weights=initial if name=='baseline' else torch.load(checkpoints[name],map_location='cpu',weights_only=False)['model']
        if name!='baseline':
            previous_stats=pickle.loads((checkpoints[name].parent/'dataset_stats.pkl').read_bytes())
            assert all(np.array_equal(previous_stats[k],stats[k]) for k in stats)
        policy=build_history_policy(config).cuda().eval();policy.load_state_dict(weights,strict=True);before=state_hash(policy)
        readout_stats={k:getattr(policy,'feedback_'+k) for k in ['qpos_mean','qpos_std','action_mean','action_std']}
        for label,sample,meta in examples:
            q=sample['current'][None].cuda();image=sample['image'][None].cuda();target=sample['actions'][None].cuda()
            with torch.no_grad():
                pred=policy(q,image);physical=physical_actions(pred,stats,bounded_gripper=True)[0]
                truth=physical_actions(target,stats,bounded_gripper=True)[0]
                inverse=feedback_preimage(target,q[:,:8],readout_stats,policy.feedback_lower,policy.feedback_upper,version=policy.feedback_version)
                restored=physical_actions(feedback_readout(inverse,q[:,:8],readout_stats,policy.feedback_lower,policy.feedback_upper,version=policy.feedback_version),stats,bounded_gripper=True)[0]
                error=float(abs(restored[:2]-truth[:2]).max());roundtrip_max=max(roundtrip_max,error)
            actual=float(q[0,7]*stats['qpos_std'][7]+stats['qpos_mean'][7]);previous=actual+float(q[0,8])*.01
            fit.append(dict(policy=name,sample=label,split=meta.get('split','train'),width_mm=actual*1000,previous_mm=previous*1000,
                teacher_prefix_mm=(truth[:2,7]*1000).cpu().tolist(),prediction_prefix_mm=(physical[:2,7]*1000).cpu().tolist(),
                prefix_error_mm=((physical[:2,7]-truth[:2,7])*1000).cpu().tolist(),
                predicted_next_command_change_mm=float((physical[1,7]-previous)*1000),
                prefix_inverse_roundtrip_max_physical_error=error))
        for query in queries:
            with torch.no_grad():prediction=physical_actions(policy(torch.from_numpy(query['current'])[None].cuda(),query['image']),stats,bounded_gripper=True)[0].cpu().numpy()
            difference=prediction-query['expected']
            reproduction.append(dict(policy=name,query=query['name'],maximum_arm_difference_rad=float(abs(difference[:,:7]).max()),
                maximum_gripper_difference_m=float(abs(difference[:,7]).max()),prefix_width_mm=(prediction[:2,7]*1000).tolist()))
            if name==query['arm']:
                assert abs(difference[:,:7]).max()<=1e-6 and abs(difference[:,7]).max()<=1e-7
                # First runtime query versus its paired teacher start is the strongest domain-control comparison.
                if query['query']==0:
                    eid=2016 if query['case']=='nominal_pre_close' else 2017
                    row=next(r for r in selected if r['episode']==f'episode_{eid}' and r['start']==5)
                    sample=dataset[row['example_index']];fi,j=dataset.samples[row['example_index']]
                    with h5py.File(dataset.paths[fi],'r') as f:rgb=f['observations/images/camera_0'][5]
                    reproduction[-1].update(runtime_teacher_RGB_equal=bool(np.array_equal(query['rgb'],rgb)),
                        runtime_teacher_15D_max_difference=float(abs(query['current']-sample['current'].numpy()).max()))
            if name in ('baseline','fixed','decay') and query['arm']=='fixed' and (
                    (query['case']=='nominal_pre_close' and query['query']>0) or (query['case']=='nominal_hold_to_lift' and query['query']==0)):
                current=query['current'].copy();width=float(current[7]*stats['qpos_std'][7]+stats['qpos_mean'][7]);values=[]
                for previous in np.arange(width-.009,width+.00001,.0001):
                    q=current.copy();q[8]=(previous-width)/.01
                    with torch.no_grad():pred=physical_actions(policy(torch.from_numpy(q)[None].cuda(),query['image']),stats,bounded_gripper=True)[0,:2,7].cpu().numpy()
                    values.append(dict(previous_mm=float(previous*1000),next_prefix_mm=(pred*1000).tolist(),next_minus_previous_mm=float((pred[-1]-previous)*1000)))
                sensitivities.append(dict(policy=name,query=query['name'],measured_width_mm=width*1000,only_previous_command_varied=True,values=values))
        if name in ('fixed','decay','baseline'):
            keeper=ApproachPreservation(config['approach_preservation'],stats,'cuda')
            arm=list(policy.model.action_head.parameters());grip=list(policy.gripper_adapter.parameters());params=arm+grip
            dedicated=[r['example_index'] for r in selected if r['episode']=='episode_2017' and r['start'] in (5,6,7,8,9)]
            batches=[('nominal_hold_first5',dedicated,0)]+[(f'ordinary_{i}',[int(x) for x in order[i] if x>=0],i) for i in [0,24,99,499]]
            local_prefix_gradient=None
            for label,indices,step in batches:
                policy.train();torch.manual_seed(42+step);values,kwargs=_batch_to_device(default_collate([dataset[i] for i in indices]),'cuda',with_history=False)
                loss=policy(*values,**kwargs);keep=keeper.loss(policy,step)
                parts=dict(ordinary=loss['loss'],l1=loss['l1'],prior_gripper=config['prior_gripper_loss_weight']*loss['prior_gripper'],
                    prefix=config['deployed_prefix_loss_weight']*loss['deployed_prefix'],preservation=keeper.weight*keep)
                vectors={}
                for label2,value in parts.items():
                    grad=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                    vectors[label2]=[torch.zeros_like(p).flatten() if g is None else g.flatten() for p,g in zip(params,grad)]
                norms={k:dict(arm=float(torch.cat(v[:2]).norm()),gripper=float(torch.cat(v[2:]).norm())) for k,v in vectors.items()}
                if label=='nominal_hold_first5':local_prefix_gradient=torch.cat(vectors['prefix'][2:]).detach().clone()
                ordinary_grip=torch.cat(vectors['ordinary'][2:]);keep_grip=torch.cat(vectors['preservation'][2:])
                full=torch.cat([a+b for a,b in zip(vectors['ordinary'],vectors['preservation'])]);norm=float(full.norm());clip=min(1.,.1/(norm+1e-6))
                gradients.append(dict(policy=name,batch=label,losses={k:float(v.detach()) for k,v in parts.items()},gradient_norms=norms,
                    global_clip_multiplier=clip,gripper_ordinary_norm_after_global_clip=norms['ordinary']['gripper']*clip,
                    gripper_direction=gradient_relation(ordinary_grip,keep_grip),
                    global_vs_nominal_prefix=gradient_relation(ordinary_grip+keep_grip,local_prefix_gradient),
                    gripper_final_bias_gradient={k:float(v[-1][0]) for k,v in vectors.items()}))
                del loss,keep,parts,vectors,values,full
            policy.eval();del keeper
        after=state_hash(policy);assert before==after and not policy.optimizer.state
        assert all(p.grad is None for p in policy.parameters())
        states.append(dict(policy=name,before_sha256=before,after_sha256=after,optimizer_state_entries=0))
        print(json.dumps(dict(policy=name,complete=True,weight_state_unchanged=True)),flush=True)
        del policy;torch.cuda.empty_cache()
    require_unchanged(hashes);require_unchanged(audit['source_sha256'])
    proof=dict(complete=True,actual_optimizer_updates=0,model_states=states,teacher_fitting=fit,
        exact_runtime_reproduction=reproduction,previous_command_sensitivity=sensitivities,gradient_probes=gradients,
        teacher_prefix_inverse_roundtrip_max_physical_error=roundtrip_max,source_sha256=hashes,hardware_commands_sent=False)
    (OUT/'policy_probe.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(complete=True,actual_optimizer_updates=0,teacher_fit_records=len(fit),gradient_records=len(gradients),roundtrip_error=roundtrip_max)),flush=True)


if __name__=='__main__':main()
