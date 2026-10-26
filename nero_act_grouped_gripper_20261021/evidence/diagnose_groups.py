"""Zero-update four-group loss comparison, including unchanged rehearsal gradients."""
from pathlib import Path
import sys,json,pickle
import numpy as np
import h5py
import torch
from torch.utils.data import default_collate

ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent/'nero_act_contact_prefix_repair_20261021'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PARENT/'evidence'))
from training.history import initialize_history,build_history_policy,_batch_to_device
from training.history_dataset import HistoryReferenceDataset
from training.history_data import causal_previous
from training.grouped_gripper import GROUP_NAMES,classify_gripper_groups,gripper_contributions,minimum_norm_weights
from training.approach_preservation import ApproachPreservation
from training.optimizer_control import gradient_relation
from training.integrity import snapshot,require_unchanged,require_finalized
from executor.model import BoundedActPolicy,physical_actions
from diagnose_gradients import state_hash


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    checkpoint=PARENT/'training/pilot_A/checkpoints/policy_last.ckpt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);config=payload['config'];require_finalized(checkpoint,config,allowed_budgets=('updates',))
    data=PARENT/'training_data/uniform_complete_v1';stats=pickle.loads((data/'dataset_stats.pkl').read_bytes())
    manifest=json.loads((data/'manifest.json').read_text());paths=[Path(r['path']) for r in manifest['episodes'] if r['split']=='train']
    dataset=HistoryReferenceDataset(paths,stats,config);groups=[];all_groups=[]
    for path in paths:
        with h5py.File(path,'r') as f:
            _,previous=causal_previous(f['action'][:],f['action_time_monotonic_s'][:],f['timestamp'][:],float(f.attrs['initial_previous_gripper_target_m']))
            groups.append(classify_gripper_groups(f['simulation/stage'].asstr()[:],f['observations/gripper_state'][:,0]-previous))
    for fi,j in dataset.samples:all_groups.append(int(groups[fi][dataset.cache[fi]['starts'][j]]))
    all_groups=np.asarray(all_groups);assert np.isin(all_groups,np.arange(4)).all()
    probes=[next(i for i,(n,t,v) in enumerate(dataset.sample_ids) if n=='episode_2018' and t==5),
        next(i for i,(n,t,v) in enumerate(dataset.sample_ids) if n=='episode_2017' and t==5),
        next(i for i,g in enumerate(all_groups) if g==2),
        next(i for i,(n,t,v) in enumerate(dataset.sample_ids) if n=='episode_2017' and all_groups[i]==3)]
    init=ROOT.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
    source=torch.load(init,map_location='cpu',weights_only=False);old=pickle.loads((init.parent/'dataset_stats.pkl').read_bytes())
    _,initial=initialize_history(source,old,stats,config,variant='A')
    order=np.load(data/'batch_order_pilot.npy');ordinals=[0,24,99,499,149,249,749,999]
    sources=[Path(__file__),ROOT/'training/grouped_gripper.py',ROOT/'progress.md',ROOT/'approved_scope.md',checkpoint,init,
        init.parent/'dataset_stats.pkl',data/'manifest.json',data/'dataset_stats.pkl',data/'batch_order_pilot.npy']+paths
    hashes=snapshot(sources);records=[];states=[]
    for name,weights in [('B252',initial),('previous1000',payload['model'])]:
        policy=build_history_policy(config).cuda();policy.load_state_dict(weights,strict=True);policy.train();before=state_hash(policy)
        params=list(policy.gripper_adapter.parameters());keeper=ApproachPreservation(config['approach_preservation'],stats,'cuda')
        def grad(loss,retain=False):return torch.cat([g.flatten() for g in torch.autograd.grad(loss,params,retain_graph=retain)])
        probe_grad=[]
        for i in probes:
            sample=dataset[i];q=sample['current'][None].cuda();image=sample['image'][None].cuda()
            prediction=physical_actions(policy(q,image),stats,bounded_gripper=True)[0,:2,7]
            target=sample['actions'][:2,7].cuda()*float(stats['action_std'][7])+float(stats['action_mean'][7])
            probe_grad.append(grad((prediction-target).abs().mean()).detach())
        for ordinal in ordinals:
            indices=[int(i) for i in order[ordinal] if i>=0];labels=all_groups[indices];torch.manual_seed(42+ordinal)
            values,kwargs=_batch_to_device(default_collate([dataset[i] for i in indices]),'cuda',with_history=False)
            prior=BoundedActPolicy.forward(policy,values[0],values[1],return_unbounded=True)
            terms=gripper_contributions(policy,prior,values[0],values[2],values[3]);group_grad=[];present=[];counts=[]
            for group in range(4):
                mask=labels==group
                if not mask.any():continue
                group_grad.append(grad(terms[torch.as_tensor(mask,device='cuda')].sum(),True).detach())
                present.append(group);counts.append(int(mask.sum()))
            summed=torch.stack(group_grad);mean=summed*torch.tensor([len(indices)/c for c in counts],device='cuda')[:,None]
            norms=mean.norm(dim=1);active=norms>1e-12
            assert active.any();unit=mean[active]/norms[active,None];scale=float(norms[active].mean())
            alpha=minimum_norm_weights(unit.cpu().double().numpy());minimum=unit.T@torch.as_tensor(alpha,dtype=unit.dtype,device='cuda')*scale
            combinations=dict(legacy=summed.sum(0),equal_group=mean.mean(0),norm_equalized=unit.mean(0)*scale,minimum_norm=minimum)
            keep=grad(keeper.weight*keeper.loss(policy,ordinal)).detach()
            candidates={}
            for method,ordinary in combinations.items():
                total=ordinary+keep;rat=float(ordinary.norm())/scale
                cosines=[gradient_relation(total,g)['cosine'] for g in probe_grad]
                passed=bool(rat>=1e-3 and all(c is not None and c>=.01 for c in cosines[:2])
                    and all(c is None or c>=-1e-5 for c in cosines[2:]))
                candidates[method]=dict(passed=passed,cosines=dict(zip(GROUP_NAMES,cosines)),noncollapse_ratio=rat)
            records.append(dict(policy=name,batch_ordinal=ordinal,check_only=ordinal not in [0,24,99,499],groups=present,counts=counts,
                group_mean_norms=norms.cpu().tolist(),group_mean_cosines=(mean@mean.T/(norms[:,None]*norms[None,:]).clamp_min(1e-30)).cpu().tolist(),
                minimum_norm_active_groups=[present[i] for i in np.flatnonzero(active.cpu().numpy())],minimum_norm_alpha=alpha.tolist(),
                preservation_norm=float(keep.norm()),candidates=candidates))
            print(json.dumps(dict(policy=name,batch=ordinal,passed={k:v['passed'] for k,v in candidates.items()})),flush=True)
        after=state_hash(policy);assert before==after and not policy.optimizer.state
        states.append(dict(policy=name,before=before,after=after,optimizer_state_entries=0));del policy,keeper;torch.cuda.empty_cache()
    require_unchanged(hashes)
    passed={m:sum(r['candidates'][m]['passed'] for r in records) for m in ['legacy','equal_group','norm_equalized','minimum_norm']}
    selected=next((m for m in ['equal_group','norm_equalized','minimum_norm'] if passed[m]==len(records)),None)
    proof=dict(complete=True,actual_optimizer_updates=0,models_unchanged=True,model_states=states,
        group_counts=dict(zip(GROUP_NAMES,np.bincount(all_groups,minlength=4).tolist())),
        probe_samples=[dataset.sample_ids[i] for i in probes],records=records,passed_records=passed,total_records=len(records),
        selected_method=selected,mechanism_gate_passed=selected is not None,source_sha256=hashes,hardware_commands_sent=False)
    (ROOT/'evidence/group_diagnostics.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(complete=True,passed_records=passed,total=len(records),selected=selected,optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
