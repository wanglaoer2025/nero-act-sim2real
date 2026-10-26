"""Read-only diagnosis after the bounded pilot failed; never performs optimizer.step."""
from pathlib import Path
import sys,json,pickle
import numpy as np
import torch
from torch.utils.data import default_collate

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evidence'))
from training.history import initialize_history,build_history_policy,_batch_to_device
from training.history_dataset import HistoryReferenceDataset
from training.approach_preservation import ApproachPreservation
from training.optimizer_control import gradient_relation
from training.integrity import snapshot,require_unchanged,require_finalized
from executor.model import physical_actions
from diagnose_gradients import state_hash


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    run=ROOT/'training/pilot_A';checkpoint=run/'checkpoints/policy_last.ckpt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);config=payload['config'];require_finalized(checkpoint,config,allowed_budgets=('updates',))
    data=ROOT/'training_data/uniform_complete_v1';stats=pickle.loads((data/'dataset_stats.pkl').read_bytes())
    manifest=json.loads((data/'manifest.json').read_text());paths=[Path(r['path']) for r in manifest['episodes'] if r['split']=='train']
    dataset=HistoryReferenceDataset(paths,stats,config);order=np.load(data/'batch_order_pilot.npy');counts=np.bincount(order[order>=0],minlength=len(dataset))
    old=json.loads((ROOT.parent/'nero_act_optimizer_decay_20261021/training_data/filtered_v1/manifest.json').read_text());old_paths={r['id']:Path(r['path']) for r in old['episodes']}
    selected=[];coverage=[]
    for eid,t in [(2017,5),(2018,5),(2018,10)]:
        ids=[i for i,(name,start,view) in enumerate(dataset.sample_ids) if name==f'episode_{eid}' and start==t]
        current=dataset[ids[0]];previous=HistoryReferenceDataset([old_paths[eid]],stats,config,weighted=False)
        previous=previous[next(i for i,(_,start,view) in enumerate(previous.sample_ids) if start==t)]
        assert torch.equal(current['image'],previous['image']) and torch.equal(current['current'],previous['current']) and torch.equal(current['actions'][:2],previous['actions'][:2])
        selected.append(ids[0]);coverage.append(dict(sample=f'episode_{eid}:{t}',batch_visits=int(counts[ids].sum()),weighted_views=len(ids),RGB_equal=True,current15D_equal=True,first2_labels_equal=True))
    init=ROOT.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
    source=torch.load(init,map_location='cpu',weights_only=False);old_stats=pickle.loads((init.parent/'dataset_stats.pkl').read_bytes())
    _,initial=initialize_history(source,old_stats,stats,config,variant='A')
    sources=[Path(__file__),checkpoint,init,init.parent/'dataset_stats.pkl',data/'manifest.json',data/'dataset_stats.pkl',data/'batch_order_pilot.npy',run/'config.json']
    sources += paths+list(old_paths[e] for e in [2017,2018])
    hashes=snapshot(sources);records=[];predictions=[];states=[]
    for name,weights in [('zero_update',initial),('pilot1000',payload['model'])]:
        policy=build_history_policy(config).cuda();policy.load_state_dict(weights,strict=True);policy.train();before=state_hash(policy)
        keeper=ApproachPreservation(config['approach_preservation'],stats,'cuda');params=list(policy.gripper_adapter.parameters())
        def gradient(value):return torch.cat([g.flatten() for g in torch.autograd.grad(value,params)])
        local,kwargs=_batch_to_device(default_collate([dataset[selected[0]]]),'cuda',with_history=False)
        torch.manual_seed(42);local_gradient=gradient(policy(*local)['loss'])
        # Exercise the production inverse-target branch on the actual failure input.
        future_grad=[]
        for shift in [-10.,10.]:
            target=local[2].clone();target[:,2:,7]+=shift;torch.manual_seed(42)
            future_grad.append(gradient(policy(local[0],local[1],target,local[3])['loss']))
        assert torch.equal(*future_grad)
        policy.eval()
        for i,c in zip(selected,coverage):
            sample=dataset[i]
            with torch.no_grad():pred=physical_actions(policy(sample['current'][None].cuda(),sample['image'][None].cuda()),stats,bounded_gripper=True)[0].cpu().numpy()
            target=sample['actions'].numpy()*stats['action_std']+stats['action_mean']
            predictions.append(dict(policy=name,sample=c['sample'],prediction_prefix_mm=(pred[:2,7]*1000).tolist(),teacher_prefix_mm=(target[:2,7]*1000).tolist(),maximum_prefix_error_mm=float(abs(pred[:2,7]-target[:2,7]).max()*1000)))
        policy.train()
        for ordinal in [0,24,99,499]:
            torch.manual_seed(42+ordinal);values,kwargs=_batch_to_device(default_collate([dataset[int(i)] for i in order[ordinal] if i>=0]),'cuda',with_history=False)
            losses=policy(*values);keep=keeper.loss(policy,ordinal)
            ordinary=gradient(losses['loss']);preservation=gradient(keeper.weight*keep)
            records.append(dict(policy=name,batch_ordinal_zero_based=ordinal,ordinary_vs_nominal=gradient_relation(ordinary,local_gradient),
                preservation_vs_nominal=gradient_relation(preservation,local_gradient),total_vs_nominal=gradient_relation(ordinary+preservation,local_gradient),optimizer_updates=0))
        after=state_hash(policy);assert before==after and len(policy.optimizer.state)==0
        states.append(dict(policy=name,before=before,after=after,optimizer_state_entries=0,production_inverse_target_tail_independent=True))
        del policy,keeper;torch.cuda.empty_cache()
    require_unchanged(hashes)
    proof=dict(complete=True,actual_optimizer_updates=0,model_states=states,coverage=coverage,predictions=predictions,gradient_records=records,
        source_sha256=hashes,hardware_commands_sent=False,causation_limit='Four fixed-batch raw gradients are diagnostic; they do not reconstruct the Adam trajectory or prove global representational impossibility')
    (ROOT/'evidence/posthoc_gradients.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
