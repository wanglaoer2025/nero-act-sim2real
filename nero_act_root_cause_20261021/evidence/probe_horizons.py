"""Split the existing 20-point loss without updating any weight or optimizer."""
from pathlib import Path
import json,pickle
import numpy as np
import torch
from torch.utils.data import default_collate
from probe_policy import ROOT,OUT,build_history_policy,state_hash,HistoryReferenceDataset,snapshot,require_unchanged,gradient_relation


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    data=(ROOT/'training_data/filtered_v1').resolve();manifest=json.loads((data/'manifest.json').read_text())
    config=json.loads((ROOT/'training/pilot_A/config.json').read_text());stats=pickle.loads((data/'dataset_stats.pkl').read_bytes())
    ds=HistoryReferenceDataset([Path(r['path']) for r in manifest['episodes'] if r['split']=='train'],stats,config)
    audit=json.loads((OUT/'training_audit.json').read_text());require_unchanged(audit['source_sha256'])
    rows=[r for r in audit['selected_teacher_windows'] if r['episode']=='episode_2017' and r['start'] in (5,6,7,8,9)]
    batch=default_collate([ds[r['example_index']] for r in rows]);args=[batch[k].cuda() for k in ['current','image','actions','padding']]
    records=[];sources=[Path(__file__),OUT/'probe_policy.py',OUT/'policy_probe.json',OUT/'training_audit.json'];states=[]
    for arm in ['fixed','decay']:
        checkpoint=ROOT.parent/f'nero_act_optimizer_{arm}_20261021/training/pilot_A/checkpoints/policy_last.ckpt';sources.append(checkpoint)
        policy=build_history_policy(config).cuda().train();policy.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);before=state_hash(policy)
        loss=policy(*args);pred=loss['a_hat'];target=args[2];weights=(~args[3]).float();weights[:,:10]*=5
        error=abs(pred-target)*weights[:,:,None];den=weights.sum()*8
        front=error[:,:2].sum()/den;tail=error[:,2:].sum()/den
        assert torch.allclose(front+tail,loss['l1'],rtol=1e-6,atol=1e-8)
        terms=dict(l1_first2=front,l1_other18=tail,prior_gripper=config['prior_gripper_loss_weight']*loss['prior_gripper'],
                   deployed_first2=config['deployed_prefix_loss_weight']*loss['deployed_prefix'],ordinary_total=loss['loss'])
        params=list(policy.gripper_adapter.parameters());vectors={}
        for name,value in terms.items():vectors[name]=torch.cat([v.flatten() for v in torch.autograd.grad(value,params,retain_graph=True)])
        physical_mean=(pred[:,:2,7]*stats['action_std'][7]+stats['action_mean'][7]).mean()
        output_gradient=torch.cat([v.flatten() for v in torch.autograd.grad(physical_mean,params,retain_graph=True)])
        variants=dict(existing=vectors['ordinary_total'],without_tail=vectors['ordinary_total']-vectors['l1_other18'],
                      first2_only=vectors['deployed_first2'])
        directions={k:float(-torch.dot(v,output_gradient)) for k,v in variants.items()}
        records.append(dict(arm=arm,examples=[r['episode']+':'+str(r['start']) for r in rows],
            teacher_width_mm=((target[0,:,7]*stats['action_std'][7]+stats['action_mean'][7])*1000).detach().cpu().tolist(),
            prediction_width_mm=((pred[0,:,7]*stats['action_std'][7]+stats['action_mean'][7])*1000).detach().cpu().tolist(),
            losses={k:float(v.detach()) for k,v in terms.items()},
            gradient_norms={k:float(v.norm()) for k,v in vectors.items()},
            final_gripper_bias_gradient={k:float(v[-1]) for k,v in vectors.items()},
            first2_vs_other18=gradient_relation(vectors['l1_first2'],vectors['l1_other18']),
            loss_variants_vs_first2={k:gradient_relation(v,vectors['deployed_first2']) for k,v in variants.items()},
            instantaneous_mean_prefix_width_direction=directions,
            derivative_is_not_an_Adam_update_or_rollout=True))
        after=state_hash(policy);assert before==after and not policy.optimizer.state and all(p.grad is None for p in policy.parameters())
        states.append(dict(arm=arm,before_sha256=before,after_sha256=after))
        del policy,loss,pred,terms,vectors;torch.cuda.empty_cache()
    proof=dict(complete=True,actual_optimizer_updates=0,model_states=states,records=records,source_sha256=snapshot(sources),hardware_commands_sent=False)
    require_unchanged(proof['source_sha256']);(OUT/'horizon_probe.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
