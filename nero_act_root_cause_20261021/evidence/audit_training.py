"""Read-only coverage and label audit of the finalized optimizer pilots."""
from pathlib import Path
import sys,json,pickle
from collections import Counter,defaultdict
import numpy as np
import h5py

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'nero_act_optimizer_fixed_20261021'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from training.history_dataset import HistoryReferenceDataset
from training.integrity import snapshot,require_unchanged
from executor.model import make_policy_observation


def main():
    old_proof=json.loads((BASE/'nero_act_optimizer_control_20261021/evidence/final_verification.json').read_text())
    require_unchanged(old_proof['artifact_sha256'])
    data=(ROOT/'training_data/filtered_v1').resolve();manifest=json.loads((data/'manifest.json').read_text())
    config=json.loads((ROOT/'training/pilot_A/config.json').read_text());stats=pickle.loads((data/'dataset_stats.pkl').read_bytes())
    paths=[Path(r['path']) for r in manifest['episodes'] if r['split']=='train']
    dataset=HistoryReferenceDataset(paths,stats,config)
    order=np.load(data/'batch_order_pilot.npy',allow_pickle=False);counts=Counter(order[order>=0].tolist())
    by_start=defaultdict(list)
    for i,(fi,j) in enumerate(dataset.samples):by_start[fi,int(dataset.cache[fi]['starts'][j])].append(i)
    stages={};groups=defaultdict(Counter);selected=[]
    for fi,path in enumerate(paths):
        with h5py.File(path,'r') as f:
            stages[fi]=f['simulation/stage'].asstr()[:]
            role=str(f.attrs.get('recovery_role','base'))
        starts=sorted(t for f,t in by_start if f==fi)
        for t in starts:
            indices=by_start[fi,t];n=sum(counts[i] for i in indices)
            groups[role][str(stages[fi][t])]+=n
        if int(path.stem.split('_')[-1]) not in [2016,2017,2018,2019]:continue
        chosen=starts[:5]+[t for t in starts if stages[fi][t] in ('close','lift') and t in (10,15,20,30,50,100,120,130)]
        for t in sorted(set(chosen)):
            indices=by_start[fi,t];current=dataset.cache[fi]['observed'][t]
            width=float(current[7]*stats['qpos_std'][7]+stats['qpos_mean'][7]);previous=width+float(current[8])*.01
            target=dataset.cache[fi]['actions'][t:t+2,7]
            selected.append(dict(episode=path.stem,start=t,stage=stages[fi][t],role=role,example_index=indices[0],
                weighted_rows=len(indices),training_visits=sum(counts[i] for i in indices),width_mm=width*1000,previous_mm=previous*1000,
                label_prefix_mm=(target*1000).tolist(),label_prefix_closing_margin_mm=((width-target)*1000).tolist(),
                label_next_command_change_mm=float((target[-1]-previous)*1000)))
    near=[]
    all_rows=[(fi,t,indices[0]) for (fi,t),indices in by_start.items()]
    currents=np.stack([dataset.cache[fi]['observed'][t] for fi,t,_ in all_rows])
    for case in ['nominal_pre_close','nominal_hold_to_lift']:
        records=json.loads((ROOT/f'results/mechanism/{case}/visual_observations.json').read_text())
        for qi in [0,len(records)-1]:
            r=records[qi];features=np.r_[np.array(r['cup_bottom_base_m'])-r['tcp_base_m'],np.array(r['target_base_m'])-r['tcp_base_m']]
            current=make_policy_observation(r['paired_feedback_state'],stats,config,r['causal_previous_gripper_target_m'],geometry_features_m=features)
            index=int(np.argmin(np.linalg.norm(currents-current,axis=1)));fi,t,example=all_rows[index]
            near.append(dict(case=case,query=qi,nearest_episode=paths[fi].stem,nearest_start=t,example_index=example,
                normalized_max_difference=float(abs(currents[index]-current).max()),normalized_L2_difference=float(np.linalg.norm(currents[index]-current)),
                current_width_mm=float(r['paired_feedback_state'][7]*1000),current_previous_mm=float(r['causal_previous_gripper_target_m']*1000),
                nearest_label_prefix_mm=(dataset.cache[fi]['actions'][t:t+2,7]*1000).tolist()))
    assert len(order)==1000 and all(r['training_visits']>0 for r in selected if r['start'] in (5,6,7,8,9))
    inputs=[ROOT/'training/pilot_A/config.json',data/'dataset_stats.pkl',data/'manifest.json',data/'batch_order_pilot.npy',Path(__file__)]+paths
    proof=dict(complete=True,optimizer_updates=0,training_episodes=len(paths),weighted_training_rows=len(dataset),
        ordinary_sample_visits=int((order>=0).sum()),visits_by_recovery_role_and_stage={k:dict(v) for k,v in groups.items()},
        selected_teacher_windows=selected,nearest_training_inputs=near,source_sha256=snapshot(inputs),hardware_commands_sent=False)
    require_unchanged(old_proof['artifact_sha256'])
    (OUT/'training_audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(complete=True,training_episodes=len(paths),ordinary_sample_visits=proof['ordinary_sample_visits'],nearest_training_inputs=near)))


if __name__=='__main__':main()
