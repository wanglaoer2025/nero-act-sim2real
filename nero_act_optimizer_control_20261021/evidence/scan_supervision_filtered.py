"""Zero-update audit of exact and nearby training observations, never held-out labels."""
from pathlib import Path
import sys,json,pickle,hashlib
from collections import defaultdict
import numpy as np
import h5py
from scipy.spatial import cKDTree
BASE=Path(__file__).resolve().parents[2]
SOURCE=BASE/'nero_act_approach_preservation_20261021'
FIXED=BASE/'nero_act_optimizer_fixed_20261021'
sys.path.insert(0,str(FIXED))
from training.history_dataset import HistoryReferenceDataset
from training.integrity import require_unchanged,snapshot
OUT=Path(__file__).resolve().parent

def spans(rows):
    target=np.stack([r['target'] for r in rows])
    return dict(arm_rad=float(np.ptp(target[:,:,:7],axis=0).max()),gripper_m=float(np.ptp(target[:,:,7],axis=0).max()))

def conflicts(values):return values['arm_rad']>.001 or values['gripper_m']>.0005

def main():
    seal=json.loads((SOURCE/'evidence/pre_optimization_freeze.json').read_text());require_unchanged(seal['source_sha256']);require_unchanged(seal['data_sha256'])
    data=SOURCE/'training_data/approach_complete_v1';manifest=json.loads((data/'manifest.json').read_text())
    stats=pickle.loads((data/'dataset_stats.pkl').read_bytes());config=json.loads((SOURCE/'training/pilot_A/config.json').read_text())
    config.update(json.loads((FIXED/'config/history_experiment.json').read_text()))
    paths=[Path(r['path']) for r in manifest['episodes'] if r['split']=='train']
    ds=HistoryReferenceDataset(paths,stats,config,weighted=False);rows=[]
    for fi,j in dict.fromkeys(ds.samples):
        c=ds.cache[fi];start=int(c['starts'][j]);q=c['observed'][start].copy();target=c['actions'][start:start+2]
        if len(target)!=2:continue
        states=q[:8]*stats['qpos_std']+stats['qpos_mean'];previous=float(states[7]+q[8]*.01)
        physical=np.r_[states,previous,q[9:]*stats['geometry_std']+stats['geometry_mean']]
        rows.append(dict(name=f'{paths[fi].stem}:{start}',kind='teacher',q=q,physical=physical,target=target,
                         image_hash=c['image_hashes'][start].decode()))
    teacher_count=len(rows)
    with np.load(SOURCE/'training_inputs/approach_replay.npz',allow_pickle=False) as replay:
        for i,q in enumerate(replay['current']):
            rgb=np.rint(replay['image'][i,0].transpose(1,2,0)*255).astype(np.uint8)
            states=q[:8]*stats['qpos_std']+stats['qpos_mean'];previous=float(states[7]+q[8]*.01)
            physical=np.r_[states,previous,q[9:]*stats['geometry_std']+stats['geometry_mean']]
            target=replay['target'][i,:2]*stats['action_std']+stats['action_mean']
            rows.append(dict(name=f'baseline_replay:{i}',kind='reference',q=q,physical=physical,target=target,
                             image_hash=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()))
    exact=defaultdict(list);images=defaultdict(list)
    for i,row in enumerate(rows):
        exact[(row['image_hash'],row['q'].tobytes())].append(row);images[row['image_hash']].append(i)
    exact_bad=[];cross=[];duplicated=0
    for group in exact.values():
        teachers=[r for r in group if r['kind']=='teacher'];references=[r for r in group if r['kind']=='reference']
        if len(teachers)>1:
            duplicated+=1;spread=spans(teachers)
            if conflicts(spread):exact_bad.append(dict(samples=[r['name'] for r in teachers],**spread))
        if teachers and references:
            spread=spans(group)
            if conflicts(spread):cross.append(dict(samples=[r['name'] for r in group],**spread))
    tolerances=np.array([.0002]*7+[.0005]*2+[.001]*6)
    near_count=near_disagree=0;examples=[]
    for indices in images.values():
        indices=[i for i in indices if rows[i]['kind']=='teacher']
        if len(indices)<2:continue
        points=np.stack([rows[i]['physical'] for i in indices])/tolerances
        pairs=cKDTree(points).query_pairs(1.,p=np.inf,output_type='ndarray');near_count+=len(pairs)
        for a,b in pairs:
            one,two=rows[indices[int(a)]],rows[indices[int(b)]];spread=spans([one,two])
            if conflicts(spread):
                near_disagree+=1
                if len(examples)<20:examples.append(dict(samples=[one['name'],two['name']],**spread))
    inputs=paths+[data/'manifest.json',data/'dataset_stats.pkl',data/'batch_order_pilot.npy',FIXED/'config/history_experiment.json',SOURCE/'training_inputs/approach_replay.npz',Path(__file__)]
    result=dict(complete=True,actual_optimizer_updates=0,train_episodes=len(paths),teacher_unique_inputs=teacher_count,reference_queries=len(rows)-teacher_count,
        exact_teacher_duplicate_groups=duplicated,exact_teacher_conflict_groups=len(exact_bad),exact_teacher_conflicts=exact_bad[:30],
        exact_reference_teacher_disagreement_groups=len(cross),reference_teacher_examples=cross[:20],
        nearby_same_RGB_teacher_pairs=near_count,nearby_target_disagreement_pairs=near_disagree,nearby_examples=examples,
        nearby_tolerances=dict(joint_rad=.0002,gripper_width_and_previous_command_m=.0005,visual_relative_m=.001),
        label_disagreement_threshold=dict(joint_rad=.001,gripper_width_m=.0005),
        near_disagreements_are_not_proven_invalid_labels=True,
        source_sha256=snapshot(inputs),hardware_commands_sent=False)
    (OUT/'supervision_scan_filtered.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','exact_teacher_conflicts','reference_teacher_examples','nearby_examples']}),flush=True)

if __name__=='__main__':main()
