"""Freeze fresh sample exclusions, boundary batches and reencoded423-frame replay."""
from pathlib import Path
import sys,json,pickle
from collections import defaultdict
import numpy as np
import h5py
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.history import initialize_history,prepare_batch_orders
from training.history_dataset import HistoryReferenceDataset
from training.contact_timebase import reencode_reference
from training.approach_preservation import validate_approach_replay
from training.integrity import snapshot,require_unchanged
from executor.model import make_policy_observation


def conflicts(dataset):
    groups=defaultdict(list)
    for fi,j in dict.fromkeys(dataset.samples):
        cache=dataset.cache[fi];t=int(cache['starts'][j]);target=cache['actions'][t:t+2]
        if len(target)!=2:continue
        key=(cache['image_hashes'][t].tobytes(),cache['observed'][t].tobytes())
        groups[key].append((fi,t,target))
    result=[]
    for group in groups.values():
        if len(group)<2:continue
        target=np.stack([r[2] for r in group]);spread=np.ptp(target,axis=0)
        if spread[:,:7].max()>.001 or spread[:,7].max()>.0005:result.append(group)
    return result


def main():
    data=ROOT/'training_data/uniform_complete_v1';manifest=json.loads((data/'manifest.json').read_text())
    stats=pickle.loads((data/'dataset_stats.pkl').read_bytes());settings=json.loads((ROOT/'config/history_experiment.json').read_text())
    init=ROOT.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
    source=torch.load(init,map_location='cpu',weights_only=False);old=pickle.loads((init.parent/'dataset_stats.pkl').read_bytes())
    settings.update(training_budget=dict(kind='updates',limit=1000),num_epochs=0,variant='A',stage='pilot')
    config,_=initialize_history(source,old,stats,settings,variant='A')
    paths=[Path(r['path']) for r in manifest['episodes'] if r['split']=='train']
    initial=HistoryReferenceDataset(paths,stats,config);bad=conflicts(initial);excluded=set()
    for group in bad:
        removed=[];moving=False
        for fi,t,target in group:
            with h5py.File(paths[fi],'r') as f:stage=f['simulation/stage'].asstr()[t];measured=f['observations/qpos'][t]
            stationary=float(abs(target[:,:7]-measured).max())<1e-6
            if stationary and stage in ('release','close'):
                name=f'{paths[fi].stem}:{t}';excluded.add(name);removed.append(name)
            else:moving=True
        if not removed or not moving:raise ValueError('Unresolved new supervision conflict: '+str([(str(paths[f]),t) for f,t,_ in group]))
    # The exclusions belong to the regenerated sample indices, never old absolute timestamps.
    declared=json.loads((ROOT/'config/history_experiment.json').read_text());declared['excluded_ambiguous_windows']=sorted(excluded)
    (ROOT/'config/history_experiment.json').write_text(json.dumps(declared,indent=2)+'\n');config.update(declared)
    selected=HistoryReferenceDataset(paths,stats,config);assert not conflicts(selected)
    resolution=dict(complete=True,exact_teacher_conflict_groups_before=len(bad),exact_teacher_conflict_groups_after=0,
        excluded_ambiguous_windows=sorted(excluded),weighted_samples_before=len(initial),weighted_samples_after=len(selected),
        unchanged_HDF5_actions=True,hardware_commands_sent=False)
    (data/'sampling_resolution.json').write_text(json.dumps(resolution,indent=2)+'\n')
    order=prepare_batch_orders(data,config)
    previous=ROOT.parent/'nero_act_approach_preservation_20261021/training_inputs/approach_replay.npz'
    old_binding=validate_approach_replay(previous);proof=json.loads(previous.with_suffix('.json').read_text())
    reference=ROOT.parent/'nero_act_restricted_heads_20261020/results/nominal_before/nominal_development'
    records=json.loads((reference/'visual_observations.json').read_text());paired_old=pickle.loads((ROOT.parent/'nero_act_approach_preservation_20261021/training_data/approach_complete_v1/dataset_stats.pkl').read_bytes())
    with np.load(previous,allow_pickle=False) as f:
        expected_current,expected_target=reencode_reference(f['current'],f['target'],paired_old,stats,config);images=f['image'].copy()
    with np.load(reference/'rollout.npz',allow_pickle=False) as trace:
        indices=proof['query_indices'];current=[];target=[]
        for i in indices:
            r=records[i];current.append(make_policy_observation(r['paired_feedback_state'],stats,config,r['causal_previous_gripper_target_m'],geometry_features_m=trace['visual_features_m'][i]))
            target.append((trace['raw_chunks'][i]-stats['action_mean'])/stats['action_std'])
    current=np.asarray(current,np.float32);target=np.asarray(target,np.float32)
    assert len(current)==423 and np.max(abs(current-expected_current))<2e-5 and np.max(abs(target-expected_target))<2e-5
    replay=ROOT/'training_inputs/approach_replay.npz'
    if replay.exists():raise FileExistsError(replay)
    np.savez_compressed(replay,current=current,image=images,target=target)
    sources=[Path(__file__),ROOT/'training/contact_timebase.py',ROOT/'config/history_experiment.json',data/'dataset_stats.pkl',
        previous,previous.with_suffix('.json'),reference/'visual_observations.json',reference/'rollout.npz',
        ROOT.parent/'nero_act_approach_preservation_20261021/training_data/approach_complete_v1/dataset_stats.pkl']
    proof.update(source_sha256=dict(old_binding,**snapshot(sources)),replay_sha256=snapshot([replay])[str(replay.resolve())],
        normalization_reencoded=True,physical_reference_unchanged=True)
    replay.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n');validate_approach_replay(replay)
    require_unchanged(old_binding)
    print(json.dumps(dict(complete=True,resolution=resolution,pilot_batches=order['pilot']['batch_count'],replay_samples=423,optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
