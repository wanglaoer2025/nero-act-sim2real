"""Append one newly verified failure recovery; old66 RGB/labels remain immutable."""
import argparse,json,pickle,shutil
from pathlib import Path
import h5py
from training.history_recoveries import validate_history_manifest,sha
from training.history_data import derive_history_episode,history_statistics
ROOT=Path(__file__).resolve().parents[1]


def validate_parent_extension(base,parent,extended):
    count=base.get('partial_recoveries')
    declaration=extended.get('additional_failure_ids')
    if (base.get('complete') is not True or base.get('full_demonstrations')!=48
            or (count,declaration) not in ((18,[2018]),(19,[2018,2019]))
            or len(base.get('episodes',[]))!=48+count
            or len(parent.get('episodes',[]))!=count
            or extended.get('episodes',[])[:count]!=parent.get('episodes')
            or {r['id'] for r in base['episodes']}!={r['id'] for r in parent['base_episodes']+parent['episodes']}):
        raise ValueError('Immutable original records and exact parent recoveries required')
    new_id=2018 if count==18 else 2019
    extra=[r for r in extended['episodes'] if r['id']==new_id]
    if len(extra)!=1 or len(extended['episodes'])!=count+1:
        raise ValueError('Exactly one declared new failure required')
    return extra


def append_recovery(base_data,recovery_dir,output):
    base_data,recovery_dir,output=map(lambda p:Path(p).resolve(),[base_data,recovery_dir,output])
    extended=json.loads((recovery_dir/'manifest.json').read_text());parent_path=Path(extended['verified_parent_batch'])
    if sha(parent_path)!=extended['parent_manifest_sha256']:raise ValueError('Verified parent manifest changed')
    parent=json.loads(parent_path.read_text());base=json.loads((base_data/'manifest.json').read_text())
    accepted=json.loads((base_data/'data_acceptance.json').read_text())
    if (accepted.get('complete') is not True or accepted['manifest_sha256']!=sha(base_data/'manifest.json')
            or base['recovery_manifest_sha256']!=sha(parent_path)):raise ValueError('Parent visual data acceptance changed')
    additional=validate_parent_extension(base,parent,extended)
    raw_source=ROOT.parent/'nero_act_generalized_pick_place_20261018/training_data/generalized_v1'
    split=json.loads((raw_source/'split.json').read_text())
    declared=json.loads((ROOT/'config/history_experiment.json').read_text())
    groups=validate_history_manifest(extended,split,additional_failure_ids=declared['additional_failure_episode_ids'])
    selected=[dict(r) for r in base['episodes']]
    for r in selected:
        if sha(r['path'])!=r['sha256']:raise ValueError('Original visual HDF changed')
    profile=json.loads((base_data/'calibration_profile.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    for r in additional:
        result=derive_history_episode(r['path'],output/f'episode_{r["id"]}.hdf5',expected_sha=r['sha256'],profile=profile)
        selected.append(dict(r,**result));print(json.dumps(dict(id=r['id'],rendered_frames=result['rendered_frames'],valid_samples=result['valid_samples'])),flush=True)
    stats=history_statistics([Path(r['path']) for r in selected if r['split']=='train'],expected_train_groups=set(groups['train_groups']))
    (output/'dataset_stats.pkl').write_bytes(pickle.dumps(stats))
    for name in ['calibration_profile.json','scene_config.json']:shutil.copy2(base_data/name,output/name)
    (output/'split.json').write_text(json.dumps({s:[r['id'] for r in selected if r['split']==s] for s in ['train','val']},indent=2))
    manifest=dict(base,episodes=selected,groups=groups,partial_recoveries=len(extended["episodes"]),additional_failure_ids=extended["additional_failure_ids"],
        recovery_manifest_sha256=sha(recovery_dir/'manifest.json'),stats_sha256=sha(output/'dataset_stats.pkl'),
        parent_visual_manifest_path=str(base_data/'manifest.json'),parent_visual_manifest_sha256=sha(base_data/'manifest.json'),
        derivation_code_sha256=sha(__file__))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    samples=[];weights=[]
    for row in selected:
        with h5py.File(row['path'],'r') as f:
            for t,v,w in zip(f['history/start_ts'][:],f['history/view_id'][:],f['history/weights'][:]):
                samples.append([row['id'],int(t),int(v),row['split']]);weights.append(int(w))
    (output/'common_samples.json').write_text(json.dumps(dict(samples=samples,weights=weights),separators=(',',':')))
    acceptance=dict(complete=True,training_groups=37,validation_groups=12,full_demonstrations=48,partial_recoveries=len(extended["episodes"]),
        current_sample_count=len(samples),rendered_frame_count=sum(r['rendered_frames'] for r in selected),
        A_B_identical_samples=True,stats_sha256=manifest['stats_sha256'],manifest_sha256=sha(output/'manifest.json'),
        common_samples_sha256=sha(output/'common_samples.json'),hardware_commands_sent=False)
    (output/'data_acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n');return acceptance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base-data','recoveries','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(append_recovery(args.base_data,args.recoveries,args.output)),flush=True)


if __name__=='__main__':main()
