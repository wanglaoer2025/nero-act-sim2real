"""Derive current RGB/15D observations from the regenerated physical states."""
from pathlib import Path
import sys,json,pickle,shutil,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import h5py

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.history_data import derive_history_episode,history_statistics
from training.history_recoveries import validate_history_manifest,sha
from training.generalized import validate_manifest
from training.legacy_visual_data import training_profile
from training.contact_timebase import verify_uniform_episode
from training.integrity import snapshot,require_unchanged


def derive(arguments):
    row,profile=arguments;output=ROOT/'training_data/uniform_complete_v1'/f'episode_{row["id"]}.hdf5'
    result=derive_history_episode(row['path'],output,expected_sha=row['sha256'],profile=profile)
    verify_uniform_episode(output)
    return dict(row,**result)


def main():
    physical=ROOT/'training_data/uniform_physical_v1';recoveries=ROOT/'training_data/uniform_recoveries_v1'
    base=json.loads((physical/'manifest.json').read_text());split=json.loads((physical/'split.json').read_text())
    recovery=json.loads((recoveries/'manifest.json').read_text());validate_manifest(base,split)
    groups=validate_history_manifest(recovery,split,additional_failure_ids=[2018,2019])
    assert [r['sha256'] for r in base['episodes']]==[r['sha256'] for r in recovery['base_episodes']]
    rows=base['episodes']+recovery['episodes'];assert len(rows)==68
    scene=json.loads((physical/'scene_config.json').read_text());profile=training_profile(scene)
    output=ROOT/'training_data/uniform_complete_v1';output.mkdir(exist_ok=False)
    (output/'calibration_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    inputs=[Path(r['path']) for r in rows]+[physical/'manifest.json',recoveries/'manifest.json',Path(__file__),ROOT/'training/history_data.py',ROOT/'training/contact_sampling.py',ROOT/'training/transition_sampling.py']
    protected=snapshot(inputs)
    state=dict(schema='history_complete_v1',complete=False,episodes=[],groups=groups,calibration_sha256=profile['sha256'],
        physical_timing_profile='uniform_20hz_gripper4_v1',hardware_deployment_allowed=False)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(derive,(row,profile)):row['id'] for row in rows}
        for future in as_completed(futures):
            row=future.result();state['episodes'].append(row)
            (output/'preparation_status.json').write_text(json.dumps(dict(completed=len(state['episodes']),total=68,last_id=row['id']),indent=2)+'\n')
            print(json.dumps(dict(completed=len(state['episodes']),total=68,id=row['id'],rendered_frames=row['rendered_frames'])),flush=True)
    state['episodes'].sort(key=lambda r:r['id'])
    stats=history_statistics([Path(r['path']) for r in state['episodes'] if r['split']=='train'],expected_train_groups=set(groups['train_groups']))
    (output/'dataset_stats.pkl').write_bytes(pickle.dumps(stats));shutil.copy2(physical/'scene_config.json',output/'scene_config.json')
    (output/'split.json').write_text(json.dumps({s:[r['id'] for r in state['episodes'] if r['split']==s] for s in ['train','val']},indent=2)+'\n')
    state.update(complete=True,full_demonstrations=48,partial_recoveries=20,source_manifest_sha256=sha(physical/'manifest.json'),
        recovery_manifest_sha256=sha(recoveries/'manifest.json'),stats_sha256=sha(output/'dataset_stats.pkl'),derivation_code_sha256=sha(ROOT/'training/history_data.py'))
    (output/'manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    samples=[];weights=[]
    for row in state['episodes']:
        with h5py.File(row['path'],'r') as f:
            for t,v,w in zip(f['history/start_ts'][:],f['history/view_id'][:],f['history/weights'][:]):
                samples.append([row['id'],int(t),int(v),row['split']]);weights.append(int(w))
    (output/'common_samples.json').write_text(json.dumps(dict(samples=samples,weights=weights),separators=(',',':')))
    acceptance=dict(complete=True,training_groups=37,validation_groups=12,full_demonstrations=48,partial_recoveries=20,
        current_sample_count=len(samples),rendered_frame_count=sum(r['rendered_frames'] for r in state['episodes']),
        A_B_identical_samples=True,stats_sha256=state['stats_sha256'],manifest_sha256=sha(output/'manifest.json'),
        common_samples_sha256=sha(output/'common_samples.json'),hardware_commands_sent=False)
    (output/'data_acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    require_unchanged(protected);(output/'derivation_inputs.json').write_text(json.dumps(protected,indent=2)+'\n')
    print(json.dumps(acceptance),flush=True)


if __name__=='__main__':main()
