"""Regenerate the same 48 scene geometries and 20 recoveries with one gripper timebase."""
from pathlib import Path
import sys,json,copy,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import h5py
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulation.collect import collect_episode,verify_replay
from training.generalized import validate_manifest
from training.history_recoveries import prepare_recoveries,validate_history_manifest,sha
from simulation.history_teacher import collect_partial
from training.contact_timebase import unified_scene,verify_uniform_episode
from training.integrity import snapshot,require_unchanged


def generate(row):
    output=ROOT/'training_data/uniform_physical_v1'/f'episode_{row["id"]}.hdf5'
    if output.exists():raise FileExistsError(output)
    with h5py.File(row['path'],'r') as f:original=json.loads(f.attrs['scene_config'])
    result=collect_episode(unified_scene(original),output)
    replay=verify_replay(output);timing=verify_uniform_episode(output)
    assert result['success'] and replay['passed'] and result['prefix_contract_audit']['passed']
    return dict(row,path=str(output.resolve()),sha256=sha(output),frames=result['frames'],replay=replay,
        prefix_contract_audit=result['prefix_contract_audit'],timing_audit=timing,original_source_sha256=row['sha256'])


def main():
    old=ROOT.parent/'nero_act_optimizer_decay_20261021/training_data/recoveries_v5/manifest.json'
    previous=json.loads(old.read_text());original_base=previous['base_episodes']
    output=ROOT/'training_data/uniform_physical_v1';output.mkdir(exist_ok=False)
    source=ROOT.parent/'nero_act_generalized_pick_place_20261018/training_data/generalized_v1'
    split=json.loads((source/'split.json').read_text());assert len(original_base)==48
    inputs=[old,source/'split.json',source/'scene_config.json',Path(__file__)]+[Path(r['path']) for r in original_base]
    protected=snapshot(inputs);state=dict(complete=False,episodes=[],rejected=[],hardware_connected=False,target_complete_episodes=48,
        timing_profile='uniform_20hz_gripper4_v1',source_sha256=protected)
    def save():
        state['episodes'].sort(key=lambda r:r['id']);(output/'manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    (output/'split.json').write_text(json.dumps(split,indent=2)+'\n')
    (output/'scene_config.json').write_text(json.dumps(unified_scene(json.loads((source/'scene_config.json').read_text())),indent=2)+'\n');save()
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(generate,r):r['id'] for r in original_base}
        for future in as_completed(futures):
            try:row=future.result()
            except BaseException as exc:
                state['rejected'].append(dict(id=futures[future],error=str(exc)));save();raise
            state['episodes'].append(row);save();print(json.dumps(dict(stage='physical_full',completed=len(state['episodes']),total=48,id=row['id'])),flush=True)
    require_unchanged(protected);state['complete']=True;save();validate_manifest(state,split)
    torch.cuda.init()
    recoveries=ROOT/'training_data/uniform_recoveries_v1'
    recovered=prepare_recoveries(output,ROOT/'results/original_failure_capture',recoveries)
    for oldrow in previous['episodes']:
        if oldrow['id'] not in (2018,2019):continue
        with h5py.File(oldrow['path'],'r') as f:scene=json.loads(f.attrs['scene_config'])
        context=recoveries/'contexts'/f'additional_{oldrow["id"]}.json'
        context.write_bytes(Path(oldrow['context_path']).read_bytes())
        report=collect_partial(unified_scene(scene),context,oldrow['role'],recoveries/f'episode_{oldrow["id"]}.hdf5',
            parent_group=oldrow['source_group'],split=oldrow['split'])
        recovered['episodes'].append(dict(oldrow,**report,context_path=str(context.resolve()),context_sha256=sha(context)))
        print(json.dumps(dict(stage='physical_recovery',completed=len(recovered['episodes']),total=20,id=oldrow['id'])),flush=True)
    recovered.update(additional_failure_ids=[2018,2019],complete=True,timing_profile='uniform_20hz_gripper4_v1')
    recovered['group_validation']=validate_history_manifest(recovered,split,additional_failure_ids=[2018,2019])
    for row in recovered['episodes']:row['timing_audit']=verify_uniform_episode(row['path'])
    (recoveries/'manifest.json').write_text(json.dumps(recovered,indent=2)+'\n')
    require_unchanged(protected)
    print(json.dumps(dict(complete=True,full_demonstrations=48,partial_recoveries=20,optimizer_updates=0,hardware_commands_sent=False)),flush=True)


if __name__=='__main__':main()
