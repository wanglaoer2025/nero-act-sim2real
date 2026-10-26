"""Audited complete demonstrations, bounded retries and resumable immutable files."""
import argparse,hashlib,json,multiprocessing,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from simulation.collect import collect_episode,verify_replay
from simulation.generalization import episode_config

ROOT=Path(__file__).resolve().parents[1]


def collect_one(center,index,output):
    output=Path(output);split='train' if index<36 else 'val';rejected=[]
    for attempt in range(3):
        started=time.monotonic();folder=output/'attempts'/f'scene_{index:02d}_candidate_{attempt}'
        folder.mkdir(parents=True,exist_ok=False);path=folder/f'episode_{index}.hdf5'
        try:
            scene=episode_config(center,index,split,attempt)
            (folder/'scene.json').write_text(json.dumps(scene,indent=2))
            result=collect_episode(scene,path);replay=verify_replay(path)
            if not result['full_task_verified'] or not replay['passed']:raise ValueError('Incomplete physical demonstration')
            canonical=output/f'episode_{index}.hdf5'
            if canonical.exists():raise FileExistsError(canonical)
            shutil.copy2(path,canonical)
            return dict(accepted=dict(id=index,path=str(canonical),status='valid',teacher_success=True,
                full_task_verified=True,split=split,source_group=f'scene_{index}',frames=result['frames'],
                replay=replay,prefix_contract_audit=result['prefix_contract_audit'],
                sha256=hashlib.sha256(canonical.read_bytes()).hexdigest(),
                scenario=scene['generalization'],elapsed_s=time.monotonic()-started),rejected=rejected)
        except Exception as exc:
            failure=dict(index=index,attempt=attempt,split=split,error=f'{type(exc).__name__}: {exc}',
                elapsed_s=time.monotonic()-started)
            (folder/'failure.json').write_text(json.dumps(failure,indent=2));rejected.append(failure)
    return dict(accepted=None,rejected=rejected)


def collect_dataset(output,*,indices=None,workers=2,resume=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=resume)
    center=json.loads((ROOT/'scene/center.json').read_text())
    manifest_path=output/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if resume and manifest_path.exists() else dict(
        complete=False,episodes=[],rejected=[],hardware_connected=False,target_complete_episodes=48)
    if manifest['complete']:raise FileExistsError('Dataset is already finalized')
    for row in manifest['episodes']:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Previously accepted demonstration changed')
    done={r['id'] for r in manifest['episodes']}
    requested=list(range(48)) if indices is None else indices
    (output/'scene_config.json').write_text(json.dumps(center,indent=2))
    (output/'split.json').write_text(json.dumps(dict(train=list(range(36)),val=list(range(36,48))),indent=2))
    def save():
        manifest['episodes'].sort(key=lambda r:r['id'])
        temp=manifest_path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2));temp.replace(manifest_path)
    save()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(collect_one,center,i,output):i for i in requested if i not in done}
        for future in as_completed(futures):
            result=future.result();manifest['rejected'].extend(result['rejected'])
            if result['accepted'] is not None:manifest['episodes'].append(result['accepted'])
            save()
            print(json.dumps(dict(scene=futures[future],accepted=result['accepted'] is not None,
                total_accepted=len(manifest['episodes']),rejected_candidates=len(manifest['rejected']),
                detail=result['accepted'] or result['rejected'])),flush=True)
    manifest['complete']=len(manifest['episodes'])==48
    save()
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices');p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true')
    a=p.parse_args();indices=None if a.indices is None else [int(x) for x in a.indices.split(',')]
    if not 1<=a.workers<=2 or (indices is not None and any(not 0<=i<48 for i in indices)):p.error('Bounded workers and indices required')
    result=collect_dataset(a.output,indices=indices,workers=a.workers,resume=a.resume)
    print(json.dumps(dict(complete=result['complete'],accepted=len(result['episodes']))),flush=True)
    if indices is None and not result['complete']:raise SystemExit(1)


if __name__=='__main__':main()
