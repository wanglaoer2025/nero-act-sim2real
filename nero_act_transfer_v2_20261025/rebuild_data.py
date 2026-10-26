"""Versioned reobservation and bounded collection after a diagnosed observer fix.

Existing physical outcomes/actions/RGB remain immutable. No optimizer or robot.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed,CancelledError
import hashlib,json,sys,time
from pathlib import Path
import h5py,numpy as np
from collect_sim import acceptance_record,collect_one,planned_jobs

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def publish_reobserved(source,target,geometry):
    source=Path(source).resolve();target=Path(target)
    with h5py.File(source) as old,h5py.File(target,'x',libver='latest') as new:
        n=len(old['action'])
        if geometry.shape!=(n,6) or not np.isfinite(geometry).all():raise ValueError('Full current-image geometry required')
        for k,v in old.attrs.items():new.attrs[k]=v
        new.attrs.update(original_episode_path=str(source),original_episode_sha256=sha(source),
                         observer_sha256=sha(ROOT/'observer.py'),reobserved_all_frames=True)
        names=[];old.visititems(lambda name,obj:names.append(name) if isinstance(obj,h5py.Dataset) else None)
        for name in names:
            ds=old[name]
            if name=='observations/visual_relative_m':new[name]=geometry
            elif name.startswith(('observations/images/','observations/policy_images/')):
                layout=h5py.VirtualLayout(ds.shape,dtype=ds.dtype)
                layout[:]=h5py.VirtualSource(str(source),name,shape=ds.shape)
                new.create_virtual_dataset(name,layout)
            else:new.create_dataset(name,data=ds[:])


def reobserve_one(job):
    import mujoco,cv2
    cv2.setNumThreads(1)
    from observer import UnifiedObserver
    sys.path.insert(0,str(ROOT/'vendor'))
    from simulation.scene import build_scene
    source=Path(job['source_row']['path']);output=Path(job['output'])
    if sha(source)!=job['source_row']['sha256']:raise ValueError('Original demonstration changed')
    with h5py.File(source) as f:
        scene=json.loads(f.attrs['scene_config']);model=build_scene(scene);data=mujoco.MjData(model)
        site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
        def tcp(q):
            data.qpos[:7]=q;mujoco.mj_forward(model,data);return data.site_xpos[site].copy()
        profile=dict(camera=scene['camera_calibration'],table_top_z_m=scene['table_top_z_m'],cup=scene['cup'])
        observer=UnifiedObserver(profile,(720,1280),tcp_from_q=tcp)
        observer.register([f['observations/images/camera_0'][0] for _ in range(3)])
        geometry=[];started=time.monotonic()
        for i in range(len(f['action'])):
            state=np.r_[f['observations/qpos'][i],f['observations/gripper_state'][i]]
            result=observer.measure(f['observations/images/camera_0'][i],
                float(f['observations/camera_monotonic_s'][i]),int(f['observations/frame_id'][i]),state)
            geometry.append(result['relative_m'])
            if (i+1)%500==0:
                print(json.dumps(dict(reobserving=source.name,frames=i+1,elapsed_s=time.monotonic()-started)),flush=True)
    publish_reobserved(source,output,np.asarray(geometry))
    result=job['source_row']['result']
    accepted=acceptance_record(output,result)
    return dict(accepted=True,**accepted,result=result,split=job['split'],position_group=job['position_group'],
        reobserved_from=str(source),source_dependencies=[dict(path=str(source),sha256=sha(source))])


def execute_job(job):
    try:return reobserve_one(job) if 'source_row' in job else collect_one(job)
    except Exception as exc:return dict(accepted=False,path=job['output'],error=type(exc).__name__+': '+str(exc))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=150)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not 1<=args.workers<=4 or not 1<=args.limit<=150:raise ValueError('Bounded150 main episodes/four workers required')
    proof=json.loads((ROOT/'evidence/dense_visibility_fragment_repair/result.json').read_text())
    if not proof['complete_visibility']:raise ValueError('Complete fixed-observer route must pass first')
    original=json.loads(args.source_manifest.read_text());protocol=json.loads((ROOT/'position_protocol.json').read_text())
    with h5py.File(original['source_scene']) as f:base=json.loads(f.attrs['scene_config'])
    args.output.mkdir(parents=True,exist_ok=args.resume)
    jobs=planned_jobs(base,args.output,protocol)
    old={Path(r['path']).name:r for r in original['episodes']}
    failed={Path(r['path']).name for r in original['rejected'] if r.get('path')}
    for job in jobs:
        if Path(job['output']).name in old:job['source_row']=old[Path(job['output']).name]
    # Reproduce both diagnosed failures first in the new version, once. Their
    # old failures are retained; this is never a same-version retry loop.
    jobs.sort(key=lambda j:0 if Path(j['output']).name in failed else 1 if 'source_row' in j else 2)
    manifest=dict(schema=original['schema'],complete=False,episodes=[],rejected=[],expected_total=150,
        source_scene=original['source_scene'],source_manifest=str(args.source_manifest.resolve()),
        previous_version_retained=True,observer_sha256=sha(ROOT/'observer.py'),
        position_protocol_sha256=sha(ROOT/'position_protocol.json'),collection_running=True,
        training_started=False,hardware_commands_sent=False)
    if args.resume:
        previous=json.loads((args.output/'manifest.json').read_text())
        for key in ['source_scene','observer_sha256','position_protocol_sha256','source_manifest']:
            if previous[key]!=manifest[key]:raise ValueError('Rebuild binding changed:'+key)
        if previous['rejected']:raise ValueError('No retry after a new-version failure')
        manifest=previous;manifest['collection_running']=True
        for row in manifest['episodes']:
            if sha(row['path'])!=row['sha256']:raise ValueError('Rebuild source changed')
    seen={r['path'] for r in manifest['episodes']}
    todo=[j for j in jobs[:args.limit] if j['output'] not in seen]
    if any(Path(j['output']).exists() for j in todo):raise ValueError('Unaccounted output; no overwrite')
    write(args.output/'manifest.json',manifest)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute_job,j) for j in todo]
        for future in as_completed(futures):
            try:row=future.result()
            except CancelledError:continue
            manifest['episodes' if row['accepted'] else 'rejected'].append(row)
            if not row['accepted']:
                manifest['stopped_on_failure']=True
                for pending in futures:pending.cancel()
            write(args.output/'manifest.json',manifest)
            print(json.dumps(dict(accepted=len(manifest['episodes']),rejected=len(manifest['rejected']),
                last=row.get('position_group'),error=row.get('error'))),flush=True)
    manifest['collection_running']=False
    manifest['complete']=len(manifest['episodes'])==150 and not manifest['rejected']
    write(args.output/'manifest.json',manifest)
    return 0 if manifest['complete'] or args.limit<150 and not manifest['rejected'] else 2


if __name__=='__main__':raise SystemExit(main())
