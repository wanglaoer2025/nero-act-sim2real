"""Complete physical simulation demonstrations, same observer as deployment.

No ACT optimizer, ROS, CAN or real hardware. Failing attempts remain on disk.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent


def planned_jobs(base,output,protocol):
    jobs=[]
    for split,repetitions in [('train',10),('dev',5)]:
        for i,offset in enumerate(protocol['positions_mm'][split]):
            for repeat in range(repetitions):
                rng=np.random.default_rng(251000+len(jobs));scene=copy.deepcopy(base)
                scene['pick_xy_m']=(np.asarray(base['pick_xy_m'])+np.asarray(offset)/1000).tolist()
                scene['initial_robot_state']['qpos_rad']=(np.asarray(base['initial_robot_state']['qpos_rad'])+rng.uniform(-.001,.001,7)).tolist()
                scene['action_delay_steps']=int(rng.integers(0,2))
                scene['appearance']['light_scale']=float(base['appearance']['light_scale']*rng.uniform(.95,1.05))
                group=f'{split}_{i:02d}'
                jobs.append(dict(scene=scene,split=split,position_group=group,
                                 output=str(Path(output)/f'{group}_{repeat:02d}.hdf5')))
    jobs.sort(key=lambda job:np.linalg.norm(np.asarray(job['scene']['pick_xy_m'])-base['pick_xy_m']))
    return jobs


def acceptance_record(path,result):
    from data_audit import audit_episode
    if result.get('success') is not True or result.get('full_task_verified') is not True or result.get('error') is not None:
        raise ValueError('Physical teacher did not complete task')
    audit=audit_episode(path)
    if not audit['main15_ready']:raise ValueError('New shared-observer contract failed: '+str(audit['errors']))
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                full_task_verified=True,audit=audit)


def collect_one(job):
    import cv2,mujoco
    cv2.setNumThreads(1)
    sys.path.insert(0,str(ROOT/'vendor'))
    from simulation.teacher import run_teacher
    from observer import UnifiedObserver
    output=Path(job['output']);output.parent.mkdir(parents=True,exist_ok=True)
    scene=job['scene'];started=time.monotonic();observer=None;renderer=None;fk=None;count=0
    f=h5py.File(output,'x')
    maxrows=3900
    datasets={}
    shapes={'action':(8,),'timestamp':(),'action_time_monotonic_s':(),'action_acknowledged':(),
            'observations/qpos':(7,),'observations/gripper_state':(1,),
            'observations/visual_relative_m':(6,),'observations/visual_valid':(),
            'observations/frame_id':(),'observations/camera_monotonic_s':(),
            'observations/feedback_monotonic_s':(),'simulation/qpos':(16,),
            'simulation/qvel':(15,),'simulation/stage':()}
    for name,shape in shapes.items():
        dtype=('S8' if name.endswith('stage') else 'i8' if name.endswith('frame_id') else
               '?' if name.endswith(('valid','acknowledged')) else 'f8')
        datasets[name]=f.create_dataset(name,(maxrows,*shape),maxshape=(None,*shape),dtype=dtype)
    images=f.create_dataset('observations/images/camera_0',(maxrows,720,1280,3),
              maxshape=(None,720,1280,3),dtype='u1',chunks=(1,720,1280,3),compression='lzf')
    policy_images=f.create_dataset('observations/policy_images/camera_0',(maxrows,180,320,3),
              maxshape=(None,180,320,3),dtype='u1',chunks=(1,180,320,3),compression='lzf')
    f.attrs.update(sim=True,synthetic=True,full_task_verified=False,teacher_success=False,
                   same_cycle_observation_before_action=True,observation_schema='nero_act_transfer_v2_15d',
                   action_semantics='simulation_submitted_targets',clock_domain='synthetic_20Hz_exposure_before_command',
                   initial_previous_gripper_target_m=scene['initial_robot_state']['gripper_total_width_m'],
                   split=job['split'],position_group=job['position_group'],scene_config=json.dumps(scene),
                   registration_source='synthetic_static_pre_task_camera_registration',
                   real_geometry_qualified=False,hardware_deployment_allowed=False)
    def capture(model,data,action,stage):
        nonlocal observer,renderer,fk,count
        if count>=maxrows:raise RuntimeError('Teacher exceeded3900-point budget')
        if renderer is None:
            model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
            renderer=mujoco.Renderer(model,height=720,width=1280)
            fk=mujoco.MjData(model)
            site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
            def tcp(q):
                fk.qpos[:7]=q;mujoco.mj_forward(model,fk)
                return fk.site_xpos[site].copy()
            profile=dict(camera=scene['camera_calibration'],table_top_z_m=scene['table_top_z_m'],cup=scene['cup'])
            observer=UnifiedObserver(profile,(720,1280),tcp_from_q=tcp)
            registration=[]
            # Static simulated pre-task registration, not future task frames.
            for _ in range(3):
                renderer.update_scene(data,camera='calibrated_nero_camera')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                registration.append(renderer.render().copy())
            observer.register(registration)
        renderer.update_scene(data,camera='calibrated_nero_camera')
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        rgb=renderer.render().copy()
        state=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
        stamp=float(data.time)
        try:observed=observer.measure(rgb,stamp,count,state)
        except ValueError as exc:raise RuntimeError('Shared observer refused: '+str(exc)) from exc
        values={'action':action,'timestamp':stamp,'action_time_monotonic_s':stamp+.001,
                'action_acknowledged':False,'observations/qpos':state[:7],
                'observations/gripper_state':[state[7]],'observations/visual_relative_m':observed['relative_m'],
                'observations/visual_valid':True,'observations/frame_id':count,
                'observations/camera_monotonic_s':stamp,'observations/feedback_monotonic_s':stamp,
                'simulation/qpos':data.qpos,'simulation/qvel':data.qvel,'simulation/stage':stage.encode()}
        for key,value in values.items():datasets[key][count]=value
        images[count]=rgb
        policy_images[count]=cv2.resize(rgb,(320,180),interpolation=cv2.INTER_LINEAR)
        count+=1
        if count%200==0:
            print(json.dumps(dict(position_group=job['position_group'],frames=count,
                                  elapsed_s=time.monotonic()-started)),flush=True)
    result=None
    try:
        result=run_teacher(scene,capture)
        trace=result.pop('trace')
        if len(trace)>count:raise RuntimeError('Applied trace has more commands than camera rows')
        datasets['action_acknowledged'][:len(trace)]=True
        if result['success'] and result['full_task_verified'] and len(trace)==count:
            f.attrs['full_task_verified']=True;f.attrs['teacher_success']=True
    except Exception as exc:
        result=dict(success=False,full_task_verified=False,error=type(exc).__name__+': '+str(exc))
    finally:
        for ds in [*datasets.values(),images,policy_images]:ds.resize(count,axis=0)
        f.close()
        if renderer is not None:renderer.close()
    result.update(elapsed_s=time.monotonic()-started,frames=count,position_group=job['position_group'],
                  split=job['split'],hardware_commands_sent=False)
    output.with_suffix('.result.json').write_text(json.dumps(result,indent=2)+'\n')
    try:accepted=acceptance_record(output,result)
    except ValueError as exc:return dict(accepted=False,path=str(output),error=str(exc),result=result)
    return dict(accepted=True,**accepted,split=job['split'],position_group=job['position_group'],result=result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--limit',type=int,default=150)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not 1<=args.workers<=4 or not 1<=args.limit<=150:raise ValueError('Bounded workers1..4/attempts1..150 required')
    proof=json.loads((ROOT/'evidence/dense_visibility_9000_fixed/result.json').read_text())
    if not proof['complete_visibility']:raise ValueError('Full route visibility must pass before collection')
    from protocol import plan_positions
    protocol=json.loads((ROOT/'position_protocol.json').read_text())
    if protocol!=plan_positions():raise ValueError('Frozen spatial protocol changed')
    source=ROOT.parent/'nero_act_new_handeye_policy_repair_20261023/data/physical/episode_9000.hdf5'
    with h5py.File(source) as f:base=json.loads(f.attrs['scene_config'])
    args.output.mkdir(parents=True,exist_ok=args.resume)
    # Start with the nominal training position, then move outward. This changes
    # collection order only; frozen groups/targets and all failures are retained.
    jobs=planned_jobs(base,args.output,protocol)
    manifest=dict(schema='nero_act_transfer_v2_qualified_data',complete=False,episodes=[],rejected=[],
                  requested=args.limit,expected_total=150,source_scene=str(source),
                  position_protocol_sha256=hashlib.sha256((ROOT/'position_protocol.json').read_bytes()).hexdigest(),
                  observer_sha256=hashlib.sha256((ROOT/'observer.py').read_bytes()).hexdigest(),
                  training_started=False,hardware_commands_sent=False)
    if args.resume:
        previous=json.loads((args.output/'manifest.json').read_text())
        for key in ['schema','source_scene','position_protocol_sha256','observer_sha256']:
            if previous[key]!=manifest[key]:raise ValueError('Resume binding changed: '+key)
        for row in previous['episodes']:
            if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Accepted source changed before resume')
        manifest=previous
        manifest['requested']=args.limit
    seen={r.get('path') for r in manifest['episodes']+manifest['rejected']}
    requested_jobs=[j for j in jobs[:args.limit] if j['output'] not in seen]
    if any(Path(j['output']).exists() for j in requested_jobs):
        raise ValueError('Unaccounted partial output; audit it before resume, never overwrite/retry')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(collect_one,job) for job in requested_jobs]
        for future in as_completed(futures):
            try:row=future.result()
            except Exception as exc:row=dict(accepted=False,error=type(exc).__name__+': '+str(exc))
            manifest['episodes' if row['accepted'] else 'rejected'].append(row)
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps(dict(finished=len(manifest['episodes'])+len(manifest['rejected']),
                accepted=len(manifest['episodes']),rejected=len(manifest['rejected']),last=row.get('position_group'),
                error=row.get('error'))),flush=True)
    manifest['complete']=len(manifest['episodes'])==150 and not manifest['rejected']
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return 0 if manifest['complete'] else 2


if __name__=='__main__':raise SystemExit(main())
