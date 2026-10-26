"""Rerender actual source physics for causal50/100ms history views."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import mujoco
import numpy as np

from executor.contract import GRIPPER_CONTEXT
from executor.model import make_policy_observation
from executor.observation_history import HistoryBuffer
from simulation.scene import build_scene
from training.contact_sampling import visual_sampling_windows
from training.legacy_visual_data import NominalImageEstimator


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def causal_previous(actions,times,exposures,initial):
    indices=np.searchsorted(times,exposures,side='left')-1
    values=np.where(indices>=0,actions[np.maximum(indices,0),7],initial)
    return indices,values


def build_history_batch(f,indices,stats,config):
    indices=np.asarray(indices,int)
    times=f['timestamp'][:];commands=f['action_time_monotonic_s'][:]
    state=np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]]
    command_indices,previous=causal_previous(f['action'][:],commands,times,float(f.attrs['initial_previous_gripper_target_m']))
    features=f['observations/visual_relative_m'][:]
    b=HistoryBuffer();b.clear(epoch=0,after_s=float(times[0])-.01)
    group=str(f.attrs['source_group'])
    for i in indices:
        observed=make_policy_observation(state[i],stats,config,float(previous[i]),geometry_features_m=features[i])
        ci=int(command_indices[i]);ack=float(commands[ci]) if ci>=0 else float(times[0]-.001)
        b.append(dict(observation=observed,frame_id=int(i),exposure_s=float(times[i]),feedback_s=float(times[i]),
            admitted_s=float(times[i]),receipt_id=f'{group}:{ci}',receipt_ack_s=ack,epoch=0,
            calibration_sha256=str(f.attrs['calibration_sha256']),camera_sha256='nominal-simulation-camera',
            session_id=group,admission_valid=bool(f['observations/visual_valid'][i]),prefix_complete=True))
    return b.snapshot(observed)


def derive_history_episode(source,target,*,expected_sha,profile,image_size=(180,320)):
    source,target=Path(source),Path(target)
    if sha(source)!=expected_sha:raise ValueError('Original physical data changed')
    if target.exists():raise FileExistsError(target)
    weights=Counter(t for _,t in visual_sampling_windows([source],20))
    views=[(t,view,np.arange(t-4*step,t+1,step)) for t in weights for view,step in enumerate([1,2]) if t>=4*step]
    requested=sorted({int(i) for _,_,indices in views for i in indices})
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    errors=Counter()
    with h5py.File(source,'r') as old,h5py.File(target,'r+') as f:
        scene=json.loads(f.attrs['scene_config']);group=scene['generalization']['source_group'];n=len(f['action'])
        model=build_scene(scene);data=mujoco.MjData(model)
        site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
        estimator=NominalImageEstimator(profile,image_size)
        del f['observations/images/camera_0']
        images=f.create_dataset('observations/images/camera_0',(n,*image_size,3),dtype='u1',chunks=(1,*image_size,3),compression='lzf')
        features=np.full((n,6),np.nan,np.float32);valid=np.zeros(n,bool);times=f['timestamp'][:]
        rgb_hashes=np.full(n,b'',dtype='S64');physics_hashes=np.full(n,b'',dtype='S64')
        with mujoco.Renderer(model,height=image_size[0],width=image_size[1]) as renderer:
            for i in requested:
                data.qpos[:]=f['simulation/qpos'][i];data.qvel[:]=f['simulation/qvel'][i];data.time=float(times[i])
                mujoco.mj_forward(model,data);renderer.update_scene(data,camera='calibrated_nero_camera')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                rgb=renderer.render().copy()
                try:cup,card,_=estimator.measure(rgb)
                except ValueError as exc:errors[str(exc)]+=1;continue
                images[i]=rgb;features[i]=np.r_[cup-data.site_xpos[site],card-data.site_xpos[site]];valid[i]=True
                rgb_hashes[i]=hashlib.sha256(rgb.tobytes()).hexdigest().encode()
                physics_hashes[i]=hashlib.sha256(np.r_[data.qpos,data.qvel,data.time].tobytes()).hexdigest().encode()
        initial=float(scene.get('initial_simulation_snapshot',{}).get('previous_target',[0]*7+[float(f['observations/gripper_state'][0,0])])[7])
        for key,values in [('action_time_monotonic_s',times+.001),('observations/camera_monotonic_s',times),
                ('observations/feedback_monotonic_s',times),('observations/frame_id',np.arange(n)),
                ('observations/visual_relative_m',features),('observations/visual_valid',valid),
                ('observations/visual_error_bound_m',np.where(valid,.03,np.nan))]:
            if key in f:del f[key]
            f[key]=values
        f['observations/rgb_sha256']=rgb_hashes;f['observations/physics_sha256']=physics_hashes
        f.attrs.update(source_group=group,split=scene['generalization']['split'],calibration_sha256=profile['sha256'],
            visual_observation_profile='visual_relative_base_v1',action_semantics='submitted_absolute_target',
            initial_previous_gripper_target_m=initial,observation_code_sha256=sha(__file__),
            observation_derivation='fresh_physics_for_every_history_frame',visual_runtime_qualified=False,
            clock_domain='synthetic_simulation_timeline',hardware_deployment_allowed=False,
            camera_refresh_hz=20,history_observation_hz_views='[20,10]',
            source_path=str(source.resolve()),source_sha256=expected_sha)
        supervised=f['supervision_valid'][:] if 'supervision_valid' in f else np.ones(n,bool)
        views=[(t,v,ix) for t,v,ix in views if supervised[t] and valid[ix].all()]
        stats={k:np.zeros(d,np.float32) if k.endswith('mean') else np.ones(d,np.float32)
               for d,names in [(8,['qpos_mean','qpos_std']),(6,['geometry_mean','geometry_std'])] for k in names}
        config=dict(gripper_command_context=GRIPPER_CONTEXT,state_dim=8,physical_state_dim=8,policy_input_dim=15,
                    visual_observation_profile='visual_relative_base_v1')
        contexts=[]
        for _,_,ix in views:
            batch=build_history_batch(f,ix,stats,config);batch.verify();contexts.append(batch.context_sha256.encode())
        if not views:raise ValueError('No fully valid causal history samples')
        f['history/indices']=np.stack([ix for _,_,ix in views]);f['history/start_ts']=np.array([t for t,_,_ in views])
        f['history/view_id']=np.array([v for _,v,_ in views]);f['history/weights']=np.array([weights[t] for t,_,_ in views])
        f['history/context_sha256']=np.array(contexts,dtype='S64')
        for key in ['action','observations/qpos','observations/gripper_state','simulation/qpos','simulation/qvel']:
            if not np.array_equal(f[key][:],old[key][:]):raise ValueError('Physical labels/states changed')
    return dict(path=str(target.resolve()),sha256=sha(target),valid_samples=len(views),
                rendered_frames=int(valid.sum()),requested_frames=len(requested),rejected_frames=dict(errors),physical_labels_unchanged=True)


def history_statistics(paths,*,expected_train_groups):
    if not paths or not expected_train_groups:raise ValueError('Nonempty verified training groups required')
    states=[];actions=[];features=[];groups=set()
    for path in paths:
        with h5py.File(path,'r') as f:
            group=str(f.attrs['source_group'])
            if f.attrs['split']!='train' or group not in expected_train_groups:raise ValueError('Validation leakage into statistics')
            groups.add(group);mask=f['observations/visual_valid'][:].astype(bool)
            states.append(np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]][mask])
            supervised=f['supervision_valid'][:] if 'supervision_valid' in f else np.ones(len(mask),bool)
            actions.append(f['action'][:][supervised]);features.append(f['observations/visual_relative_m'][:][mask])
    if groups!=set(expected_train_groups):raise ValueError('Missing training parent groups')
    result={}
    for name,parts,floor in [('qpos',states,1e-2),('action',actions,1e-2),('geometry',features,1e-6)]:
        x=np.concatenate(parts).astype(np.float32)
        if not len(x) or not np.isfinite(x).all():raise ValueError('Finite training-only statistics required')
        result[name+'_mean']=x.mean(0,dtype=np.float64).astype(np.float32)
        result[name+'_std']=np.maximum(x.std(0,dtype=np.float64),floor).astype(np.float32)
    return result


def prepare_history_data(source:Path,recoveries:Path,output:Path)->dict:
    import pickle
    from training.history_recoveries import validate_history_manifest
    from training.generalized import validate_manifest
    from training.legacy_visual_data import training_profile
    source,recoveries,output=map(Path,[source,recoveries,output])
    recovery=json.loads((recoveries/'manifest.json').read_text());split=json.loads((source/'split.json').read_text())
    groups=validate_history_manifest(recovery,split)
    base=json.loads((source/'manifest.json').read_text());validate_manifest(base,split)
    if [(r['id'],r['source_group'],r['split']) for r in base['episodes']]!=[(r['id'],r['source_group'],r['split']) for r in recovery['base_episodes']]:
        raise ValueError('Recovery parent manifest differs from physical source')
    scene=json.loads((source/'scene_config.json').read_text());profile=training_profile(scene)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'calibration_profile.json').open('w') as handle:json.dump(profile,handle,indent=2)
    selected=[];status=dict(schema='history_complete_v1',complete=False,episodes=selected,groups=groups,
                           calibration_sha256=profile['sha256'],hardware_deployment_allowed=False)
    (output/'preparation_status.json').write_text(json.dumps(dict(completed=0,total=66),indent=2))
    for row in base['episodes']+recovery['episodes']:
        report=derive_history_episode(row['path'],output/f'episode_{row["id"]}.hdf5',expected_sha=row['sha256'],profile=profile)
        selected.append(dict(row,**report))
        print(json.dumps(dict(completed=len(selected),total=66,id=row['id'],valid_samples=report['valid_samples'],rendered_frames=report['rendered_frames'])),flush=True)
        (output/'preparation_status.json').write_text(json.dumps(dict(completed=len(selected),total=66,last=selected[-1]),indent=2))
    paths=[Path(r['path']) for r in selected if r['split']=='train']
    stats=history_statistics(paths,expected_train_groups=set(groups['train_groups']))
    (output/'dataset_stats.pkl').write_bytes(pickle.dumps(stats))
    shutil.copy2(source/'scene_config.json',output/'scene_config.json')
    partition={s:[r['id'] for r in selected if r['split']==s] for s in ['train','val']}
    (output/'split.json').write_text(json.dumps(partition,indent=2))
    status.update(complete=True,full_demonstrations=48,partial_recoveries=18,
        source_manifest_sha256=sha(source/'manifest.json'),recovery_manifest_sha256=sha(recoveries/'manifest.json'),
        stats_sha256=sha(output/'dataset_stats.pkl'),derivation_code_sha256=sha(__file__))
    (output/'manifest.json').write_text(json.dumps(status,indent=2)+'\n')
    # Freeze the exact common A/B index and batch order before either optimizer.
    sample_ids=[];weights=[]
    for row in selected:
        with h5py.File(row['path'],'r') as f:
            for t,v,w in zip(f['history/start_ts'][:],f['history/view_id'][:],f['history/weights'][:]):
                sample_ids.append([row['id'],int(t),int(v),row['split']]);weights.append(int(w))
    (output/'common_samples.json').write_text(json.dumps(dict(samples=sample_ids,weights=weights),separators=(',',':')))
    acceptance=dict(complete=True,training_groups=37,validation_groups=12,full_demonstrations=48,partial_recoveries=18,
        current_sample_count=len(sample_ids),rendered_frame_count=sum(r['rendered_frames'] for r in selected),
        A_B_identical_samples=True,stats_sha256=status['stats_sha256'],manifest_sha256=sha(output/'manifest.json'),
        common_samples_sha256=sha(output/'common_samples.json'),hardware_commands_sent=False)
    (output/'data_acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    return acceptance


def main()->int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--recoveries',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare_history_data(args.source,args.recoveries,args.output)),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
