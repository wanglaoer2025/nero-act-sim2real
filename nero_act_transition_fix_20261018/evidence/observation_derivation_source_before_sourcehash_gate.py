"""Observation-only derivatives: fresh critical RGB from recorded pre-action physics."""
import argparse,copy,hashlib,json,shutil
from pathlib import Path
import h5py,mujoco,numpy as np
from simulation.scene import build_scene
from simulation.collect import verify_action_contract
from training.transition_sampling import critical_inputs,critical_windows

ROOT=Path(__file__).resolve().parents[1]
IMAGE='observations/images/camera_0'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_identical_physics(source,target):
    with h5py.File(source,'r') as a,h5py.File(target,'r') as b:
        for key,value in a.attrs.items():
            if key not in b.attrs or not np.array_equal(value,b.attrs[key]):
                raise ValueError('Original scene/role attribute changed: '+key)
        names=[]
        a.visititems(lambda name,obj:names.append(name) if isinstance(obj,h5py.Dataset) and name!=IMAGE else None)
        for name in names:
            if name not in b or a[name].dtype!=b[name].dtype or not np.array_equal(a[name][:],b[name][:]):
                raise ValueError('Recorded physical states/actions changed: '+name)
    return True


def require_fresh_inputs(paths):
    starts={i:set() for i in range(len(paths))}
    for index,t in critical_windows(paths,20,command_context=True):starts[index].add(t)
    sources=[]
    for i,path in enumerate(paths):
        with h5py.File(path,'r') as f:
            if f.attrs.get('fresh_critical_rgb_hz')!=20 or 'observations/fresh_rgb_mask' not in f:
                raise ValueError('Dense contact inputs require refreshed current-state RGB')
            mask=f['observations/fresh_rgb_mask'][:]
            if mask.shape!=(len(f['action']),) or mask.dtype!=np.bool_ or not all(mask[t] for t in starts[i]):
                raise ValueError('A sampled input would reuse a stale RGB frame')
            source=Path(f.attrs['observation_source_path']);expected=f.attrs['observation_source_sha256']
        if sha(source)!=expected:raise ValueError('Observation derivative source changed')
        require_identical_physics(source,path);sources.append(source)
    return sources


def derive_episode(source,target):
    source,target=Path(source),Path(target)
    if target.exists():raise FileExistsError(target)
    source_hash=sha(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    with h5py.File(source,'r') as original,h5py.File(target,'r+') as f:
        if original.attrs.get('control_hz')!=20 or original.attrs.get('camera_refresh_hz')!=2:
            raise ValueError('Expected recorded20Hz physics/2Hz RGB source')
        config=json.loads(original.attrs['scene_config']);stages=original['simulation/stage'].asstr()[:]
        indices=sorted({t for t,_,_ in critical_inputs(stages)})
        model=build_scene(config);data=mujoco.MjData(model);anchors=[]
        with mujoco.Renderer(model,height=90,width=160) as renderer:
            for t in sorted(set(indices)|{0}):
                data.qpos[:]=original['simulation/qpos'][t];data.qvel[:]=original['simulation/qvel'][t]
                data.time=float(original['timestamp'][t]);mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera='calibrated_nero_camera')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                rgb=renderer.render().copy()
                if t%10==0:
                    difference=int(np.abs(rgb.astype(int)-original[IMAGE][t].astype(int)).max())
                    if difference:raise ValueError(f'Original render/camera mismatch at{t}: {difference}')
                    anchors.append(t)
                if t in indices:f[IMAGE][t]=rgb
        mask=np.arange(len(stages))%10==0;mask[indices]=True
        f['observations/fresh_rgb_mask']=mask
        f.attrs.update(fresh_critical_rgb_hz=20,observation_source_path=str(source.resolve()),
            observation_source_sha256=source_hash,observation_derivation='recorded_preaction_full_qpos_qvel_no_step')
    require_identical_physics(source,target)
    if sha(source)!=source_hash:raise RuntimeError('Source changed during derivation')
    return dict(source=str(source.resolve()),source_sha256=source_hash,target=str(target.resolve()),
        target_sha256=sha(target),fresh_critical_frames=len(indices),identical_anchor_frames=anchors,
        physical_arrays_and_original_attributes_identical=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    old=ROOT.parent/'nero_act_contact_recovery_20261018'
    sources=[('generalized_v1',ROOT.parent/'nero_act_generalized_pick_place_20261018/training_data/generalized_v1'),
        ('recoveries_v1',old/'training_data/recoveries_v1')]
    reports=[];pending=[]
    for name,directory in sources:
        manifest=json.loads((directory/'manifest.json').read_text());derived=copy.deepcopy(manifest)
        output=args.output/name;output.mkdir()
        for row in derived['episodes']:
            report=derive_episode(row['path'],output/Path(row['path']).name)
            row.update(path=report['target'],sha256=report['target_sha256'],observation_derivation=report)
            verify_action_contract(row['path']);reports.append(report)
            print(json.dumps(dict(dataset=name,id=row['id'],fresh_frames=report['fresh_critical_frames'])),flush=True)
        for file in ['split.json','scene_config.json']:
            if (directory/file).exists():shutil.copy2(directory/file,output/file)
        pending.append((output/'manifest.json',derived))
    require_fresh_inputs([Path(r['target']) for r in reports])
    proof=dict(complete=True,episodes=len(reports),records=reports,
        physical_validation='identical_nonimage_arrays_to_hash_verified_previously_replayed_sources',
        no_new_physics_steps=True,hardware_tested=False)
    (args.output/'observation_derivation.json').write_text(json.dumps(proof,indent=2))
    for path,manifest in pending:path.write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
