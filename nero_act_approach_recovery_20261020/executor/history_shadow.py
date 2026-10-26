"""Saved nominal history predictions only; imports no robot or camera interfaces."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np

from executor.observation_history import HistoryBuffer,lookup_receipt,make_history_item
from executor.model import make_policy_observation
from vision.synchronization import CameraFrame


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_history_capture(capture:Path,stats:dict,config:dict):
    capture=Path(capture).resolve();manifest_path=capture/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if (manifest.get('schema')!='numeric_history_saved_observation_v1'
            or manifest.get('source_scope')!='nominal_simulation'
            or manifest.get('calibration_sha256')!=config.get('calibration_sha256')
            or len(manifest.get('frames',[]))!=5):
        raise ValueError('Explicit saved nominal five-frame history and matched calibration required; real shadow remains unqualified')
    buffer=HistoryBuffer();buffer.clear(epoch=manifest['epoch'],after_s=manifest['after_s'])
    for row in manifest['frames']:
        path=(capture/row['rgb_path']).resolve()
        if not path.is_relative_to(capture) or not path.is_file() or sha(path)!=row['rgb_sha256']:
            raise ValueError('Saved history RGB path/checksum changed')
        bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if bgr is None:raise ValueError('Unreadable saved history image')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        frame=CameraFrame(row['frame_id'],rgb,row['exposure_s'],row['received_s'],manifest['camera_metadata_sha256'],True)
        paired=dict(state=row['state'],monotonic_s=row['feedback_s'])
        previous=float(lookup_receipt(manifest['receipts'],row['exposure_s'])['rows'][-1][7])
        features=np.asarray(row['geometry_features_m'],np.float32)
        observed=make_policy_observation(paired['state'],stats,config,previous,geometry_features_m=features)
        packet=dict(frame=frame,paired_feedback=paired,geometry=SimpleNamespace(calibration_sha256=manifest['calibration_sha256']),
            details=dict(session_id=manifest['session_id'],admitted_s=row['admitted_s'],
                admission_valid=row.get('admission_valid',True),prefix_complete=row.get('prefix_complete',True),
                geometry_features_m=features.tolist(),policy_observation=observed.tolist()))
        buffer.append(make_history_item(packet,manifest['receipts'],stats,config,epoch=manifest['epoch']))
    batch=buffer.snapshot(observed);batch.verify()
    bindings=dict(capture_manifest_sha256=sha(manifest_path),history_context_sha256=batch.context_sha256,
        calibration_sha256=manifest['calibration_sha256'],source_scope=manifest['source_scope'],
        saved_rgb_file_sha256=[r['rgb_sha256'] for r in manifest['frames']])
    return rgb,np.asarray(paired['state'],np.float32),previous,features,batch,bindings


def run_saved_shadow(policy,capture:Path,output:Path)->dict:
    rgb,state,previous,features,batch,bindings=load_history_capture(capture,policy.stats,policy.config)
    kwargs=dict(geometry_features_m=features)
    if policy.config.get('numeric_history_profile','none')!='none':kwargs['history']=batch
    raw=np.asarray(policy.predict(rgb,state,previous,**kwargs),float)
    if raw.shape!=(20,8) or not np.isfinite(raw).all():raise ValueError('Invalid saved ACT output')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.save(output/'raw20x8.npy',raw,allow_pickle=False)
    report=dict(bindings,checkpoint_sha256=policy.metadata['checkpoint_sha256'],
        numeric_history_profile=policy.config.get('numeric_history_profile','none'),raw20x8=raw.tolist(),
        raw20x8_sha256=sha(output/'raw20x8.npy'),source_sha256=sha(__file__),
        motion_sent=False,gripper_sent=False,real_hardware_tested=False,live_frontend_qualified=False,
        measured_geometry=False,complete_task_qualification=False,hardware_deployment_allowed=False,
        mode='offline_saved_nominal_history_only')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def capture_from_dataset(episode:Path,sample_index:int,output:Path)->Path:
    """Preserve5 actual rendered RGB/state/feature pairs; include past receipts only."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with h5py.File(episode,'r') as f:
        indices=f['history/indices'][sample_index];times=f['timestamp'][:]
        commands=f['action_time_monotonic_s'][:];actions=f['action'][:]
        physical=np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]]
        scene=json.loads(f.attrs['scene_config']);scope=f'saved:{Path(episode).stem}'
        previous=scene.get('initial_simulation_snapshot',{}).get('previous_target',physical[0].tolist())
        initial=np.asarray(previous,float).copy();initial[7]=float(f.attrs['initial_previous_gripper_target_m'])
        receipts=[dict(receipt_id=f'{scope}:initial',rows=[initial.tolist()],sent_s=float(times[0]-.002),
            ack_s=float(times[0]-.001),accepted=True,kind='submitted_prefix',scope=scope)]
        for i in np.flatnonzero(commands<times[indices[-1]]):
            receipts.append(dict(receipt_id=f'{scope}:{int(i)}',rows=[actions[i].tolist()],
                sent_s=float(commands[i]),ack_s=float(commands[i]),accepted=True,kind='submitted_prefix',scope=scope))
        frames=[]
        for i in indices:
            if not f['observations/visual_valid'][i]:raise ValueError('Cannot capture invalid history')
            rgb=f['observations/images/camera_0'][i];path=output/f'frame_{int(i):05d}.png'
            if not cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise RuntimeError('Cannot preserve RGB')
            frames.append(dict(frame_id=int(i),exposure_s=float(times[i]),feedback_s=float(times[i]),
                received_s=float(times[i]),admitted_s=float(times[i]),state=physical[i].tolist(),
                geometry_features_m=f['observations/visual_relative_m'][i].tolist(),rgb_path=path.name,
                rgb_sha256=sha(path),admission_valid=True,prefix_complete=True))
        manifest=dict(schema='numeric_history_saved_observation_v1',source_scope='nominal_simulation',
            session_id=scope,epoch=0,after_s=float(times[indices[0]]-.001),frames=frames,receipts=receipts,
            calibration_sha256=str(f.attrs['calibration_sha256']),camera_metadata_sha256='saved_verified_nominal_renderer',
            source_hdf5=str(Path(episode).resolve()),source_hdf5_sha256=sha(episode),real_hardware_tested=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return output


def load_offline_policy(directory:Path):
    import pickle
    import torch
    from executor.policy_worker import FrozenAct
    from executor.model import BoundedActPolicy,verify_paired_readout_statistics
    from training.integrity import require_finalized
    directory=Path(directory)
    if (directory/'policy.ckpt').is_file():return FrozenAct(directory,device='cpu')
    checkpoint=directory/'checkpoints/policy_last.ckpt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['config'].get('training_budget')!={'kind':'updates','limit':1000}:
        raise ValueError('Use a formal exported bundle or the explicitly offline finalized1000-update pilot')
    require_finalized(checkpoint,payload['config'],allowed_budgets=('updates',))
    # Reuse the exact inference method; this loader never creates a stream/camera.
    policy=object.__new__(FrozenAct)
    policy.config=dict(payload['config'],device='cpu',pretrained_backbone=False);policy.device='cpu'
    policy.stats=pickle.loads((checkpoint.parent/'dataset_stats.pkl').read_bytes())
    verify_paired_readout_statistics(policy.config,policy.stats)
    torch.set_num_threads(1);policy.model=BoundedActPolicy(policy.config).eval();policy.model.load_state_dict(payload['model'],strict=True)
    policy.metadata=dict(checkpoint_sha256=sha(checkpoint),hardware_deployment_allowed=False,offline_pilot_only=True)
    return policy


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['model','capture','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();report=run_saved_shadow(load_offline_policy(args.model),args.capture,args.output)
    print(json.dumps({k:v for k,v in report.items() if k!='raw20x8'}),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
