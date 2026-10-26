"""Infer from saved upright RGB/named feedback; no robot, camera or CAN connection.

python -m src.act.run_act_shadow --checkpoint ... --capture-npz ...
  --safety-summary ... --output ... [--device auto]
This is a recorded-observation audit, never a live control loop.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from src.act.nero_dataset import preprocess_images
from src.act.nero_policy import NeroActPolicy
from src.act.safety_filter import SafetyFilter

JOINT_NAMES=[f'joint{i}' for i in range(1,8)]+['gripper']


def load_capture(path):
    with np.load(path,allow_pickle=False) as capture:
        names=capture['joint_names'].tolist()
        if len(names)!=8 or len(set(names))!=8 or set(names)!=set(JOINT_NAMES):
            raise ValueError('Capture requires joint1..joint7 and gripper exactly once')
        states=np.asarray(capture['state'],dtype=np.float32)
        if states.ndim!=2 or states.shape[1]!=8:raise ValueError('Capture states must be [T,8]')
        states=states[:,[names.index(n) for n in JOINT_NAMES]]
        rgb=capture['rgb'].copy();timestamps=capture['sample_monotonic_s'].copy()
    if states.ndim!=2 or states.shape[1]!=8 or not np.isfinite(states).all():
        raise ValueError('Capture states must be finite [T,8]')
    if rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[-1]!=3 or len(rgb)!=len(states):
        raise ValueError('Capture images must be uint8 upright RGB [T,H,W,3]')
    if not len(states) or timestamps.shape!=(len(states),) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps)<=0):
        raise ValueError('Capture timestamps must be finite and strictly increasing')
    return states,rgb,timestamps


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--stats',type=Path)
    parser.add_argument('--capture-npz',type=Path,required=True)
    parser.add_argument('--safety-summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--sample-stride',type=int,default=10)
    parser.add_argument('--task-coordinates-mm',type=float,nargs=6,
                        help='Measured base XYZ of source and destination cup centres, for a 14-D model only')
    args=parser.parse_args()
    if args.sample_stride<1:parser.error('sample-stride must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    device_name=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device_name=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
    device=torch.device(device_name);torch.set_num_threads(1)
    bundle=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=dict(bundle['config'])
    expected_state=14 if config.get('coordinate_conditioning',False) else 8
    if config['state_dim']!=expected_state or config['action_dim']!=8 or config['camera_names']!=['camera_0'] or config.get('action_representation','absolute_joint_position')!='absolute_joint_position':
        raise ValueError('Shadow audit requires single-camera, 8-D actions and explicit 8-D/14-D state')
    task_coordinates=None
    if expected_state==14:
        if args.task_coordinates_mm is None:raise ValueError('14-D policy requires measured --task-coordinates-mm; simulator coordinates are not a real substitute')
        task_coordinates=np.asarray(args.task_coordinates_mm,dtype=np.float32)/1000
        if not np.isfinite(task_coordinates).all():raise ValueError('Task coordinates must be finite base XYZ')
    elif args.task_coordinates_mm is not None:raise ValueError('8-D policy does not accept task coordinates')
    config.update(device=device_name,pretrained_backbone=False)
    policy=NeroActPolicy(config).to(device);policy.load_state_dict(bundle['model'],strict=True);policy.eval()
    stats_path=args.stats or args.checkpoint.parent/'dataset_stats.pkl'
    with stats_path.open('rb') as stream:stats=pickle.load(stream)
    for name in ['qpos_mean','qpos_std','action_mean','action_std']:
        stats[name]=np.asarray(stats[name],dtype=np.float32)
        dimension=expected_state if name.startswith('qpos_') else 8
        if stats[name].shape!=(dimension,) or not np.isfinite(stats[name]).all() or (name.endswith('std') and np.any(stats[name]<=0)):
            raise ValueError(f'Invalid normalization: {name}')
    summary=json.loads(args.safety_summary.read_text())
    safety=SafetyFilter(summary['settings'],summary['train_action_min'],summary['train_action_max'],
                        summary['train_qpos_min'],summary['train_qpos_max'])
    states,rgb,timestamps=load_capture(args.capture_npz)
    initial_error=None
    try:safety.validate_initial_state(states[0],margin=.15)
    except ValueError as exc:initial_error=str(exc)
    rows=[];predictions=[]
    for index in range(0,len(states),args.sample_stride):
        started=time.monotonic()
        image=preprocess_images(rgb[index:index+1],config['image_size'])[None].to(device)
        model_state=states[index] if task_coordinates is None else np.r_[states[index],task_coordinates]
        state=torch.from_numpy((model_state-stats['qpos_mean'])/stats['qpos_std'])[None].to(device)
        with torch.inference_mode():prediction=policy(state,image)[0]
        if device.type=='cuda':torch.cuda.synchronize(device)
        prediction=prediction.cpu().numpy()*stats['action_std']+stats['action_mean']
        if prediction.shape!=(config['chunk_size'],8) or not np.isfinite(prediction).all():raise ValueError('Invalid predicted chunk')
        predictions.append(prediction)
        safe=None;safety_error=None;info=None
        try:safe,info=safety.filter(prediction[0],states[index])
        except ValueError as exc:safety_error=str(exc)
        rows.append({'sample_index':index,'recorded_timestamp_s':float(timestamps[index]),
                     'inference_and_preprocessing_ms':1000*(time.monotonic()-started),
                     'state':states[index].tolist(),'normalized_state':state.cpu().numpy()[0].tolist(),
                     'first_predicted_action':prediction[0].tolist(),'preview_safe_action':None if safe is None else safe.tolist(),
                     'safety_info':info,'safety_error':safety_error})
    np.savez_compressed(args.output/'predictions.npz',chunks=np.stack(predictions),sample_indices=[r['sample_index'] for r in rows])
    report={'mode':'recorded_observation_shadow','checkpoint_epoch':bundle.get('epoch'),
            'state_dim':expected_state,'task_coordinates_base_m':None if task_coordinates is None else task_coordinates.tolist(),
            'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'device':device_name,'capture_samples':len(states),'inferred_observations':len(rows),
            'initial_state_error':initial_error,'finite_predictions':True,
            'safety_errors':sum(r['safety_error'] is not None for r in rows),
            'prediction_ms_p95':float(np.percentile([r['inference_and_preprocessing_ms'] for r in rows],95)),
            'image_orientation':'already upright RGB; no additional rotation',
            'camera_connected':False,'robot_sdk_connected':False,'motion_sent':False,
            'motion_permitted':False,'real_transfer_verified':False,
            'scope':'saved input compatibility and action preview; no closed-loop or live scheduling validation'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'trace.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return int(initial_error is not None or report['safety_errors']>0)


if __name__=='__main__':raise SystemExit(main())
