"""Reproduce the frozen nominal failure and save complete offline contexts."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
INITIALIZER=ROOT.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'


def verify_trace_reference(reference:Path,candidate:Path)->dict:
    reference,candidate=Path(reference),Path(candidate)
    a,b=[json.loads((p/'result.json').read_text()) for p in [reference,candidate]]
    if any(a.get(k)!=b.get(k) for k in ['active_policy_points','error','success']):
        raise ValueError('Recording changed the original stop decision/budget')
    maximum={}
    with np.load(reference/'rollout.npz') as left,np.load(candidate/'rollout.npz') as right:
        for key in ['raw_chunks','sent_targets']:
            if left[key].shape!=right[key].shape or not np.isfinite(right[key]).all():
                raise ValueError('Recording changed trace shape/finiteness: '+key)
            maximum[key]=float(np.max(np.abs(left[key]-right[key]),initial=0))
            if maximum[key]>1e-7:raise ValueError('Recording changed physical policy trace: '+key)
    return dict(matches_reference=True,active_policy_points=b['active_policy_points'],error=b['error'],
                maximum_physical_trace_difference=maximum,hardware_commands_sent=False)


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True,help='Archived result directory containing result.json/rollout.npz')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from executor.model import BoundedActPolicy,verify_paired_readout_statistics
    from training.integrity import require_finalized
    from simulation.supervised_rollout import rollout
    payload=torch.load(INITIALIZER,map_location='cpu',weights_only=False)
    require_finalized(INITIALIZER,payload['config'])
    if not torch.cuda.is_available():raise RuntimeError('Reference failure used CUDA; matching CUDA runtime required')
    torch.set_num_threads(2);torch.manual_seed(42)
    config=dict(payload['config'],device='cuda',pretrained_backbone=False)
    with (INITIALIZER.parent/'dataset_stats.pkl').open('rb') as handle:stats=pickle.load(handle)
    verify_paired_readout_statistics(config,stats)
    policy=BoundedActPolicy(config).cuda();policy.load_state_dict(payload['model'],strict=True);policy.eval()
    policy.config=config;policy.device=torch.device('cuda')
    scene=json.loads((args.source_run/'scene_configuration.json').read_text())
    source_result=json.loads((args.source_run/'result.json').read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rollout(scene,policy,stats,source_result['scenario'],args.output,capture_control=True)
    result=verify_trace_reference(args.source_run,args.output)
    (args.output/'reference_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
