"""Isolated saved-input history interventions, never admitted to runtime or qualification."""
import argparse
import json
from pathlib import Path

import numpy as np


def isolated_interventions(observations):
    values=np.asarray(observations,np.float32)
    if values.shape!=(5,15) or not np.isfinite(values).all():raise ValueError('Saved five by15 numeric history required')
    masked=values.copy();masked[:4]=0
    reversed_past=values.copy();reversed_past[:4]=values[:4][::-1]
    return dict(unaltered=values.copy(),past_masked=masked,past_reversed=reversed_past)


def run_interventions(training_run,capture,output):
    import torch
    from executor.history_shadow import load_offline_policy,load_history_capture,sha
    from executor.model import make_policy_observation,physical_actions
    from executor.policy_worker import preprocess_images
    policy=load_offline_policy(training_run)
    if policy.config.get('numeric_history_profile')!='causal_numeric_gru5_v1':raise ValueError('B history pilot required')
    rgb,state,previous,features,batch,bindings=load_history_capture(capture,policy.stats,policy.config)
    observed=make_policy_observation(state,policy.stats,policy.config,previous,geometry_features_m=features)
    records={}
    with torch.inference_mode():
        for name,values in isolated_interventions(batch.observations).items():
            history=dict(observations=torch.from_numpy(values)[None],age_s=torch.from_numpy(batch.age_s.copy())[None],
                valid=torch.from_numpy(batch.valid.copy())[None])
            normalized=policy.model(torch.from_numpy(observed)[None],
                preprocess_images(rgb[None],policy.config['image_size'])[None],history=history)
            records[name]=physical_actions(normalized[0],policy.stats,bounded_gripper=True).cpu().numpy()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'interventions.npz',**records)
    report=dict(bindings,checkpoint_sha256=policy.metadata['checkpoint_sha256'],
        source_seal_sha256=sha(Path(__file__).resolve().parents[1]/'evidence/pre_optimization_freeze.json'),
        source_sha256=sha(__file__),mode='isolated_invalid_past_diagnostic_only',
        admitted_to_runtime=False,complete_task_qualification=False,motion_sent=False,gripper_sent=False,
        real_hardware_tested=False,raw_sha256=sha(output/'interventions.npz'),
        changes={k:dict(max_joint_delta_rad=float(np.max(np.abs(v[:,:7]-records['unaltered'][:,:7]))),
            max_gripper_delta_m=float(np.max(np.abs(v[:,7]-records['unaltered'][:,7])))) for k,v in records.items()})
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['training-run','capture','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run_interventions(args.training_run,args.capture,args.output)),flush=True)


if __name__=='__main__':main()
