"""Fixed-update ACT mechanism checks;20mm lift is never complete-task qualification."""
import argparse
import copy
import json
from pathlib import Path
import pickle

import h5py
import torch

from simulation.visual_evaluation import remaining_flow_passed

ROOT=Path(__file__).resolve().parents[1]
REMAINING_ROLES={'pre_close','support_release','release_retreat'}


def mechanism_passed(result:dict,role:str)->bool:
    if role=='hold_to_lift':
        return bool(result.get('mechanism_success') is True and result.get('current_cup_lift_m',0)>=.02
            and 0<result.get('new_active_points',401)<=400 and result.get('active_policy_points',3901)<=3900
            and result.get('error') is None and result.get('contact_state')=='holding'
            and result.get('unexpected_arm_cup_contact_frames',1)==0 and result.get('execution_contract_failures',1)==0
            and not result.get('premature_release_before_support',True))
    return role in REMAINING_ROLES and remaining_flow_passed(result)


def _passed(record):
    return bool(record.get('complete') is True and len(record.get('lift_probes',[]))==3
        and all(v is True for v in record['lift_probes']) and set(record.get('remaining_roles',{}))==REMAINING_ROLES
        and all(v is True for v in record['remaining_roles'].values()))


def pilot_gate(A:dict,B:dict)->str:
    return 'formal_pair' if _passed(B) else 'A_development_only' if _passed(A) else 'stop'


def load_pilot(training_run):
    from training.integrity import require_finalized
    from executor.model import BoundedActPolicy,verify_paired_readout_statistics
    training_run=Path(training_run);checkpoint=training_run/'checkpoints/policy_last.ckpt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=dict(payload['config'],device='cuda',pretrained_backbone=False)
    if config.get('training_budget')!={'kind':'updates','limit':1000} or config.get('history_runtime_profile')!='history_common_v1':
        raise ValueError('Only the fixed-last1000-update common-history pilot may enter mechanism checks')
    require_finalized(checkpoint,payload['config'],allowed_budgets=('updates',))
    if not torch.cuda.is_available():raise RuntimeError('Host CUDA pilot evaluation runtime required')
    torch.set_num_threads(2);torch.manual_seed(42)
    with (training_run/'checkpoints/dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
    verify_paired_readout_statistics(config,stats)
    policy=BoundedActPolicy(config).cuda();policy.load_state_dict(payload['model'],strict=True);policy.eval()
    policy.config=config;policy.device=torch.device('cuda')
    return policy,stats


def run_mechanism(training_run:Path,captures:Path,output:Path)->dict:
    from simulation.supervised_rollout import rollout
    from training.history_recoveries import validate_history_manifest,sha
    captures,output=Path(captures),Path(output)
    manifest=json.loads((captures/'manifest.json').read_text())
    base_dir=ROOT.parent/'nero_act_generalized_pick_place_20261018/training_data/generalized_v1'
    split=json.loads((base_dir/'split.json').read_text());validate_history_manifest(manifest,split)
    policy,stats=load_pilot(training_run)
    output.mkdir(parents=True,exist_ok=False)
    base={r['id']:r for r in manifest['base_episodes']}
    nominal=ROOT/'results/original_failure_capture'
    nominal_scene=json.loads((nominal/'scene_configuration.json').read_text())
    nominal_scenario=json.loads((nominal/'result.json').read_text())['scenario']
    nominal_contexts=json.loads((nominal/'runtime_checkpoints.json').read_text())['named']
    probes=[];remaining={};records=[]
    cases=[]
    for role in ['pre_close','hold_to_lift']:
        cases.append((f'nominal_{role}',nominal_scene,nominal_scenario,nominal_contexts[role],'hold_to_lift',None))
    heldout=[r for r in manifest['episodes'] if r['split']=='val']
    for row in heldout:
        parent=base[row['parent_episode_id']]
        if sha(parent['path'])!=parent['sha256']:raise ValueError('Held-out parent physical source changed')
        with h5py.File(parent['path'],'r') as f:scene=json.loads(f.attrs['scene_config'])
        scenario=dict(seed=scene['generalization']['seed'],query_stride=2,servo_delay_steps=scene.get('action_delay_steps',0),
                      observation_age_s=.05,jitter=False,force=False)
        cases.append((f'heldout_{row["role"]}',scene,scenario,json.loads(Path(row['context_path']).read_text()),row['role'],parent['sha256']))
    for name,scene,scenario,context,role,parent_sha in cases:
        result=rollout(copy.deepcopy(scene),policy,stats,copy.deepcopy(scenario),output/name,resume_context=context,
            runtime_profile='history_common_v1',resume_policy_transition=True,expected_parent_source_sha256=parent_sha,
            mechanism_role='hold_to_lift' if role=='hold_to_lift' else None)
        passed=mechanism_passed(result,role)
        if role=='hold_to_lift':probes.append(passed)
        else:remaining[role]=passed
        records.append(dict(name=name,role=role,passed=passed,result=result,partial_recovery_not_full_demonstration=True))
        print(json.dumps(dict(name=name,role=role,passed=passed,new_points=result.get('new_active_points'),error=result.get('error'))),flush=True)
        (output/'status.json').write_text(json.dumps(dict(completed=len(records),total=6),indent=2))
    summary=dict(complete=True,lift_probes=probes,remaining_roles=remaining,records=records,
        complete_task_qualification=False,hardware_deployment_allowed=False,training_run=str(Path(training_run).resolve()),
        checkpoint_sha256=sha(Path(training_run)/'checkpoints/policy_last.ckpt'),
        source_seal_sha256=sha(ROOT/'evidence/pre_optimization_freeze.json'))
    summary['mechanism_and_recoveries_passed']=_passed(summary)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['training-run','captures','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run_mechanism(args.training_run,args.captures,args.output);return 0


if __name__=='__main__':raise SystemExit(main())
