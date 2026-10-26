"""Controlled before/after nominal simulation, never formal or hardware qualification."""
import argparse,copy,hashlib,json,pickle
from pathlib import Path
import torch
from executor.model import BoundedActPolicy
from simulation.history_mechanism import load_pilot
from simulation.supervised_rollout import rollout
from simulation.visual_evaluation import full_task_passed,file_hashes,require_files,evaluation_dependencies
from training.history import initialize_history,frozen_output_head_state
from training.integrity import require_finalized,require_unchanged
ROOT=Path(__file__).resolve().parents[1]
INITIALIZER=ROOT.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['before','after'],required=True);args=parser.parse_args()
 torch.set_num_threads(2);torch.manual_seed(42)
 data=ROOT/'training_data/approach_complete_v1';scene=json.loads((data/'scene_config.json').read_text());settings=json.loads((ROOT/'config/history_experiment.json').read_text())
 assert settings['adaptation_scope']=='output_heads_v1'
 if args.mode=='before':
  payload=torch.load(INITIALIZER,map_location='cpu',weights_only=False);require_finalized(INITIALIZER,payload['config'])
  old=pickle.loads((INITIALIZER.parent/'dataset_stats.pkl').read_bytes());stats=pickle.loads((data/'dataset_stats.pkl').read_bytes())
  overrides=dict(settings,device='cuda',pretrained_backbone=False,variant='A',stage='pilot',scene_config=scene,training_budget={'kind':'updates','limit':1000},num_epochs=0)
  config,weights=initialize_history(payload,old,stats,overrides,variant='A')
  policy=BoundedActPolicy(config).cuda().eval();policy.load_state_dict(weights,strict=True);policy.config=config;policy.device=torch.device('cuda')
  dependencies=[INITIALIZER,INITIALIZER.parent/'dataset_stats.pkl',data/'dataset_stats.pkl',data/'scene_config.json',ROOT/'config/history_experiment.json']
  dependencies += [p for folder in ['training','executor','simulation','vision','calibration','policy_source'] for p in (ROOT/folder).rglob('*.py')]
  dependencies += [ROOT/'simulation/scene_template.xml',ROOT/'config/history_runtime_contract.json']
  dependencies += [p for p in (ROOT/'simulation/assets').rglob('*') if p.is_file()]
 else:
  policy,stats=load_pilot(ROOT/'training/pilot_A')
  dependencies=evaluation_dependencies(ROOT,ROOT/'training/pilot_A',protocol_path=ROOT/'evaluation/history_protocol.json',checkpoint_name='policy_last.ckpt')
  before=json.loads((ROOT/'results/nominal_before/bindings.json').read_text());require_files(before)
 dependencies += [Path(__file__)];bindings=file_hashes(dependencies)
 output=ROOT/f'results/nominal_{args.mode}';output.mkdir(exist_ok=False)
 (output/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
 frozen=frozen_output_head_state(policy)
 case=dict(name='nominal_development',seed=191899,query_stride=2,servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False,scene=copy.deepcopy(scene))
 result=rollout(scene,policy,stats,case,output/'nominal_development',runtime_profile='history_common_v1')
 require_files(bindings);assert frozen==frozen_output_head_state(policy)
 result['full_task_passed']=full_task_passed(result)
 summary=dict(complete=True,mode=args.mode,optimizer_updates=0 if args.mode=='before' else 1000,full_task_passed=result['full_task_passed'],result=result,
  frozen_parameter_state=frozen,assessment_role='restricted_nominal_comparison_not_formal',formal_qualification=False,hardware_deployment_allowed=False,independent16_evaluated=False)
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(mode=args.mode,passed=result['full_task_passed'],error=result['error'],active_points=result['active_policy_points'],events=[e['kind'] for e in result['simulated_events']])),flush=True)
 return 0

if __name__=='__main__':raise SystemExit(main())
