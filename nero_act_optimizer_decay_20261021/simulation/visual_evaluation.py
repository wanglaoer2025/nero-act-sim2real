"""Frozen visual15D closed-loop physics assessment in the retained nominal domain."""
import argparse
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from executor.model import BoundedActPolicy, verify_paired_readout_statistics
from executor.ros_execution import RobotGeometry
from training.integrity import require_finalized
from training.legacy_visual_data import NominalImageEstimator, training_profile
from simulation.scene import build_scene
from simulation.generalization import pose, validate_initial_pose
from simulation.teacher import inverse_kinematics

ROOT = Path(__file__).resolve().parents[1]


class ExposureHistory:
    def __init__(self, delay_steps):
        if delay_steps not in (0,1):raise ValueError('Only zero/one50ms exposure delay is declared')
        self.rows=deque(maxlen=delay_steps+1)
        self.required=delay_steps+1

    def append(self, snapshot):self.rows.append(copy.deepcopy(snapshot))
    def reset(self):self.rows.clear()
    def take(self):
        if len(self.rows)<self.required:raise ValueError('Delayed exposure warmup required')
        return copy.deepcopy(self.rows[0])


def evaluation_dependencies(root, training_run, *, protocol_path=None,checkpoint_name='policy_best_val.ckpt'):
    root,training_run=Path(root),Path(training_run)
    files=[p for folder in ['executor','simulation','training','vision','calibration','policy_source']
        for p in (root/folder).rglob('*.py')]
    files += [p for p in (root/'simulation/assets').rglob('*') if p.is_file()]
    files += [root/'simulation/scene_template.xml',root/'evaluation/protocol.json',
        training_run/'checkpoints'/checkpoint_name,training_run/'checkpoints/dataset_stats.pkl']
    if protocol_path is not None:files.append(Path(protocol_path))
    runtime=root/'config/history_runtime_contract.json'
    if runtime.is_file():files.append(runtime)
    candidate=root/'evidence/candidate_selection.json'
    if candidate.is_file():files.append(candidate)
    baseline=root/'evidence/baseline_development_protocol.json'
    if baseline.is_file():files.append(baseline)
    return sorted(set(files))


def file_hashes(paths):
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def require_files(hashes):
    for p, expected in hashes.items():
        if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen evaluation dependency changed: '+p)


def aggregate_cases(records, expected_ids):
    names=[r['name'] for r in records]
    if len(records)!=16 or len(set(names))!=16 or set(names)!=set(expected_ids):
        raise ValueError('All16 unique expected cases must finish before full-suite finalization')
    return dict(complete=True,finalized=True,total=16,passed=sum(r['passed'] is True for r in records),
                records=records,hardware_deployment_allowed=False)


def _completion_criteria(result):
    return bool(result.get('success') is True and result.get('error') is None
        and result.get('max_lift_m', 0) >= .18
        and result.get('maximum_horizontal_transfer_m', 0) > .1
        and result.get('stable_released_frames', 0) >= 100
        and result.get('unexpected_arm_cup_contact_frames', 1) == 0
        and result.get('execution_contract_failures', 1) == 0
        and not result.get('premature_release_before_support', True)
        and result.get('active_policy_points', 4000) <= 3900
        and [e['kind'] for e in result.get('simulated_events', [])] == ['GRASP','CARRY','RELEASE','RETREAT'])


def full_task_passed(result):
    return not result.get('resumed_partial_not_full',False) and _completion_criteria(result)


def remaining_flow_passed(result):
    return result.get('resumed_partial_not_full') is True and _completion_criteria(result)


def select_independent_candidate(records:list[dict])->str|None:
    variants=[r.get('variant') for r in records]
    if len(set(variants))!=len(variants) or any(v not in ('A','B') for v in variants):
        raise ValueError('Only one preselected checkpoint per A/B variant may enter selection')
    eligible=[r for r in records if r.get('finalized') is True and r.get('development_passed')==16
        and r.get('development_total')==16 and np.isfinite(r.get('zero_latent_val_loss',np.nan))]
    if not eligible:return None
    if len(eligible)==2 and abs(eligible[0]['zero_latent_val_loss']-eligible[1]['zero_latent_val_loss'])<=1e-8:return 'A'
    return min(eligible,key=lambda r:r['zero_latent_val_loss'])['variant']


class ImageFeatures:
    def __init__(self, scene, image_size=(180,320), *, camera_translation_m=None):
        observed_scene = copy.deepcopy(scene)
        if camera_translation_m is not None:
            t=np.asarray(observed_scene['camera_calibration']['transform_base_from_raw_optical_m'])
            t[:3,3] += np.asarray(camera_translation_m)
            observed_scene['camera_calibration']['transform_base_from_raw_optical_m'] = t.tolist()
        self.estimator = NominalImageEstimator(training_profile(observed_scene), image_size)
        self.robot = RobotGeometry(scene)

    def __call__(self, rgb, state):
        cup, card, diagnostics = self.estimator.measure(rgb)
        tcp = self.robot.pose(state)['tcp']
        return np.r_[cup-tcp,card-tcp].astype(np.float32), dict(cup_bottom_base_m=cup.tolist(),
            target_base_m=card.tolist(), tcp_base_m=tcp.tolist(), **diagnostics)


def image_features(rgb, state, scene):
    return ImageFeatures(scene, rgb.shape[:2])(rgb, state)[0]


def resolved_case_provenance(protocol,protocol_path=None):
    cases=protocol.get('resolved_cases',[])
    if (protocol.get('parameters_frozen_before_optimization') is not True
            or len(cases)!=16 or len({r.get('name') for r in cases})!=16
            or any(not isinstance(r.get('scene'),dict) or type(r.get('seed')) is not int for r in cases)):
        raise ValueError('All16 complete numeric physical cases must be resolved and frozen before optimization')
    path=Path(protocol_path or ROOT/'evaluation/protocol.json').resolve()
    return dict(all_parameters_predeclared_before_training=True,
        original_protocol_sha256=file_hashes([path])[str(path)],
        measured_geometry=False,live_frontend_qualified=False,hardware_deployment_allowed=False,
        assessment_role='fresh_fully_resolved_nominal_domain_cases_frozen_before_optimization')


def build_cases(protocol, center,*,protocol_path=None):
    if 'resolved_cases' in protocol:
        return copy.deepcopy(protocol['resolved_cases']),resolved_case_provenance(protocol,protocol_path)
    result=[]
    for i, row in enumerate(protocol['cases']):
        c = copy.deepcopy(center)
        case = dict(name=row['id'],seed=protocol['seed']+i,query_stride=2,
            servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False)
        if 'source_offset_m' in row:
            c['pick_xy_m']=(np.asarray(c['pick_xy_m'])+row['source_offset_m']).tolist()
        if 'initial_translation_m' in row:
            model=build_scene(c);initial=c['initial_robot_state']
            _,point,rotation=pose(model,initial['qpos_rad'],initial['gripper_total_width_m'])
            q=inverse_kinematics(model,point+row['initial_translation_m'],np.array(initial['qpos_rad']),rotation)
            initial['qpos_rad']=q.tolist()
            c['generalization']=dict(initial_translation_m=row['initial_translation_m'],initial_rotation_rad=[0,0,0])
            validate_initial_pose(build_scene(c),c,center)
        if row['id']=='heldout_nominal_camera_perturbation':
            case['observation_camera_translation_m']=[.002,0,0]
        if row['id']=='heldout_observation_delay':
            case.update(servo_delay_steps=0,observation_delay_steps=1,observation_age_s=.05)
        case['scene']=c;result.append(case)
    if len(result)!=16 or len({r['name'] for r in result})!=16:
        raise ValueError('All original16 case IDs required')
    path=Path(protocol_path or ROOT/'evaluation/protocol.json').resolve()
    provenance=dict(all_parameters_predeclared_before_training=False,
        reason='Three final case IDs had no numeric perturbations in the original protocol; definitions completed after training',
        original_protocol_sha256=file_hashes([path])[str(path)],
        measured_geometry=False,live_frontend_qualified=False,hardware_deployment_allowed=False)
    return result,provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nominal-only',action='store_true')
    parser.add_argument('--case-index',type=int)
    parser.add_argument('--development-regression',action='store_true')
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--pilot-development',action='store_true')
    parser.add_argument('--nominal-summary',type=Path)
    args=parser.parse_args()
    if args.pilot_development and (not args.nominal_only or args.development_regression or args.case_index is not None):
        parser.error('Pilot development is only an explicit A nominal development run')
    checkpoint=args.training_run/'checkpoints'/('policy_last.ckpt' if args.pilot_development else 'policy_best_val.ckpt')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if args.pilot_development and (payload['config'].get('variant')!='A' or payload['config'].get('training_budget')!={'kind':'updates','limit':1000}):
        raise ValueError('Only the finalized A1000-update pilot has this restricted development role')
    require_finalized(checkpoint,payload['config'],allowed_budgets=('updates',) if args.pilot_development else ('epochs',))
    gate_inputs=[]
    if payload['config'].get('history_runtime_profile')=='history_common_v1' and (args.pilot_development or args.development_regression):
        from executor.history_qualification import require_development_gate
        gate_inputs=require_development_gate(ROOT,args.training_run,pilot=args.pilot_development,nominal_summary=args.nominal_summary)
    if (payload['config'].get('history_runtime_profile')=='history_common_v1'
            and not args.nominal_only and not args.development_regression):
        from executor.history_qualification import require_independent_candidate
        require_independent_candidate(ROOT,args.training_run)
    config=dict(payload['config'],device='cuda',pretrained_backbone=False)
    if config.get('visual_observation_profile')!='visual_relative_base_v1':
        raise ValueError('Visual-relative15D trained policy required')
    with (args.training_run/'checkpoints/dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
    verify_paired_readout_statistics(config,stats)
    torch.set_num_threads(2);torch.manual_seed(42)
    policy=BoundedActPolicy(config).cuda();policy.load_state_dict(payload['model']);policy.eval()
    policy.config=config;policy.device=torch.device('cuda')
    declared_protocol_path=Path(args.protocol or ROOT/'evaluation/protocol.json').resolve()
    selected_protocol_path=declared_protocol_path
    protocol=json.loads(selected_protocol_path.read_text())
    if payload['config'].get('visual_sampling_profile'):
        proof=json.loads((args.training_run/'artifacts/training_integrity.json').read_text())
        protocol_path=str(declared_protocol_path)
        if proof['protected_sha256'].get(protocol_path)!=file_hashes([declared_protocol_path])[protocol_path]:
            raise ValueError('Independent numeric evaluation protocol was not bound before this training run')
    if args.development_regression:
        selected_protocol_path=ROOT/'evidence/baseline_development_protocol.json'
        protocol=json.loads(selected_protocol_path.read_text())
    if args.nominal_only:
        cases=[dict(name='nominal_development',seed=191899,query_stride=2,servo_delay_steps=0,
            observation_age_s=.05,jitter=False,force=False,scene=config['scene_config'])]
        provenance=dict(assessment_role='nominal_development_not_final_test',hardware_deployment_allowed=False)
    else: cases,provenance=build_cases(protocol,config['scene_config'],protocol_path=selected_protocol_path)
    if args.development_regression:provenance['assessment_role']='paired_baseline_development_not_independent'
    if args.case_index is not None:
        if args.nominal_only or not 0<=args.case_index<16:parser.error('case-index requires one of the full16 cases')
        cases=[cases[args.case_index]]
        provenance['partial_suite_case_index']=args.case_index
    dependencies=evaluation_dependencies(ROOT,args.training_run,protocol_path=declared_protocol_path,checkpoint_name=checkpoint.name)
    dependencies+=gate_inputs
    bindings=file_hashes(dependencies)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(cases=cases,provenance=provenance,
        dependencies=bindings,observation_schema='visual_relative_base_v1',hardware_deployment_allowed=False),indent=2))
    results=[];started=time.monotonic()
    from simulation.supervised_rollout import rollout
    for case in cases:
        result=rollout(case['scene'],policy,stats,case,args.output/case['name'],
            runtime_profile=config.get('history_runtime_profile','legacy_trace_v1'))
        result['full_task_passed']=full_task_passed(result);results.append(result)
        if not result['full_task_passed'] and result['error'] is None:
            result['failure_reason']='declared_point_budget_exhausted_before_full_task_completion'
        print(json.dumps(dict(case=case['name'],full_task_passed=result['full_task_passed'],error=result['error'],
            max_lift_m=result['max_lift_m'],active_points=result['active_policy_points'])),flush=True)
        (args.output/'status.json').write_text(json.dumps(dict(completed=len(results),total=len(cases),
            passed=sum(r['full_task_passed'] for r in results),last_case=case['name']),indent=2))
    require_files(bindings)
    summary=dict(complete=True,finalized=True,passed=sum(r['full_task_passed'] for r in results),total=len(cases),
        results=results,calibration_sha256=config['calibration_sha256'],observation_schema='visual_relative_base_v1',
        provenance=provenance,source_and_model_unchanged=True,hardware_deployment_allowed=False,
        real_hardware_tested=False,elapsed_s=time.monotonic()-started)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(
        passed=summary['passed'],total=summary['total'],finalized=True,hardware_deployment_allowed=False)),flush=True)


if __name__=='__main__':main()
