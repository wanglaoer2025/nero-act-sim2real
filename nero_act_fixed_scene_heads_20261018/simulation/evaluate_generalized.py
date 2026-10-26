"""Frozen online-policy position tests, not a trajectory replay or hardware test."""
import argparse,copy,hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch
from simulation.scene import build_scene
from simulation.teacher import inverse_kinematics
from simulation.generalization import pose,rotation_xyz,validate_initial_pose
from simulation.evaluate import run_frozen_cases
from simulation.supervised_rollout import rollout
from executor.bundle import verify_model_bundle
from executor.model import BoundedActPolicy,verify_paired_readout_statistics
from executor.evaluation_integrity import evaluation_sources,verify_frozen_artifacts,SUPERVISED_PROTOCOL

ROOT=Path(__file__).resolve().parents[1]


def test_cases(center,site):
    result=[];common=dict(query_stride=10,servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False)
    for x in [-.02,0,.02]:
        for y in [-.02,0,.02]:
            c=copy.deepcopy(center);c['pick_xy_m']=(np.array(center['pick_xy_m'])+[x,y]).tolist()
            result.append(dict(common,name=f'cup_{round(x*1000):+d}_{round(y*1000):+d}',scene=c))
    translations=[[.02,0,.01],[-.02,0,-.01],[0,.02,.02],[0,-.02,-.02]]
    for i,xyz in enumerate(translations):
        c=copy.deepcopy(center);r=np.deg2rad([3 if i%2 else -3,1.5,-1.5])
        m=build_scene(c);initial=c['initial_robot_state'];_,point,rotation=pose(m,initial['qpos_rad'],initial['gripper_total_width_m'])
        q=inverse_kinematics(m,point+xyz,np.array(initial['qpos_rad']),rotation_xyz(r)@rotation)
        c['initial_robot_state']['qpos_rad']=q.tolist()
        c['generalization']=dict(initial_translation_m=xyz,initial_rotation_rad=r.tolist())
        validate_initial_pose(build_scene(c),c,center)
        result.append(dict(common,name=f'initial_pose_{i}',scene=c))
    for i in range(3):
        c=copy.deepcopy(center);c['pick_xy_m']=site['source_xy_m'];c['place_xy_m']=site['target_xy_m']
        c['initial_robot_state']=dict(qpos_rad=site['actual_state'][:7],gripper_total_width_m=[.094,.095,.0935][i])
        c['arm_kp']=(np.array(c['arm_kp'])*[.99,1.,1.01][i]).tolist()
        result.append(dict(common,name=f'last_site_geometry_prepared_open_{i}',scene=c,servo_delay_steps=i,
            requires_gripper_preparation=abs(site['actual_state'][7]-c['initial_robot_state']['gripper_total_width_m'])>.004,
            historical_snapshot_only=True))
    for i,c in enumerate(result):c['seed']=450000+i
    return result


def fresh_test_cases(center,site):
    """Predeclared unseen starts/appearance/delay; old16 are development regressions."""
    cases=test_cases(center,site)
    for i,case in enumerate(cases):
        rng=np.random.default_rng(780000+i);c=case['scene']
        initial=c['initial_robot_state'];m=build_scene(c)
        _,point,rotation=pose(m,initial['qpos_rad'],initial['gripper_total_width_m'])
        offset=rng.uniform(-.004,.004,3);angles=rng.uniform(-.025,.025,3)
        if 9<=i<13:
            m0=build_scene(center);s0=center['initial_robot_state']
            _,p0,r0=pose(m0,s0['qpos_rad'],s0['gripper_total_width_m'])
            offset+=np.asarray(c['generalization']['initial_translation_m'])*.7
            point,rotation=p0,r0
        q=inverse_kinematics(m,point+offset,np.array(initial['qpos_rad']),rotation_xyz(angles)@rotation)
        initial.update(qpos_rad=q.tolist(),gripper_total_width_m=float(rng.uniform(.09,.095)))
        c['appearance'].update(table_rgb=rng.uniform(.78,.96,3).tolist(),
            cup_rgb=[float(rng.uniform(.85,1)),float(rng.uniform(.68,.87)),float(rng.uniform(.015,.09))],
            light_scale=float(rng.uniform(.82,1.15)))
        c['generalization']=dict(independent_seed=780000+i,initial_translation_m=offset.tolist(),
            initial_rotation_rad=angles.tolist())
        case.update(name='fresh_'+case['name'],seed=780000+i,servo_delay_steps=i%3)
        if i<13:validate_initial_pose(build_scene(c),c,center)
    return cases


def load_predeclared_cases(config):
    path=Path(config['independent_cases_file'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=config['independent_cases_sha256']:
        raise ValueError('Predeclared independent cases changed after training was bound')
    cases=json.loads(path.read_text())
    if (not isinstance(cases,list) or len(cases)!=16 or len({c['name'] for c in cases})!=16
            or any(not isinstance(c.get('scene'),dict) for c in cases)):
        raise ValueError('Exactly16 independently declared physical scenarios required')
    return cases


def assessment_role(config,*,fresh=False,development=False):
    if fresh:return 'fresh_independent_16_predeclared_before_training'
    if development or config.get('prior_evaluation_scope')=='development_regression_after_failed_state_corrections':
        return 'development_regression_used_for_corrections'
    return 'original_run14_test_definition'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--site',type=Path,default=ROOT/'evidence/site_snapshot.json')
    p.add_argument('--nominal-only',action='store_true')
    p.add_argument('--fresh-independent',action='store_true')
    p.add_argument('--development-regression',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.fresh_independent and (a.development_regression or a.nominal_only):p.error('Fresh independent suite must run all16 cases separately')
    metadata=verify_model_bundle(a.model);payload=torch.load(a.model/'policy.ckpt',map_location='cpu',weights_only=False)
    config=dict(payload['config'],device='cuda',pretrained_backbone=False)
    with (a.model/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
    verify_paired_readout_statistics(config,stats);torch.set_num_threads(2)
    policy=BoundedActPolicy(config).cuda();policy.load_state_dict(payload['model']);policy.eval()
    policy.config=config;policy.device=torch.device('cuda')
    center=config['scene_config']
    if a.fresh_independent:cases=load_predeclared_cases(config)
    else:cases=test_cases(center,json.loads(a.site.read_text()))
    if a.nominal_only:cases=[cases[4]]
    sources=evaluation_sources()
    protocol=dict(checkpoint_sha256=metadata['checkpoint_sha256'],stats_sha256=metadata['stats_sha256'],
        scene_sha256=metadata['scene_sha256'],training_config_sha256=metadata['training_config_sha256'],
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        scenarios=cases,teacher_free=True,real_hardware_tested=False,hardware_deployment_allowed=False,
        supervised_contact_protocol=SUPERVISED_PROTOCOL,
        site_snapshot_sha256=None if a.fresh_independent else hashlib.sha256(a.site.read_bytes()).hexdigest(),
        evaluation_scope='online_closed_loop_generalization; historical site snapshot with prepared opening; no hardware readiness claim')
    protocol['assessment_role']=assessment_role(config,fresh=a.fresh_independent,development=a.development_regression)
    if a.fresh_independent:protocol['predeclared_independent_cases_sha256']=config['independent_cases_sha256']
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summary=run_frozen_cases(a.output,protocol,
        lambda case:rollout(case['scene'],policy,stats,case,a.output/case['name']),
        lambda:verify_frozen_artifacts(a.model/'policy.ckpt',protocol))
    print(json.dumps(dict(passed=summary['passed'],total=summary['total'],hardware_tested=False)),flush=True)


if __name__=='__main__':main()
