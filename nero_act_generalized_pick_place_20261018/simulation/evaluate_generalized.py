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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--site',type=Path,default=ROOT/'evidence/site_snapshot.json')
    p.add_argument('--nominal-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    metadata=verify_model_bundle(a.model);payload=torch.load(a.model/'policy.ckpt',map_location='cpu',weights_only=False)
    config=dict(payload['config'],device='cuda',pretrained_backbone=False)
    with (a.model/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
    verify_paired_readout_statistics(config,stats);torch.set_num_threads(2)
    policy=BoundedActPolicy(config).cuda();policy.load_state_dict(payload['model']);policy.eval()
    policy.config=config;policy.device=torch.device('cuda')
    center=config['scene_config'];site=json.loads(a.site.read_text());cases=test_cases(center,site)
    if a.nominal_only:cases=[cases[4]]
    sources=evaluation_sources()
    protocol=dict(checkpoint_sha256=metadata['checkpoint_sha256'],stats_sha256=metadata['stats_sha256'],
        scene_sha256=metadata['scene_sha256'],training_config_sha256=metadata['training_config_sha256'],
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        scenarios=cases,teacher_free=True,real_hardware_tested=False,hardware_deployment_allowed=False,
        supervised_contact_protocol=SUPERVISED_PROTOCOL,site_snapshot_sha256=hashlib.sha256(a.site.read_bytes()).hexdigest(),
        evaluation_scope='online_closed_loop_generalization; historical site snapshot with prepared opening; no hardware readiness claim')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summary=run_frozen_cases(a.output,protocol,
        lambda case:rollout(case['scene'],policy,stats,case,a.output/case['name']),
        lambda:verify_frozen_artifacts(a.model/'policy.ckpt',protocol))
    print(json.dumps(dict(passed=summary['passed'],total=summary['total'],hardware_tested=False)),flush=True)


if __name__=='__main__':main()
