"""Predeclare unseen16 physical conditions before run16, same test distribution."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
from simulation.evaluate_generalized import test_cases
from simulation.generalization import pose,rotation_xyz,validate_initial_pose
from simulation.teacher import inverse_kinematics
from simulation.scene import build_scene
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'nero_act_contact_recovery_20261018'
center=json.loads((ROOT/'scene/center.json').read_text());site=json.loads((OLD/'evidence/site_snapshot.json').read_text())
cases=test_cases(center,site)
for i,case in enumerate(cases):
 rng=np.random.default_rng(810000+i);c=case['scene'];initial=c['initial_robot_state'];m=build_scene(c)
 _,point,rotation=pose(m,initial['qpos_rad'],initial['gripper_total_width_m'])
 offset=rng.uniform(-.004,.004,3);angles=rng.uniform(-.025,.025,3)
 if 9<=i<13:
  m0=build_scene(center);s0=center['initial_robot_state']
  _,p0,r0=pose(m0,s0['qpos_rad'],s0['gripper_total_width_m'])
  offset+=np.asarray(c['generalization']['initial_translation_m'])*.7;point,rotation=p0,r0
 q=inverse_kinematics(m,point+offset,np.array(initial['qpos_rad']),rotation_xyz(angles)@rotation)
 initial.update(qpos_rad=q.tolist(),gripper_total_width_m=float(rng.uniform(.09,.095)))
 c['appearance'].update(table_rgb=rng.uniform(.78,.96,3).tolist(),
  cup_rgb=[float(rng.uniform(.85,1)),float(rng.uniform(.68,.87)),float(rng.uniform(.015,.09))],
  light_scale=float(rng.uniform(.82,1.15)))
 c['generalization']=dict(independent_seed=810000+i,initial_translation_m=offset.tolist(),initial_rotation_rad=angles.tolist())
 case.update(name='fresh2_'+case['name'],seed=810000+i,servo_delay_steps=i%3)
 if i<13:validate_initial_pose(build_scene(c),c,center)
old=json.loads((OLD/'evidence/predeclared_independent_16.json').read_text())
assert all(a['scene']!=b['scene'] for a,b in zip(cases,old))
path=ROOT/'evidence/predeclared_independent_16_v2.json'
with path.open('x') as f:json.dump(cases,f,indent=2)
print(dict(cases=16,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prior_tests_now_development=True))
