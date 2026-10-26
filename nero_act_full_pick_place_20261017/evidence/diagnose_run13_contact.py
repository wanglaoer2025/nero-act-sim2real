"""Restore the final query state and identify the recorded first unexpected contact."""
import json
from pathlib import Path
import mujoco,numpy as np
from simulation.scene import build_scene
from simulation.snapshot import restore
from simulation.nero_domain_randomization import NeroServo,disturbance_torque
from simulation.teacher import observe
p=Path('evidence/run13_supervised_development/nominal');c=json.loads((p/'scene_configuration.json').read_text());m=json.loads((p/'measurements.json').read_text());r=json.loads((p/'result.json').read_text());record=json.loads((p/'physical_snapshots.json').read_text());t=np.load(p/'rollout.npz')
model=build_scene(c);data=mujoco.MjData(model);servo=NeroServo(model,data,c);previous=restore(model,data,servo,record['snapshots'][-1]);start=record['query_steps'][-1];found=[];maxerror=0
for step in range(start,len(t['sent_targets'])):
 state=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]];maxerror=max(maxerror,float(max(abs(state-t['states'][step]))))
 target=t['sent_targets'][step];servo.command(target[:7],target[7]);data.qfrc_applied[:7]=disturbance_torque(c,step)
 for micro in range(25):
  servo.apply();mujoco.mj_step(model,data)
  if observe(model,data,c)['unexpected_contact']:
   for index,contact in enumerate(data.contact[:data.ncon]):
    bodies=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in [contact.geom1,contact.geom2]]
    if 'cup' in bodies and any(b not in ['world','cup','gripper_link1','gripper_link2'] for b in bodies):
     force=np.zeros(6);mujoco.mj_contactForce(model,data,index,force);found.append(dict(step=step,micro=micro,bodies=bodies,distance_m=float(contact.dist),force=force.tolist(),measurement=observe(model,data,c)))
   break
 if found:break
 mujoco.mj_forward(model,data)
report=dict(first_unexpected_contacts=found,replay_max_q_error=maxerror,last_measurement=m[-1],carry_measurement=m[next(e['physics_step'] for e in r['simulated_events'] if e['kind']=='CARRY')-1])
Path('evidence/run13_contact_diagnosis.json').write_text(json.dumps(report,indent=2));print(report)
