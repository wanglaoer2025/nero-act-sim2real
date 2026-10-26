import json
from pathlib import Path
import h5py,numpy as np,mujoco
from executor.ros_execution import RobotGeometry
from simulation.scene import build_scene
from simulation.supported_attachment import SupportedAttachment
root=Path.cwd();rows=[]
for number in [2016,2017,2013,2014]:
 with h5py.File(root/f'training_data/history_complete_v1/episode_{number}.hdf5','r') as f:
  scene=json.loads(f.attrs['scene_config']);q=f['simulation/qpos'][:];vel=f['simulation/qvel'][:];actions=f['action'][:];stages=f['simulation/stage'].asstr()[:]
 model=build_scene(scene);d=mujoco.MjData(model);geo=RobotGeometry(scene)
 flange=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'link7');cup=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cup')
 checks=[]
 for i in range(5,min(len(actions),200)):
  if stages[i] not in ('close','lift','release'):continue
  d.qpos[:]=q[i];d.qvel[:]=vel[i];mujoco.mj_forward(model,d)
  a=SupportedAttachment(radius=scene['cup']['outer_diameter_m']/2,half_height=scene['cup']['height_m']/2)
  a.observe(dict(position=d.xpos[flange],rotation=d.xmat[flange].reshape(3,3)),d.xpos[cup],d.xmat[cup].reshape(3,3),supported=True)
  current=np.r_[q[i,:7],q[i,7]-q[i,8]]
  bottom=a.bottom(geo.pose(actions[i]),scene['table_top_z_m']);initial=a.bottom(geo.pose(current),scene['table_top_z_m'])
  checks.append(dict(index=i,stage=stages[i],current_bottom_m=initial,commanded_bottom_m=bottom,command_minus_current_m=bottom-initial,refused_if_attachment_active=bool(bottom<-1e-6)))
 rows.append(dict(episode=number,checked=len(checks),would_be_refused=sum(r['refused_if_attachment_active'] for r in checks),first20=checks[:20]))
 print(number,rows[-1]['would_be_refused'],'/',len(checks),flush=True)
(root/'evidence/teacher_guard_compatibility.json').write_text(json.dumps(dict(complete=True,diagnostic_only=True,does_not_assert_attachment_active_before_lift=True,records=rows),indent=2)+'\n')
