"""Inspect existing TRAIN labels at physical table support, without changing data."""
import json
from pathlib import Path
import h5py,mujoco,numpy as np
from simulation.scene import build_scene
from simulation.teacher import observe
from executor.ros_execution import RobotGeometry
root=Path(__file__).resolve().parents[1];ids=json.loads((root/'training/run10_supervised_release/split.json').read_text())['train'];records=[]
for i in ids:
 p=root/'training_data_current_source'/f'episode_{i}.hdf5'
 with h5py.File(p,'r') as f:
  c=json.loads(f.attrs['scene_config']);stages=f['simulation/stage'].asstr()[:]
  q=f['simulation/qpos'][:];v=f['simulation/qvel'][:];a=f['action'][:]
 indices=np.flatnonzero(stages=='place')[::10]
 if not len(indices):continue
 model=build_scene(c);data=mujoco.MjData(model);geometry=RobotGeometry(c);n=0;down=[]
 for t in indices:
  data.qpos[:]=q[t];data.qvel[:]=v[t];mujoco.mj_forward(model,data);m=observe(model,data,c)
  if m['supported'] and m['bilateral'] and np.linalg.norm(np.array(m['cup_xyz'])[:2]-c['place_xy_m'])<.02:
   n+=1;down.append(float(m['tcp_xyz'][2]-geometry.pose(a[t])['tcp'][2]))
 if n:records.append(dict(episode=i,sampled_supported_closed_place_labels=n,max_commanded_downward_offset_m=max(down)))
report=dict(training_only=True,episodes=records,supported_closed_place_labels=sum(r['sampled_supported_closed_place_labels'] for r in records),
 interpretation='These supported descent/closed labels conflict with the paused runtime stop-then-open continuation; this is evidence of protocol mismatch, not proof that single-frame ACT is impossible.')
(root/'evidence/supported_place_label_audit.json').write_text(json.dumps(report,indent=2));print(report)
