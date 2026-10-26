import hashlib,json
from pathlib import Path
import h5py,numpy as np
from training.generalized import combined_inputs
from simulation.evaluate_generalized import fresh_test_cases
from executor.ros_execution import RobotGeometry
root=Path(__file__).resolve().parents[1];base=root.parent/'nero_act_generalized_pick_place_20261018/training_data/generalized_v1'
manifest,split,paths=combined_inputs(json.loads((base/'manifest.json').read_text()),json.loads((base/'split.json').read_text()),
 json.loads((root/'training_data/recoveries_v1/manifest.json').read_text()))
report=dict(full_episodes=48,partial_corrections=len(paths)-48,train=len(split['train']),val=len(split['val']),initial_lift_dz_mm={})
for path in paths[-6:]:
 with h5py.File(path) as f:
  c=json.loads(f.attrs['scene_config']);state=np.r_[f['observations/qpos'][0],f['observations/gripper_state'][0]]
  g=RobotGeometry(c);p=g.pose(state)['position'];dz=[float((g.pose(a)['position'][2]-p[2])*1000) for a in f['action'][:10]]
  report['initial_lift_dz_mm'][path.stem]=dz
  assert min(dz)>0,'first correction prefix must actually point upward'
center=json.loads((root/'scene/center.json').read_text());site=json.loads((root/'evidence/site_snapshot.json').read_text())
cases=fresh_test_cases(center,site)
with (root/'evidence/predeclared_independent_16.json').open('x') as f:json.dump(cases,f,indent=2)
report['independent_definition_sha256']=hashlib.sha256((root/'evidence/predeclared_independent_16.json').read_bytes()).hexdigest()
(root/'evidence/recovery_input_acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
