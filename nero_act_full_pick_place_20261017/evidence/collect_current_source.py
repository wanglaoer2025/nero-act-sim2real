"""Recorded invocation of the tested simulation collector; no hardware access."""
import copy,json
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay
root=Path(__file__).resolve().parents[1]
out=root/'training_data_current_source'
base=json.loads((out/'scene_config.json').read_text())
manifest=out/'manifest.json'
if manifest.exists():raise FileExistsError(manifest)
entries=[]; rejected=[]
for index in range(12):
 config=copy.deepcopy(base);rng=np.random.default_rng(73000+index)
 if index:
  for key in ['pick_xy_m','place_xy_m']:
   config[key]=(np.asarray(config[key])+rng.uniform(-.003,.003,2)).tolist()
  config['initial_robot_state']['qpos_rad']=(np.asarray(config['initial_robot_state']['qpos_rad'])+rng.uniform(-.002,.002,7)).tolist()
 config['appearance']['light_scale']=.88+.01*index
 if index in [3,5,7,9,11]:config['arm_disturbance']=dict(seed=74000+index,max_torque_nm=.15,period_steps=47,duration_steps=3,start_step=90)
 path=out/f'episode_{index}.hdf5'
 try:
  result=collect_episode(config,path);replay=verify_replay(path)
  entries.append(dict(path=str(path),status='valid',seed=73000+index,frames=result['frames'],teacher_success=True,replay=replay))
  print(json.dumps(dict(episode=index,success=True,frames=result['frames'],replay=replay)),flush=True)
 except Exception as exc:
  rejected.append(dict(episode=index,error=str(exc)));print(json.dumps(rejected[-1]),flush=True)
 manifest.write_text(json.dumps(dict(episodes=entries,rejected=rejected),indent=2))
 if rejected:raise RuntimeError('New-profile physical demonstration failed; retained, no training')
(out/'split.json').write_text(json.dumps(dict(train=list(range(8)),val=[8,9],diagnostic=[10,11]),indent=2))
