"""Additional complete simulated demonstrations at independently split delays."""
import copy,json
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay
root=Path(__file__).resolve().parents[1];out=root/'training_data_current_source'
base=json.loads((out/'scene_config.json').read_text());manifest=out/'delayed_manifest.json'
if manifest.exists():raise FileExistsError(manifest)
entries=[];rejected=[]
for index in range(6):
 config=copy.deepcopy(base);rng=np.random.default_rng(75000+index)
 config['action_delay_steps']=[2,10,2,10,2,10][index]
 for key in ['pick_xy_m','place_xy_m']:
  config[key]=(np.asarray(config[key])+rng.uniform(-.003,.003,2)).tolist()
 config['initial_robot_state']['qpos_rad']=(np.asarray(config['initial_robot_state']['qpos_rad'])+rng.uniform(-.002,.002,7)).tolist()
 config['appearance']['light_scale']=.90+.005*index
 path=out/f'episode_{20+index}.hdf5'
 try:
  result=collect_episode(config,path);replay=verify_replay(path)
  entries.append(dict(path=str(path),status='valid',seed=75000+index,frames=result['frames'],teacher_success=True,full_task_verified=True,split='train' if index<4 else 'val',delay_steps=config['action_delay_steps'],replay=replay))
  print(json.dumps(dict(episode=20+index,delay_steps=config['action_delay_steps'],success=True,replay=replay)),flush=True)
 except Exception as exc:
  rejected.append(dict(episode=20+index,error=str(exc)));print(json.dumps(rejected[-1]),flush=True)
 manifest.write_text(json.dumps(dict(episodes=entries,rejected=rejected),indent=2))
 if rejected:raise RuntimeError('Delayed full demonstration failed; retained, no training')
