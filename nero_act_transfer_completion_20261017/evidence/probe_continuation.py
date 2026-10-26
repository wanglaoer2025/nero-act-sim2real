import json
from pathlib import Path
from executor.policy_worker import FrozenAct
from simulation.supervised_rollout import rollout
from simulation.evaluate import scenarios
r=Path(__file__).resolve().parents[1];p=FrozenAct(r/'model/frozen_run11',device='cuda');c=json.loads((r/'model/frozen_run11/scene_config.json').read_text());q=json.loads((r/'evidence/paused_pose.json').read_text())['state'];c['initial_robot_state']['qpos_rad']=q[:7];c['initial_robot_state']['gripper_total_width_m']=q[7]
p.model.config,p.model.device=p.config,p.device
print(json.dumps(rollout(c,p.model,p.stats,scenarios()[0],r/'evidence/current_pose_development'),indent=2))
