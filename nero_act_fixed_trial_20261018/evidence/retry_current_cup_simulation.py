"""Diagnostic current-cup simulation only; does not authorize hardware."""
from pathlib import Path
import copy,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from executor.policy_worker import FrozenAct
from executor.fixed_permission import MODEL
from simulation.supervised_rollout import rollout
out=ROOT/'evidence/retry_current_cup_sim_01';out.mkdir(exist_ok=False)
scene=json.loads((MODEL/'scene_config.json').read_text())
fit=json.loads((ROOT/'evidence/retry_restored_geometry_01/report.json').read_text())
health=json.loads((ROOT/'evidence/retry_request_restored_health_01.json').read_text())
scene['pick_xy_m']=fit['measured']['cup_xyz'][:2]
scene['place_xy_m']=fit['measured']['target_xyz'][:2]
scene['initial_robot_state']['qpos_rad']=health['actual'][:7]
scene['initial_robot_state']['gripper_total_width_m']=.095
case=dict(name='current_cup_offset_27mm_diagnostic',seed=2026101901,query_stride=10,
          servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False)
protocol=dict(scene=scene,scenario=case,checkpoint=str(MODEL/'policy.ckpt'),hardware_connected=False,
              real_hardware_authorization=False,replaces_original_evaluation=False)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
frozen=FrozenAct(MODEL,device='cuda');policy=frozen.model
policy.config=frozen.config;policy.device='cuda'
result=rollout(scene,policy,frozen.stats,case,out/'rollout')
(out/'summary.json').write_text(json.dumps(dict(result=result,hardware_connected=False,
 real_hardware_authorization=False,replaces_original_evaluation=False),indent=2))
print(json.dumps(result),flush=True)
