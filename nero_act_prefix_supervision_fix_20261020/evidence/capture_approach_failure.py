import copy,json
from pathlib import Path
import numpy as np,mujoco
from simulation.history_mechanism import load_pilot
from simulation.supervised_rollout import rollout
from simulation.scene import build_scene
from simulation.teacher import observe
ROOT=Path('/home/user/wq/nero_act_prefix_supervision_fix_20261020');OUT=Path('/home/user/wq/nero_act_approach_recovery_20261020/evidence/approach_failure_capture_v2')
policy,stats=load_pilot(ROOT/'training/pilot_B')
scene=copy.deepcopy(policy.config['scene_config'])
case=dict(name='nominal_development',seed=191899,query_stride=2,servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False,scene=scene)
# Read-only observer of the existing complete checkpoint, retaining one safe frame.
import simulation.runtime_checkpoint as checkpoints
original_capture=checkpoints.capture_runtime
safe=[]
def retain_safe(model,data,servo,previous,**kwargs):
 c=original_capture(model,data,servo,previous,**kwargs)
 m=observe(model,data,scene)
 if m['tcp_xyz'][2]>=.16 and c['control']['active_points']>100 and not m['unexpected_contact']:
  safe[:]=[(c,m)]
 return c
checkpoints.capture_runtime=retain_safe
r=rollout(scene,policy,stats,case,OUT,capture_control=True,runtime_profile='history_common_v1')
with np.load(OUT/'rollout.npz') as new,np.load(ROOT/'results/pilot_B_nominal_diagnostic/nominal_development/rollout.npz') as old:
 assert np.array_equal(new['raw_chunks'],old['raw_chunks']) and np.array_equal(new['sent_targets'],old['sent_targets'])
candidates=safe
assert candidates
c,m=candidates[-1];p=OUT.parent/'safe_approach_context.json';p.write_text(json.dumps(c))
print(json.dumps(dict(exact_trace_match=True,failed_at=r['active_policy_points'],safe_active=c['control']['active_points'],safe_measurement=m,context=str(p))),flush=True)
