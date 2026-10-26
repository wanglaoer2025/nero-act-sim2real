import json
from pathlib import Path
import numpy as np
from simulation.history_mechanism import load_pilot
from simulation.supervised_rollout import rollout
from simulation.history_capture import verify_trace_reference
ROOT=Path('/home/user/wq/nero_act_contact_transition_balance_20261020');OUT=Path('/home/user/wq/nero_act_contact_recovery_20261020/evidence/contact_failure_capture')
policy,stats=load_pilot(ROOT/'training/pilot_B');original=ROOT/'results/original_failure_capture'
scene=json.loads((original/'scene_configuration.json').read_text());scenario=json.loads((original/'result.json').read_text())['scenario']
context=json.loads((original/'runtime_checkpoints.json').read_text())['named']['pre_close']
r=rollout(scene,policy,stats,scenario,OUT,resume_context=context,capture_control=True,runtime_profile='history_common_v1',resume_policy_transition=True,mechanism_role='hold_to_lift')
with np.load(OUT/'rollout.npz') as new,np.load(ROOT/'results/mechanism_B/nominal_pre_close/rollout.npz') as old:
 assert np.array_equal(new['raw_chunks'],old['raw_chunks']) and np.array_equal(new['sent_targets'],old['sent_targets'])
records=json.loads((OUT/'runtime_checkpoints.json').read_text())['records'];latest=records[-1]
(Path('/home/user/wq/nero_act_contact_recovery_20261020/evidence')/'partial_contact_context.json').write_text(json.dumps(latest))
print(json.dumps(dict(exact_old_trace_match=True,error=r['error'],active=r['new_active_points'],context_active=latest['control']['active_points'],context_path=str(OUT.parent/'partial_contact_context.json'))),flush=True)
