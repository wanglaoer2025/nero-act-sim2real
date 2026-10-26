"""Compare the previous frozen policy on the rehearsal, then run six offline recoveries."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch
from simulation.history_mechanism import load_pilot,run_mechanism
from training.approach_preservation import ApproachPreservation
root=Path(__file__).resolve().parents[1]
settings=json.loads((root/'config/history_experiment.json').read_text())
previous=root.parent/'nero_act_restricted_heads_20261020/training/pilot_A'
policy,stats=load_pilot(previous)
new_stats=pickle.loads((root/'training/pilot_A/checkpoints/dataset_stats.pkl').read_bytes())
assert all(np.array_equal(stats[k],new_stats[k]) for k in ['qpos_mean','qpos_std','action_mean','action_std','geometry_mean','geometry_std'])
keeper=ApproachPreservation(settings['approach_preservation'],stats,'cuda')
result=keeper.assess(policy)
result.update(complete=True,optimizer_updates=1000,training_run=str(previous),
 checkpoint_sha256=hashlib.sha256((previous/'checkpoints/policy_last.ckpt').read_bytes()).hexdigest(),
 rehearsal_sha256=hashlib.sha256(Path(settings['approach_preservation']['replay_path']).read_bytes()).hexdigest(),
 hardware_commands_sent=False)
(root/'evidence/previous_replay_assessment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(previous_replay_assessment=result)),flush=True)
del keeper,policy;torch.cuda.empty_cache()
run_mechanism(root/'training/pilot_A',root/'training_data/recoveries_v5',root/'results/mechanism_preservation')
