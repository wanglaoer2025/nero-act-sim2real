"""Repeat one previously successful development case with frozen ACT inference only."""
import copy,json,pickle
from pathlib import Path
import numpy as np,torch
from executor.bundle import verify_model_bundle
from executor.model import BoundedActPolicy,verify_paired_readout_statistics
from executor.evaluation_integrity import verify_frozen_artifacts
from training.integrity import require_finalized
from simulation.supervised_rollout import rollout
from simulation.evaluate import run_frozen_cases
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'model/run15_first_lift_recovery'
OLD=ROOT/'evaluation/run15_development16'
OUT=ROOT/'evidence/run15_simulation_operation_01'
metadata=verify_model_bundle(MODEL)
payload=torch.load(MODEL/'policy.ckpt',map_location='cpu',weights_only=False)
# This is a repeated known development condition, never a new independent test.
source=json.loads((OLD/'protocol.json').read_text())
case=next(copy.deepcopy(c) for c in source['scenarios'] if c['name']=='last_site_geometry_prepared_open_0')
protocol=dict(source,scenarios=[case],assessment_role='repeat_of_known_successful_development_case',
    demonstration_only=True,not_independent_acceptance=True,real_hardware_tested=False,
    hardware_deployment_allowed=False,real_onsite_permissions_created=False)
require_finalized(ROOT/'training/run15_first_lift_recovery/checkpoints/policy_best_l1.ckpt',payload['config'])
verify_frozen_artifacts(MODEL/'policy.ckpt',protocol)
OUT.mkdir(exist_ok=False);(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
config=dict(payload['config'],device='cuda',pretrained_backbone=False)
with (MODEL/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
verify_paired_readout_statistics(config,stats);torch.set_num_threads(2)
policy=BoundedActPolicy(config).cuda();policy.load_state_dict(payload['model']);policy.eval()
policy.config=config;policy.device=torch.device('cuda')
summary=run_frozen_cases(OUT,protocol,lambda c:rollout(c['scene'],policy,stats,c,OUT/c['name']),
    lambda:verify_frozen_artifacts(MODEL/'policy.ckpt',protocol))
original=np.load(OLD/case['name']/'rollout.npz');repeated=np.load(OUT/case['name']/'rollout.npz')
comparison=dict(original_case=case['name'],success=bool(summary['results'][0]['success']),
    same_states=np.array_equal(original['states'],repeated['states']),
    same_sent_targets=np.array_equal(original['sent_targets'],repeated['sent_targets']),
    hardware_tested=False,independent_acceptance=False)
(OUT/'repeat_comparison.json').write_text(json.dumps(comparison,indent=2));print(json.dumps(comparison),flush=True)
