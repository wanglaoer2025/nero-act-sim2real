"""Single-factor diagnostic of old independent failure; never independent acceptance."""
import copy,hashlib,json,pickle
from pathlib import Path
import numpy as np,torch
from executor.bundle import verify_model_bundle
from executor.model import BoundedActPolicy,verify_paired_readout_statistics
from executor.evaluation_integrity import evaluation_sources,verify_frozen_artifacts
from simulation.supervised_rollout import rollout
from simulation.evaluate import run_frozen_cases
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'nero_act_contact_recovery_20261018'
MODEL=ROOT/'model/run15';OUT=ROOT/'evidence/run15_gray_table_ablation';OUT.mkdir(exist_ok=False)
source=json.loads((OLD/'evaluation/run15_independent16/protocol.json').read_text())
case=copy.deepcopy(next(c for c in source['scenarios'] if c['name']=='fresh_cup_+0_+0'))
original_table=case['scene']['appearance']['table_rgb'];case['scene']['appearance']['table_rgb']=[float(np.mean(original_table))]*3
case['name']='old_fresh_nominal_gray_table_only'
source.update(scenarios=[case],assessment_role='single_factor_diagnostic_not_independent',
 source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in evaluation_sources()},
 hardware_deployment_allowed=False,real_hardware_tested=False,original_table_rgb=original_table)
verify_model_bundle(MODEL);verify_frozen_artifacts(MODEL/'policy.ckpt',source)
p=torch.load(MODEL/'policy.ckpt',map_location='cpu',weights_only=False);c=dict(p['config'],device='cuda',pretrained_backbone=False)
with (MODEL/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
verify_paired_readout_statistics(c,stats);torch.set_num_threads(2)
policy=BoundedActPolicy(c).cuda().eval();policy.load_state_dict(p['model']);policy.config=c;policy.device=torch.device('cuda')
(OUT/'protocol.json').write_text(json.dumps(source,indent=2))
s=run_frozen_cases(OUT,source,lambda item:rollout(item['scene'],policy,stats,item,OUT/item['name']),
 lambda:verify_frozen_artifacts(MODEL/'policy.ckpt',source))
r=s['results'][0];print(json.dumps(dict(success=r['success'],error=r['error'],max_lift_m=r['max_lift_m'],
 transfer_m=r['maximum_horizontal_transfer_m'],hardware_tested=False)),flush=True)
