"""Three new diagnostic contexts for supervised-release data; never hardware eligible."""
import hashlib,json,pickle
from pathlib import Path
import torch
from simulation import evaluate
from simulation.supervised_rollout import rollout
from executor.model import BoundedActPolicy,verify_paired_readout_statistics
from executor.bundle import verify_model_bundle
from executor.evaluation_integrity import evaluation_sources,verify_frozen_artifacts,SUPERVISED_PROTOCOL
root=Path(__file__).resolve().parents[1];directory=root/'model/candidate_run9_target_recovery'
output=root/'evidence/run9_supervised_recovery_diagnostics';output.mkdir(exist_ok=False)
meta=verify_model_bundle(directory);bundle=torch.load(directory/'policy.ckpt',map_location='cpu',weights_only=False)
config=dict(bundle['config'],device='cuda',pretrained_backbone=False)
with (directory/'dataset_stats.pkl').open('rb') as stream:stats=pickle.load(stream)
verify_paired_readout_statistics(config,stats);torch.set_num_threads(2)
policy=BoundedActPolicy(config).cuda();policy.load_state_dict(bundle['model'],strict=True);policy.eval()
policy.config,policy.device=config,torch.device('cuda');scene=evaluate.evaluation_scene(config)
cases=[c for c in evaluate.scenarios(9000) if c['name'] in ['independent_3','independent_6','delay_500ms']]
protocol=dict(meta,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in evaluation_sources()},
 scenarios=cases,supervised_contact_protocol=SUPERVISED_PROTOCOL,diagnostic_only=True,hardware_deployment_allowed=False)
(output/'protocol.json').write_text(json.dumps(protocol,indent=2));verify_frozen_artifacts(directory/'policy.ckpt',protocol)
results=[]
for case in cases:
 results.append(rollout(scene,policy,stats,case,output/case['name']));print(json.dumps(results[-1]),flush=True)
verify_frozen_artifacts(directory/'policy.ckpt',protocol)
(output/'summary.json').write_text(json.dumps(dict(results=results,diagnostic_only=True,hardware_deployment_allowed=False),indent=2))
