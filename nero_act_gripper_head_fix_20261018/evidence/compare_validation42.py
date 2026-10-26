"""Frozen zero-latent comparison on the previously declared validation42 scene."""
import copy,hashlib,json,pickle
from pathlib import Path
import h5py,torch
from executor.bundle import verify_model_bundle
from executor.model import BoundedActPolicy,verify_paired_readout_statistics
from executor.evaluation_integrity import evaluation_sources,verify_frozen_artifacts,SUPERVISED_PROTOCOL
from simulation.evaluate import run_frozen_cases
from simulation.supervised_rollout import rollout

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'nero_act_pose_objective_fix_20261018'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    output=ROOT/'evaluation/run23_validation42';output.mkdir(exist_ok=False)
    source=ROOT.parent/'nero_act_grasp_state_recovery_20261018/evaluation/run19_grasp_sources/protocol.json'
    cases=[copy.deepcopy(c) for c in json.loads(source.read_text())['scenarios'] if c['name']=='validation42_grasp']
    assert len(cases)==1
    directory=ROOT/'model/run23_gripper_head_only';meta=verify_model_bundle(directory)
    b=torch.load(directory/'policy.ckpt',map_location='cpu',weights_only=False);c=dict(b['config'],device='cuda',pretrained_backbone=False)
    with (directory/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
    verify_paired_readout_statistics(c,stats);torch.set_num_threads(2)
    p=BoundedActPolicy(c).cuda();p.load_state_dict(b['model']);p.eval();p.config=c;p.device=torch.device('cuda')
    protocol=dict(checkpoint_sha256=meta['checkpoint_sha256'],stats_sha256=meta['stats_sha256'],
        scene_sha256=meta['scene_sha256'],training_config_sha256=meta['training_config_sha256'],
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in evaluation_sources()},scenarios=cases,
        teacher_free=True,real_hardware_tested=False,hardware_deployment_allowed=False,
        supervised_contact_protocol=SUPERVISED_PROTOCOL,assessment_role='same_validation42_development_comparison_not_independent_acceptance',source_protocol_sha256=sha(source))
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    run_frozen_cases(output,protocol,lambda case:rollout(case['scene'],p,stats,case,output/case['name']),
        lambda:verify_frozen_artifacts(directory/'policy.ckpt',protocol))

if __name__=='__main__':main()
