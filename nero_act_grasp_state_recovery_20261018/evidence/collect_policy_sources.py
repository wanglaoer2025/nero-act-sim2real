"""Record frozen ACT-visited states for the declared low-grasp correction batch."""
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
    output=ROOT/'evaluation/run19_grasp_sources';output.mkdir(exist_ok=False)
    cases=[copy.deepcopy(json.loads((OLD/'evaluation/run19_nominal/protocol.json').read_text())['scenarios'][0])]
    rows=json.loads((OLD/'training/run19_pose_objective/manifest.json').read_text())['episodes']
    for index in [41,42]:
        row=next(r for r in rows if r['id']==index);assert row['split']=='val'
        assert sha(Path(row['path']))==row['sha256']
        with h5py.File(row['path']) as f:scene=json.loads(f.attrs['scene_config'])
        cases.append(dict(name=f'validation{index}_grasp',scene=scene,seed=950000+index,
            query_stride=10,servo_delay_steps=scene.get('action_delay_steps',0),observation_age_s=.05,jitter=False,force=False))
    directory=OLD/'model/run19_pose_objective';meta=verify_model_bundle(directory)
    b=torch.load(directory/'policy.ckpt',map_location='cpu',weights_only=False);c=dict(b['config'],device='cuda',pretrained_backbone=False)
    with (directory/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
    verify_paired_readout_statistics(c,stats);torch.set_num_threads(2)
    p=BoundedActPolicy(c).cuda();p.load_state_dict(b['model']);p.eval();p.config=c;p.device=torch.device('cuda')
    protocol=dict(checkpoint_sha256=meta['checkpoint_sha256'],stats_sha256=meta['stats_sha256'],
        scene_sha256=meta['scene_sha256'],training_config_sha256=meta['training_config_sha256'],
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in evaluation_sources()},scenarios=cases,
        teacher_free=True,real_hardware_tested=False,hardware_deployment_allowed=False,
        supervised_contact_protocol=SUPERVISED_PROTOCOL,assessment_role='development_training_or_validation_source_states_not_acceptance')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    run_frozen_cases(output,protocol,lambda case:rollout(case['scene'],p,stats,case,output/case['name']),
        lambda:verify_frozen_artifacts(directory/'policy.ckpt',protocol))

if __name__=='__main__':main()
