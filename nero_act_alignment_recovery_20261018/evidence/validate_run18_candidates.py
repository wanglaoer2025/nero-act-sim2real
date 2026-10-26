"""Task validation on held-out full demonstration scenes; not independent acceptance."""
import hashlib,json,pickle
from pathlib import Path
import h5py,torch
from executor.bundle import verify_model_bundle
from executor.model import BoundedActPolicy,verify_paired_readout_statistics
from executor.evaluation_integrity import evaluation_sources,verify_frozen_artifacts,SUPERVISED_PROTOCOL
from simulation.evaluate import run_frozen_cases
from simulation.supervised_rollout import rollout

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/run18_validation_candidates'
MODELS=['run18_alignment_recovery','run18_last_validation_candidate']
IDS=[36,37]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'training/run18_alignment_recovery/manifest.json').read_text())
    rows=[next(r for r in manifest['episodes'] if r['id']==i) for i in IDS]
    assert all(r['split']=='val' and r['full_task_verified'] for r in rows)
    cases=[]
    for row in rows:
        assert sha(Path(row['path']))==row['sha256']
        with h5py.File(row['path']) as f:c=json.loads(f.attrs['scene_config'])
        assert 'initial_simulation_snapshot' not in c
        cases.append(dict(name=f"heldout_full_scene_{row['id']}",scene=c,seed=940000+row['id'],
            query_stride=10,servo_delay_steps=c.get('action_delay_steps',0),observation_age_s=.05,jitter=False,force=False))
    (OUT/'predeclared_comparison.json').write_text(json.dumps(dict(candidates=MODELS,validation_ids=IDS,
        selection='most complete physical successes, then fewer unsafe stops, then validation L1',
        no_independent_test_cases_read=True),indent=2))
    reports=[];torch.set_num_threads(2)
    for name in MODELS:
        directory=ROOT/'model'/name;metadata=verify_model_bundle(directory)
        b=torch.load(directory/'policy.ckpt',map_location='cpu',weights_only=False)
        c=dict(b['config'],device='cuda',pretrained_backbone=False)
        with (directory/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
        verify_paired_readout_statistics(c,stats);p=BoundedActPolicy(c).cuda();p.load_state_dict(b['model']);p.eval()
        p.config=c;p.device=torch.device('cuda');output=OUT/name;output.mkdir()
        protocol=dict(checkpoint_sha256=metadata['checkpoint_sha256'],stats_sha256=metadata['stats_sha256'],
            scene_sha256=metadata['scene_sha256'],training_config_sha256=metadata['training_config_sha256'],
            source_sha256={str(f.relative_to(ROOT)):sha(f) for f in evaluation_sources()},scenarios=cases,
            teacher_free=True,real_hardware_tested=False,hardware_deployment_allowed=False,
            supervised_contact_protocol=SUPERVISED_PROTOCOL,assessment_role='heldout_validation_checkpoint_diagnostic')
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
        s=run_frozen_cases(output,protocol,lambda case:rollout(case['scene'],p,stats,case,output/case['name']),
            lambda:verify_frozen_artifacts(directory/'policy.ckpt',protocol))
        reports.append(dict(model=name,epoch=b['epoch'],passed=s['passed'],total=s['total'],
            val_l1=b['metrics']['val_l1'],errors=[r['error'] for r in s['results']],
            events=[r['simulated_events'] for r in s['results']]))
        (OUT/'comparison.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports[-1]),flush=True)
        del p

if __name__=='__main__':main()
