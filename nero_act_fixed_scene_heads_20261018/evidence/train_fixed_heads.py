"""One declared20-epoch fixed-scene output-head run with immutable provenance."""
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evidence.probe_fixed_heads_v3 import configuration,SOURCE
from executor.bundle import verify_model_bundle
from simulation.refresh_observations import require_fresh_inputs
from training.command_context import ReferenceDataset
from training.integrity import snapshot,require_unchanged
from training.train import FullTaskPolicy,save_checkpoint,trainer
from training.transition_sampling import critical_windows,critical_validation_windows


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    verify_model_bundle(SOURCE)
    probe=json.loads((ROOT/'evidence/fixed_heads_probe_v3.json').read_text())
    # The global1mm fitting diagnostic remains failed; it is not hardware permission.
    assert probe['weights_saved'] is False and probe['frozen_base_and_legacy_gripper_row_passed']
    assert probe['final_l1'] < probe['initial_l1']
    release=[r for r in probe['records'] if r['id']==5100 and r['stage']=='release']
    assert release and all(r['gripper_mae_mm']<1. for r in release)
    assert '\nOK\n' in (ROOT/'evidence/full_suite.log').read_text()
    data=ROOT/'training_data/fixed_v3'
    manifest=json.loads((data/'manifest.json').read_text());split=json.loads((data/'split.json').read_text())
    assert manifest['complete'] and not manifest['rejected'] and len(manifest['episodes'])==14
    paths=[];origins={}
    for row in manifest['episodes']:
        path=Path(row['path']);assert sha(path)==row['sha256'] and sha(row['raw_path'])==row['raw_sha256']
        assert row['replay']['passed'] and row['prefix_contract_audit']['passed'] and row['teacher_success']
        assert path.name==f"episode_{row['id']}.hdf5"
        assert row['full_task_verified']==(row['id']<5100)
        with h5py.File(path,'r') as stream:
            assert stream.attrs['sim'] and stream.attrs['teacher_success']
            assert stream.attrs['task']==row['task']
            assert bool(stream.attrs['full_task_verified'])==row['full_task_verified']
        if row['provenance']:
            origin=Path(row['provenance']['origin'])
            assert origins.get(str(origin),row['split'])==row['split'];origins[str(origin)]=row['split']
            with h5py.File(path,'r') as stream:
                correction=json.loads(stream.attrs['scene_config'])['correction_source']
                assert correction['source_group']==row['source_group'] and correction['split']==row['split']
            for name,key in [('physical_snapshots.json','snapshot_sha256'),('scene_configuration.json','scene_sha256')]:
                assert sha(origin/name)==row['provenance'][key]
            assert sha(origin.parent/'protocol.json')==row['provenance']['protocol_sha256']
        paths.append(path)
    assert len({str(p.resolve()) for p in paths})==14
    assert set(split['train']).isdisjoint(split['val'])
    assert set(split['train']+split['val'])=={r['id'] for r in manifest['episodes']}
    assert all(r['id'] in split[r['split']] for r in manifest['episodes'])
    assert {r['source_group'] for r in manifest['episodes'] if r['split']=='train'}.isdisjoint(
        {r['source_group'] for r in manifest['episodes'] if r['split']=='val'})
    observation_sources=require_fresh_inputs(paths)
    output=ROOT/'training/run27_fixed_heads';output.mkdir(exist_ok=False);(output/'artifacts').mkdir()
    for row in manifest['episodes']:row['status']='valid'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'split.json').write_text(json.dumps(split,indent=2))
    shutil.copy2(SOURCE/'dataset_stats.pkl',output/'frozen_pretraining_stats.pkl')
    config=configuration()
    config.update(num_epochs=20,seed=42,dataset_dir=str(data),manifest=str(output/'manifest.json'),
        split_file=str(output/'split.json'),checkpoint_dir=str(output/'checkpoints'),
        artifact_dir=str(output/'artifacts'),training_log=str(output/'training.csv'),
        normalization_stats=str(output/'frozen_pretraining_stats.pkl'),init_checkpoint=str(SOURCE/'policy.ckpt'),
        training_integrity_required=True,scene_config=json.loads((data/'scene_config.json').read_text()),
        real_initial_state=[*json.loads((data/'scene_config.json').read_text())['initial_robot_state']['qpos_rad'],.095],
        reach_deadline_steps=3900,task_budget_rule='preserve_predeclared_run26_3900_no_extension',
        trainable_parameter_scope='first7_action_head_rows_and_existing_nonlinear_gripper_adapter_only',
        cvae_adaptation_scope='frozen_posterior; output_heads_use_zero_latent_prior',
        normalization_source='frozen_original_training_only_stats; no_new_validation_statistics',
        training_revision='current_fixed_scene_arm_output_and_gripper_v1',hardware_deployment_allowed=False,
        checkpoint_selection='minimum_zero_latent_validation_total_loss',
        data_roles='12_full_8train4val_and2_release_partial_1train1val',
        probe_global_1mm_diagnostic_passed=probe['passed'],
        training_decision='authorized_full_training_same_configuration_after_mechanism_fit; no_fourth_tuning; full14_not_relaxed',
        deployment_scene_binding=None,source_training_scene_config=None)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    sources=[p for folder in ['executor','training','simulation','policy_source'] for p in (ROOT/folder).rglob('*.py')]
    inputs=paths+observation_sources+sources+[
        Path(__file__),ROOT/'evidence/probe_fixed_heads.py',ROOT/'evidence/collect_fixed_data_v3.py',
        ROOT/'evidence/run27_training_pipeline.py',ROOT/'evidence/training_decision_after_probe.json',ROOT/'evidence/probe_fixed_heads_v3.py',ROOT/'evidence/fixed_heads_probe_v3.json',SOURCE/'policy.ckpt',SOURCE/'dataset_stats.pkl',
        data/'manifest.json',data/'split.json',data/'scene_config.json',data/'predeclared.json',
        output/'config.json',output/'manifest.json',output/'split.json',output/'frozen_pretraining_stats.pkl']
    for row in manifest['episodes']:
        if row['provenance']:
            origin=Path(row['provenance']['origin'])
            inputs += [origin/'physical_snapshots.json',origin/'scene_configuration.json',origin.parent/'protocol.json']
    hashes=snapshot(inputs)
    (output/'artifacts/training_inputs_before.json').write_text(json.dumps(hashes,indent=2))
    trainer.NeroActPolicy=FullTaskPolicy;trainer.NeroEpisodicDataset=ReferenceDataset
    trainer.fixed_window_samples=partial(critical_windows,command_context=True)
    trainer.fixed_validation_samples=partial(critical_validation_windows,command_context=True)
    trainer.save_checkpoint=save_checkpoint
    torch.set_num_threads(2)
    trainer.train(config)
    require_unchanged(hashes)
    summary=json.loads((output/'artifacts/training_summary.json').read_text())
    assert summary['last_metrics']['epoch']==20
    original=torch.load(SOURCE/'policy.ckpt',map_location='cpu',weights_only=False)['model'];audits=[]
    for checkpoint in sorted((output/'checkpoints').glob('policy*.ckpt')):
        learned=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
        changed=[k for k,v in learned.items() if not torch.equal(v,original[k])]
        assert changed and all(k.startswith('gripper_adapter.') or k in
            ['model.action_head.weight','model.action_head.bias'] for k in changed)
        assert all(torch.equal(learned[key][7],original[key][7]) for key in
            ['model.action_head.weight','model.action_head.bias'])
        audits.append(dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),changed_tensors=changed))
    (ROOT/'evidence/frozen_heads_audit.json').write_text(json.dumps(dict(passed=True,records=audits),indent=2))
    proof=dict(finalized=True,declared_epochs=20,actual_epochs=20,protected_sha256=hashes,
        output_sha256=snapshot(list((output/'checkpoints').glob('policy*.ckpt'))+
            [output/'checkpoints/dataset_stats.pkl',output/'artifacts/training_summary.json']))
    (output/'artifacts/training_integrity.json').write_text(json.dumps(proof,indent=2))


if __name__=='__main__':main()
