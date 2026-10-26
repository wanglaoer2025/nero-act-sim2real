import json,pickle,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import torch

ROOT=Path(__file__).resolve().parents[1]

class TrainingReviewFixes(unittest.TestCase):
    def test_interrupted_candidate_consumes_its_slot_without_blocking_next_candidate(self):
        from simulation.collect_generalized import collect_one
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);old=out/'attempts/scene_00_candidate_0';old.mkdir(parents=True)
            keep=old/'original.txt';keep.write_text('interrupted evidence')
            with patch('simulation.collect_generalized.episode_config',return_value={}) as configure,patch('simulation.collect_generalized.collect_episode',side_effect=RuntimeError('TEST_REJECT')):
                try:result=collect_one({},0,out)
                except FileExistsError:self.fail('Interrupted candidate blocks safe resume')
            self.assertIsNone(result['accepted'])
            self.assertEqual([c.args[3] for c in configure.call_args_list],[1,2])
            self.assertEqual(keep.read_text(),'interrupted evidence')

    def test_verified_hdf_and_canonical_copy_recover_after_manifest_commit_was_interrupted(self):
        from simulation.collect_generalized import collect_one
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);folder=out/'attempts/scene_00_candidate_0';folder.mkdir(parents=True)
            source=folder/'episode_0.hdf5';scene=dict(generalization=dict(index=0,split='train',source_group='scene_0'))
            with h5py.File(source,'w') as f:
                f.attrs.update(sim=True,teacher_success=True,full_task_verified=True,task='complete_pick_carry_place_retreat',scene_config=json.dumps(scene))
            source.with_suffix('.result.json').write_text(json.dumps(dict(success=True,full_task_verified=True,frames=2)))
            canonical=out/'episode_0.hdf5';shutil.copy2(source,canonical);before=canonical.read_bytes()
            with patch('simulation.collect_generalized.collect_episode') as collect,patch('simulation.collect_generalized.verify_replay',return_value=dict(passed=True)) as replay,patch('simulation.collect_generalized.verify_action_contract',return_value=dict(passed=True),create=True):
                try:result=collect_one({},0,out)
                except FileExistsError:self.fail('Verified completed output cannot recover its manifest entry')
            self.assertEqual(result['accepted']['id'],0);collect.assert_not_called();replay.assert_called_once()
            self.assertEqual(canonical.read_bytes(),before)

    def test_filename_id_must_match_manifest_id_before_split_selection(self):
        from training.generalized import validate_manifest
        rows=[dict(id=i,path=f'/tmp/episode_{i}.hdf5',status='valid',full_task_verified=True,
                   teacher_success=True,split='train' if i<36 else 'val',source_group=f'scene{i}',
                   replay=dict(passed=True),prefix_contract_audit=dict(passed=True)) for i in range(48)]
        rows[0]['path'],rows[36]['path']=rows[36]['path'],rows[0]['path']
        with self.assertRaises(ValueError):validate_manifest(dict(episodes=rows,rejected=[]),dict(train=list(range(36)),val=list(range(36,48))),check_files=False)

    def test_completed_epochs_alone_cannot_export_a_generalized_run(self):
        from training.export import export_model
        config=json.loads((ROOT/'initializers/run13/training_config.json').read_text())
        config.update(num_epochs=1,training_integrity_required=True,training_revision='position_generalization_causal_context_v1')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ckpt=root/'checkpoints/policy_best_val.ckpt';ckpt.parent.mkdir()
            (root/'artifacts').mkdir();(root/'artifacts/training_summary.json').write_text(json.dumps(dict(last_metrics=dict(epoch=1))))
            torch.save(dict(config=config,model={},epoch=1),ckpt)
            with (ckpt.parent/'dataset_stats.pkl').open('wb') as f:pickle.dump(config['constraint_loss_stats'],f)
            with self.assertRaises(ValueError):export_model(ckpt,root/'exported')
            self.assertFalse((root/'exported').exists())

    def test_changed_training_source_or_input_fails_integrity_verification(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('training.integrity'))
        from training.integrity import snapshot,require_unchanged
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.py';p.write_text('original')
            hashes=snapshot([p]);require_unchanged(hashes)
            p.write_text('changed')
            with self.assertRaises(RuntimeError):require_unchanged(hashes)
