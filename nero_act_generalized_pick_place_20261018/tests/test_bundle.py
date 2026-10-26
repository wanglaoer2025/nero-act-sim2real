import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class BundleTests(unittest.TestCase):
    def test_export_binds_completed_training_and_keeps_hardware_disabled(self):
        self.assertIsNotNone(importlib.util.find_spec('training.export'), 'completed model export missing')
        from training.export import export_model
        from executor.policy_worker import verify_model_bundle
        import pickle
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'run'
            (run/'checkpoints').mkdir(parents=True)
            (run/'artifacts').mkdir()
            checkpoint = run/'checkpoints/policy_best_val.ckpt'
            config = dict(chunk_size=20, state_dim=8, action_dim=8, num_epochs=1,
                          task_scope='complete_pick_carry_place_retreat', bounded_gripper_output=True)
            config['scene_config'] = dict(initial_robot_state={'qpos_rad': [.123]*7},
                                        tcp_link7_xyz_m=[.152, 0, -.0235], table_top_z_m=0)
            torch.save(dict(epoch=1, config=config, model={}), checkpoint)
            with (checkpoint.parent/'dataset_stats.pkl').open('wb') as f:
                pickle.dump(dict(qpos_mean=np.zeros(8), qpos_std=np.ones(8),
                                 action_mean=np.zeros(8), action_std=np.ones(8)), f)
            target = Path(directory)/'export'
            with self.assertRaises(ValueError):
                export_model(checkpoint, target)
            (run/'artifacts/training_summary.json').write_text(json.dumps(dict(last_metrics={'epoch': 1})))
            export_model(checkpoint, target)
            meta = verify_model_bundle(target)
            self.assertFalse(meta['hardware_deployment_allowed'])
            self.assertEqual(meta['selected_epoch'], 1)
            self.assertEqual(meta['initial_state'], config['scene_config']['initial_robot_state'])
            (target/'scene_config.json').write_text('{}')
            with self.assertRaises(ValueError):
                verify_model_bundle(target)

    def test_mismatched_statistics_and_wrong_task_are_rejected_before_model_load(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.policy_worker'), 'sealed policy loader missing')
        from executor.policy_worker import verify_model_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'policy.ckpt').write_bytes(b'weight placeholder')
            (root/'dataset_stats.pkl').write_bytes(b'stat placeholder')
            meta = dict(task_scope='complete_pick_carry_place_retreat', action_chunk_shape=[20, 8],
                        checkpoint_sha256=hashlib.sha256(b'weight placeholder').hexdigest(),
                        stats_sha256=hashlib.sha256(b'stat placeholder').hexdigest(), source_sha256={})
            (root/'runtime_metadata.json').write_text(json.dumps(meta))
            self.assertEqual(verify_model_bundle(root)['action_chunk_shape'], [20, 8])
            (root/'dataset_stats.pkl').write_bytes(b'wrong statistics')
            with self.assertRaises(ValueError):
                verify_model_bundle(root)
            (root/'dataset_stats.pkl').write_bytes(b'stat placeholder')
            meta['task_scope'] = 'aligned_noncontact_air_sequence'
            (root/'runtime_metadata.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                verify_model_bundle(root)


if __name__ == '__main__':
    unittest.main()
