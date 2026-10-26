import importlib.util
import unittest
import hashlib
import json
from pathlib import Path
import tempfile


class EvaluationTests(unittest.TestCase):
    def test_failed_final_seal_never_leaves_an_eligible_complete_summary(self):
        from simulation import evaluate
        self.assertTrue(callable(getattr(evaluate, 'run_frozen_cases', None)),
                        'evaluation must finalize only after integrity verification')
        from executor.evaluation_integrity import verify_frozen_artifacts
        for changed in ['dataset_stats.pkl', 'dependency.py', 'policy.ckpt',
                        'scene_config.json', 'training_config.json']:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory);model = root/'model';model.mkdir()
                protocol = dict(scenarios=[{'name': 'test'}], source_sha256={})
                for name, key in [('policy.ckpt', 'checkpoint_sha256'),
                                  ('dataset_stats.pkl', 'stats_sha256'),
                                  ('scene_config.json', 'scene_sha256'),
                                  ('training_config.json', 'training_config_sha256')]:
                    (model/name).write_text(name)
                    protocol[key] = hashlib.sha256((model/name).read_bytes()).hexdigest()
                dependency = root/'dependency.py';dependency.write_text('original')
                protocol['source_sha256']['dependency.py'] = hashlib.sha256(dependency.read_bytes()).hexdigest()
                output = root/'result';output.mkdir()
                def case(_):
                    target = dependency if changed == 'dependency.py' else model/changed
                    target.write_text('changed during evaluation')
                    return dict(success=True)
                with self.assertRaises(RuntimeError):
                    evaluate.run_frozen_cases(output, protocol, case,
                        lambda: verify_frozen_artifacts(model/'policy.ckpt', protocol, source_root=root))
                summary = json.loads((output/'summary.json').read_text())
                self.assertFalse(summary['complete']);self.assertFalse(summary['finalized'])
                self.assertTrue(summary['integrity_error'])

    def test_only_successful_final_integrity_check_marks_evaluation_finalized(self):
        from simulation import evaluate
        self.assertTrue(callable(getattr(evaluate, 'run_frozen_cases', None)))
        with tempfile.TemporaryDirectory() as directory:
            called = []
            evaluate.run_frozen_cases(Path(directory), dict(scenarios=[{'name': 'test'}]),
                lambda _: dict(success=True), lambda: called.append('verified'))
            result = json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(called, ['verified']);self.assertTrue(result['complete'])
            self.assertTrue(result['finalized'])
    def test_frozen_protocol_binds_servo_camera_geometry_and_network_dependencies(self):
        from simulation import evaluate
        self.assertTrue(callable(getattr(evaluate, 'evaluation_sources', None)))
        names = {str(p.relative_to(evaluate.ROOT)) for p in evaluate.evaluation_sources()}
        self.assertTrue({'simulation/nero_domain_randomization.py', 'simulation/nero_scene_camera.py',
                         'simulation/scene_template.xml', 'simulation/snapshot.py'} <= names)
        self.assertTrue(any(name.startswith('simulation/assets/') for name in names))

    def test_checkpoint_scene_overrides_historical_geometry_and_start(self):
        from simulation import evaluate
        self.assertTrue(callable(getattr(evaluate, 'evaluation_scene', None)))
        scene = {'initial_robot_state': {'qpos_rad': [.123]*7}, 'pick_xy_m': [-.525, .015]}
        self.assertEqual(evaluate.evaluation_scene({'scene_config': scene}), scene)

    def test_fresh_cohort_changes_seeds_without_changing_acceptance_cases(self):
        from simulation.evaluate import scenarios
        original = scenarios()
        fresh = scenarios(seed_offset=1000)
        self.assertEqual([x['name'] for x in fresh], [x['name'] for x in original])
        self.assertTrue(set(x['seed'] for x in original).isdisjoint(x['seed'] for x in fresh))
        for old, new in zip(original, fresh):
            self.assertEqual(dict(old, seed=old['seed']+1000), new)

    def test_protocol_uses_independent_seeds_and_includes_realistic_cadence(self):
        self.assertIsNotNone(importlib.util.find_spec('simulation.evaluate'), 'independent ACT evaluator missing')
        from simulation.evaluate import scenarios
        cases = scenarios()
        self.assertEqual(len(cases), 14)
        independent = [x for x in cases if x['name'].startswith('independent_')]
        self.assertEqual(len(independent), 8)
        self.assertTrue(all(x['seed'] >= 90000 for x in independent))
        self.assertTrue(any(x['query_stride'] == 11 for x in cases))
        self.assertTrue(any(x['servo_delay_steps'] == 10 for x in cases))


if __name__ == '__main__':
    unittest.main()
