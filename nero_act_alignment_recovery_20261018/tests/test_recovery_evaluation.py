import json,unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class RecoveryEvaluationTests(unittest.TestCase):
    def test_fresh_cases_change_actual_observations_and_dynamics_not_only_seed(self):
        from simulation import evaluate_generalized as module
        self.assertTrue(hasattr(module,'fresh_test_cases'),'a reused failed-case suite cannot be independent acceptance')
        center=json.loads((ROOT/'scene/center.json').read_text())
        site=json.loads((ROOT/'evidence/site_snapshot.json').read_text())
        old=module.test_cases(center,site);fresh=module.fresh_test_cases(center,site)
        self.assertEqual(len(fresh),16)
        self.assertEqual(len({c['name'] for c in fresh}),16)
        for before,after in zip(old,fresh):
            self.assertNotEqual(before['seed'],after['seed'])
            self.assertFalse(np.array_equal(before['scene']['initial_robot_state']['qpos_rad'],
                after['scene']['initial_robot_state']['qpos_rad']))
            self.assertNotEqual(before['scene']['appearance'],after['scene']['appearance'])
            self.assertIn(after['servo_delay_steps'],[0,1,2])
        self.assertEqual(fresh,module.fresh_test_cases(center,site))
        for c in fresh[:13]:module.validate_initial_pose(module.build_scene(c['scene']),c['scene'],center)

    def test_saved_independent_definition_is_hash_bound_and_loaded_exactly(self):
        import hashlib,tempfile
        from simulation import evaluate_generalized as module
        import training.generalized as training
        self.assertTrue(hasattr(module,'load_predeclared_cases'))
        self.assertTrue(hasattr(training,'bind_independent_cases'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'declared.json';cases=[dict(name=f'fresh_{i}',scene=dict(marker=i)) for i in range(16)]
            path.write_text(json.dumps(cases));config=training.bind_independent_cases(path)
            self.assertEqual(config['independent_cases_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(module.load_predeclared_cases(config),cases)
            path.write_text(json.dumps(list(reversed(cases))))
            with self.assertRaises(ValueError):module.load_predeclared_cases(config)

    def test_recovery_config_forces_old_cases_to_development_including_default_and_nominal(self):
        from simulation import evaluate_generalized as module
        self.assertTrue(hasattr(module,'assessment_role'))
        recovery=dict(prior_evaluation_scope='development_regression_after_failed_state_corrections')
        self.assertEqual(module.assessment_role(recovery),'development_regression_used_for_corrections')
        self.assertEqual(module.assessment_role(recovery,development=True),'development_regression_used_for_corrections')
        self.assertEqual(module.assessment_role({}),'original_run14_test_definition')
