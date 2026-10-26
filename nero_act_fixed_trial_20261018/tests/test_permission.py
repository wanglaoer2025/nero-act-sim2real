import copy
import importlib.util
import unittest
import hashlib
from pathlib import Path
from unittest.mock import patch


class PermissionTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve()
        self.sources = {'tests/test_permission.py': hashlib.sha256(source.read_bytes()).hexdigest()}
        mocked = patch('executor.permission.evaluation_sources', return_value=[source], create=True)
        mocked.start();self.addCleanup(mocked.stop)

    def fixture(self):
        from executor.permission import CASES, SCOPE
        metadata = dict(task_scope=SCOPE, checkpoint_sha256='abc', action_chunk_shape=[20, 8],
                        scene_sha256='scene', stats_sha256='stats', training_config_sha256='config')
        evaluation = dict(task_scope=SCOPE, checkpoint_sha256='abc', stats_sha256='stats',
            scene_sha256='scene', training_config_sha256='config', teacher_free=True,
            complete=True, finalized=True, total=14, passed=14, source_sha256=self.sources,
            supervised_contact_protocol='physics_pauses_fine_grasp_support_hold_v7',
            results=[dict(success=True, physical_success=True, error=None, teacher_free=True,
                premature_release_before_support=False, constraint_attachment_used=False,
                execution_contract_failures=0, unexpected_arm_cup_contact_frames=0,
                supervision_failures=0,
                scenario={'name': name}) for name in CASES])
        receipt = dict(scope=SCOPE, checkpoint_sha256='abc', stats_sha256='stats',
            training_config_sha256='config', confirmation='ACT_FULL', consumed=True,
            authorized_wall_s=100., experimental_full_task_trial=True,
            scene_sha256='scene', operator_at_estop=True, empty_cup=True,
            swept_volume_clear=True, sole_control_confirmed=True)
        return metadata, evaluation, receipt

    def test_evaluated_statistics_scene_and_config_must_match_runtime_and_receipt(self):
        from executor.permission import require_experimental_permission
        metadata, evaluation, receipt = self.fixture()
        require_experimental_permission(metadata, evaluation, receipt, now=101.)
        for key in ['stats_sha256', 'scene_sha256', 'training_config_sha256']:
            for record in [evaluation, receipt]:
                with self.subTest(key=key, record=record is evaluation):
                    mutated = dict(record, **{key: 'different'})
                    with self.assertRaises(PermissionError):
                        require_experimental_permission(metadata,
                            mutated if record is evaluation else evaluation,
                            mutated if record is receipt else receipt, now=101.)

    def test_incomplete_integrity_and_missing_physical_evidence_cannot_authorize(self):
        from executor.permission import require_experimental_permission
        metadata, evaluation, receipt = self.fixture()
        for change in [dict(finalized=False), dict(source_sha256={}),dict(supervised_contact_protocol=None)]:
            with self.assertRaises(PermissionError):
                require_experimental_permission(metadata, dict(evaluation, **change), receipt, now=101.)
        for key, value in [('physical_success', False), ('premature_release_before_support', True),
                           ('constraint_attachment_used', True)]:
            broken = copy.deepcopy(evaluation);broken['results'][0][key] = value
            with self.assertRaises(PermissionError):
                require_experimental_permission(metadata, broken, receipt, now=101.)

    def test_user_token_cannot_override_failed_or_wrong_model_evaluation(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.permission'), 'scoped execution permission guard missing')
        from executor.permission import require_experimental_permission
        metadata, evaluation, receipt = self.fixture()
        require_experimental_permission(metadata, evaluation, receipt, now=101.)
        with self.assertRaises(PermissionError):
            require_experimental_permission(metadata, evaluation,
                                            dict(receipt, operator_at_estop=False), now=101.)
        broken = copy.deepcopy(evaluation)
        broken['results'][0]['success'] = False
        for record in [dict(evaluation, passed=12), broken, dict(evaluation, checkpoint_sha256='wrong')]:
            with self.assertRaises(PermissionError):
                require_experimental_permission(metadata, record, receipt, now=101.)
        with self.assertRaises(PermissionError):
            require_experimental_permission(metadata, evaluation, receipt, now=2000.)
        duplicate = copy.deepcopy(evaluation)
        duplicate['results'][-1]['scenario']['name'] = duplicate['results'][0]['scenario']['name']
        with self.assertRaises(PermissionError):
            require_experimental_permission(metadata, duplicate, receipt, now=101.)


if __name__ == '__main__':
    unittest.main()
