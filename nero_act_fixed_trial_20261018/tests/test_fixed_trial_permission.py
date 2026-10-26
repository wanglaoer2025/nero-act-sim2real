import copy
import importlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'nero_act_fixed_scene_heads_20261018'


class FixedTrialPermissionTests(unittest.TestCase):
    def fixture(self):
        meta=json.loads((SOURCE/'model/run27_fixed_heads/runtime_metadata.json').read_text())
        evaluation=json.loads((SOURCE/'evaluation/run27_current_scene14/summary.json').read_text())
        receipt=json.loads((ROOT/'fixed_trial/receipt.json').read_text())
        self.now=receipt['authorized_wall_s']+1
        return meta,evaluation,receipt

    def checker(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.fixed_permission'),
            'a separately bound fixed-trial permission checker is required')
        return importlib.import_module('executor.fixed_permission').require_fixed_trial_permission

    def test_exact_acknowledged_candidate_may_start_trial_but_not_general_deployment(self):
        check=self.checker();meta,evaluation,receipt=self.fixture()
        check(meta,evaluation,receipt,now=self.now)
        from executor.permission import require_experimental_permission
        with self.assertRaises(PermissionError):
            require_experimental_permission(meta,evaluation,receipt,now=self.now)

    def test_wrong_model_changed_evaluation_missing_ack_or_expired_receipt_is_denied(self):
        check=self.checker();meta,evaluation,receipt=self.fixture()
        for change in [dict(note='a second receipt must not create another trial'),dict(confirmation='ACT_FULL'),dict(accepted_incomplete_evaluation=False),
                       dict(operator_at_estop=False),dict(sole_control_confirmed=False),
                       dict(one_physical_trial_only=False),dict(general_deployment_accepted=True),
                       dict(proposal_sha256='wrong'),dict(checkpoint_sha256='wrong'),
                       dict(speed_percent=10),dict(speed_percent=5.0),dict(gripper_force_limit_n=2.)]:
            with self.subTest(change=change),self.assertRaises(PermissionError):
                check(meta,evaluation,dict(receipt,**change),now=self.now)
        with self.assertRaises(PermissionError):check(meta,evaluation,receipt,now=self.now+2000)
        for record,change in [(meta,dict(checkpoint_sha256='wrong')),
                              (evaluation,dict(passed=14)),(evaluation,dict(finalized=False))]:
            with self.subTest(change=change),self.assertRaises(PermissionError):
                check(dict(meta,**change) if record is meta else meta,
                      dict(evaluation,**change) if record is evaluation else evaluation,receipt,now=self.now)

    def test_nominal_or_safety_failure_cannot_be_relabelled_as_approved_evaluation(self):
        check=self.checker();meta,evaluation,receipt=self.fixture()
        for field,value in [('success',False),('physical_success',False),
                            ('unexpected_arm_cup_contact_frames',1)]:
            changed=copy.deepcopy(evaluation);changed['results'][0][field]=value
            with self.subTest(field=field),self.assertRaises(PermissionError):
                check(meta,changed,receipt,now=self.now)


if __name__=='__main__':unittest.main()
