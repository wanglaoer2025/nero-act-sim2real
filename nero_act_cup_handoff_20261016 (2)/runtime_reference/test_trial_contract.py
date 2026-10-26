import copy
import unittest
import numpy as np
from trial_contract import air_trial_allowed
from full_core import FullGate


def record():
    return {'experimental_air_trial_allowed': True, 'execution_scope': 'aligned_noncontact_air_sequence',
        'trial_profile': {'initial_state': [0.]*7+[.06], 'endpoint_state': [.04]*7+[.06], 'arm_lower_rad': [-.01]*7,
            'arm_upper_rad': [.05]*7, 'initial_tolerance_rad': .005,
            'initial_gripper_tolerance_m': .004, 'gripper_lower_m': .052,
            'gripper_upper_m': .078, 'maximum_seconds': 20}}


class ContractTests(unittest.TestCase):
    def test_approval_is_limited_to_explicit_air_task(self):
        value = record(); self.assertTrue(air_trial_allowed(value))
        value['execution_scope'] = 'cup_grasp'
        self.assertFalse(air_trial_allowed(value))

    def test_old_shadow_artifact_is_not_upgraded_by_loading_it(self):
        self.assertFalse(air_trial_allowed({'hardware_deployment_allowed': False}))

    def test_widening_motion_or_pinching_aperture_invalidates_contract(self):
        value = record(); value['trial_profile']['arm_upper_rad'][0] = 2.
        self.assertFalse(air_trial_allowed(value))
        value = record(); value['trial_profile']['gripper_lower_m'] = .001
        self.assertFalse(air_trial_allowed(value))

    def test_growing_initial_tolerance_is_rejected(self):
        value = record(); value['trial_profile']['initial_tolerance_rad'] = .15
        self.assertFalse(air_trial_allowed(value))

    def test_feedback_envelope_applies_at_every_read(self):
        profile = record()['trial_profile']
        summary = {'settings': {'joint_limits_rad': [[-3, 3]]*7}, 'train_qpos_min': [-.1]*7,
                   'train_qpos_max': [.1]*7, 'trial_profile': profile}
        gate = FullGate(summary, True)
        state = np.r_[np.zeros(7), .06]; state[0] = .1
        with self.assertRaisesRegex(ValueError, 'trial envelope'):
            gate.state(state)

    def test_shifted_start_requires_a_new_start_not_a_rebase(self):
        profile = record()['trial_profile']
        summary = {'settings': {'joint_limits_rad': [[-3, 3]]*7}, 'train_qpos_min': [-.1]*7,
                   'train_qpos_max': [.1]*7, 'trial_profile': profile}
        gate = FullGate(summary, True)
        state = np.r_[np.zeros(7), .06]; state[0] = .01
        self.assertTrue(any('trained air-sequence start' in issue for issue in gate.startup_issues(state)))


if __name__ == '__main__':
    unittest.main()
