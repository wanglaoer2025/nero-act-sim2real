import importlib.util
import unittest


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('simulation.outcome'),
                             'full-task physical acceptance monitor is missing')
        from simulation.outcome import OutcomeMonitor
        self.monitor = OutcomeMonitor(pick_xy=[-.5, 0], place_xy=[-.5, -.16], table_z=0)

    def sample(self, **changes):
        measurement = dict(cup_xyz=[-.5, -.16, .0475], cup_upright=1., cup_speed=0.,
                           bilateral=False, supported=True, unexpected_contact=False,
                           gripper_width=.095, tcp_xyz=[-.5, -.16, .255], contract_valid=True)
        measurement.update(changes)
        return measurement

    def complete_transfer(self):
        for _ in range(20):
            self.monitor.update(self.sample(cup_xyz=[-.5, 0, .2475], bilateral=True,
                                           supported=False, gripper_width=.075))
        for _ in range(20):
            self.monitor.update(self.sample(cup_xyz=[-.5, -.16, .2475], bilateral=True,
                                           supported=False, gripper_width=.075))
        self.monitor.update(self.sample(bilateral=True, gripper_width=.075))
        for _ in range(100):
            self.monitor.update(self.sample())

    def test_cup_already_on_card_without_lift_and_transfer_is_not_success(self):
        for _ in range(150):
            self.monitor.update(self.sample())
        self.assertFalse(self.monitor.result()['success'])

    def test_transport_drop_then_stable_landing_is_not_supported_release(self):
        for _ in range(20):
            self.monitor.update(self.sample(cup_xyz=[-.5, 0, .2475], bilateral=True,
                                           supported=False, gripper_width=.075))
        for _ in range(20):
            self.monitor.update(self.sample(cup_xyz=[-.5, -.16, .2475], bilateral=True,
                                           supported=False, gripper_width=.075))
        for _ in range(100):
            self.monitor.update(self.sample())
        self.assertFalse(self.monitor.result()['success'])

    def test_actual_lift_transfer_release_retreat_and_five_second_stability(self):
        self.complete_transfer()
        self.assertTrue(self.monitor.result()['success'])

    def test_task_outcome_does_not_override_contract_failure(self):
        self.complete_transfer()
        self.monitor.update(self.sample(contract_valid=False))
        self.assertFalse(self.monitor.result()['success'])

    def test_slip_after_placement_invalidates_the_hold_window(self):
        self.complete_transfer()
        self.monitor.update(self.sample(cup_xyz=[-.5, -.20, .0475]))
        self.assertFalse(self.monitor.result()['success'])

    def test_contact_and_incomplete_observation_window_do_not_pass(self):
        self.complete_transfer()
        self.monitor.update(self.sample(bilateral=True, gripper_width=.075))
        self.assertFalse(self.monitor.result()['success'])


if __name__ == '__main__':
    unittest.main()
