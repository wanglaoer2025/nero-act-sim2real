import importlib.util
import unittest


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.contact'), 'contact-aware guards missing')
        from executor.contact import contact_sync_allowed, ContactMonitor
        self.allowed, self.monitor = contact_sync_allowed, ContactMonitor()

    def test_contact_sync_requires_current_authorization_force_and_freshness(self):
        valid = dict(command=.065, feedback=.074, force=.5, age_s=.02, authorized=True)
        self.assertTrue(self.allowed(**valid))
        for field, value in [('authorized', False), ('force', .01), ('force', 2.),
                             ('age_s', .151), ('feedback', .09), ('command', float('nan'))]:
            self.assertFalse(self.allowed(**dict(valid, **{field: value})))

    def test_release_without_actual_support_confirmation_is_rejected(self):
        self.monitor.authorize('close', 'GRASP', dict(aligned=True))
        for i in range(13):
            self.monitor.update(width=.083, force=.5, target=.075, stamp=1.+i*.05)
        self.assertEqual(self.monitor.state, 'holding')
        with self.assertRaises(ValueError):
            self.monitor.authorize('release', 'RELEASE', dict(supported=False, upright=True))
        self.monitor.authorize('release', 'RELEASE', dict(supported=True, upright=True))
        self.assertTrue(self.monitor.release_authorized)

    def test_duplicate_feedback_does_not_establish_contact_and_slip_stops(self):
        self.monitor.authorize('close', 'GRASP', dict(aligned=True))
        for _ in range(30):
            self.monitor.update(width=.083, force=.5, target=.075, stamp=1.)
        self.assertNotEqual(self.monitor.state, 'holding')
        for i in range(1, 13):
            self.monitor.update(width=.083, force=.5, target=.075, stamp=1.+i*.05)
        with self.assertRaises(RuntimeError):
            self.monitor.update(width=.075, force=0., target=.075, stamp=1.7)

    def test_slow_cumulative_width_loss_is_not_rebased_away(self):
        self.monitor.authorize('close', 'GRASP', dict(aligned=True))
        for i in range(13):
            self.monitor.update(width=.083, force=.5, target=.075, stamp=1.+i*.05)
        with self.assertRaises(RuntimeError):
            for i in range(1, 36):
                self.monitor.update(width=.083-i*.0001, force=.5, target=.075, stamp=1.6+i*.05)


if __name__ == '__main__':
    unittest.main()
