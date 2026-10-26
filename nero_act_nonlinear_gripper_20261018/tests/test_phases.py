import importlib.util
import unittest
import numpy as np
from executor.contact import ContactMonitor


class PhaseTests(unittest.TestCase):
    def test_grasp_approach_queries_each_point_before_crossing_confirmation_depth(self):
        from executor.phases import PhaseGuard
        guard=PhaseGuard([-.52,.01],[-.52,-.16]);contact=ContactMonitor()
        self.assertTrue(callable(getattr(guard,'single_point_grasp_prefix',None)))
        self.assertTrue(guard.single_point_grasp_prefix([-.52,.01,.09],[[-.52,.01,.084]],contact))
        self.assertFalse(guard.single_point_grasp_prefix([-.52,.01,.20],[[-.52,.01,.19]],contact))
        contact.close_authorized=True
        self.assertFalse(guard.single_point_grasp_prefix([-.52,.01,.055],[[-.52,.01,.054]],contact))

    def test_open_gripper_cannot_descend_below_original_grasp_region_without_confirmation(self):
        from executor.phases import PhaseGuard
        guard=PhaseGuard([-.52,.01],[-.52,-.16]);contact=ContactMonitor()
        with self.assertRaises(RuntimeError):
            guard.inspect([-.52,.01,.044],np.tile([-.52,.01,.043],(10,1)),np.full(10,.095),.095,contact)

    def test_first_small_close_at_grasp_depth_requires_grasp_before_any_motion(self):
        from executor.phases import PhaseGuard
        guard=PhaseGuard([-.52,.01],[-.52,-.16]);contact=ContactMonitor()
        points=np.tile([-.52,.01,.055],(10,1))
        widths=np.linspace(.0948,.0925,10)
        self.assertEqual(guard.inspect(points[0],points,widths,.095,contact),'GRASP')
        self.assertIsNone(guard.inspect([-.52,.01,.20],points,widths,.095,contact))
        self.assertIsNone(guard.inspect(points[0],points,np.full(10,.095),.095,contact))
        with self.assertRaises(RuntimeError):guard.inspect([-.52,.05,.055],points,widths,.095,contact)

    def test_released_cup_does_not_authorize_another_closure(self):
        from executor.phases import PhaseGuard
        guard = PhaseGuard([-.52, .01], [-.52, -.16]);guard.retreat_confirmed = True
        contact = ContactMonitor();contact.state = 'released';contact.close_authorized = True
        contact.release_authorized = contact.carry_authorized = True
        points = np.tile([-.52, -.16, .245], (10, 1))
        with self.assertRaises(RuntimeError):
            guard.inspect(points[0], points, np.full(10, .08), .095, contact)

    def test_grasp_and_release_permissions_follow_actual_geometry(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.phases'))
        from executor.phases import PhaseGuard
        guard = PhaseGuard([-.52, .01], [-.52, -.16])
        contact = ContactMonitor()
        near = np.tile([-.52, .01, .055], (10, 1))
        with self.assertRaises(RuntimeError):
            guard.inspect([-.52, .01, .3], near, np.full(10, .08), .095, contact)
        self.assertEqual(guard.inspect(near[0], near, np.full(10, .08), .095, contact), 'GRASP')
        contact.authorize('close', 'GRASP', {'aligned': True})
        contact.state = 'holding'
        contact.contact_width = .075
        high = np.tile([-.52, -.16, .255], (10, 1))
        with self.assertRaises(RuntimeError):
            guard.inspect(high[0], high, np.full(10, .075), .065, contact)
        low = high.copy();low[:, 2] = .055
        contact.carry_authorized = True
        self.assertIsNone(guard.inspect(high[0], high, np.full(10, .068), .065, contact))
        self.assertEqual(guard.inspect(low[0], low, np.full(10, .075), .065, contact), 'RELEASE')


if __name__ == '__main__':unittest.main()
