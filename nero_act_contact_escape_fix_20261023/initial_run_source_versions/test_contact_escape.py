import json
from pathlib import Path
import unittest

import numpy as np

from contact_escape import unsafe_attached_prefix,unsafe_attached_rows


class ContactEscapeTests(unittest.TestCase):
    def known_failure(self):
        p=Path(__file__).resolve().parent.parent/'nero_act_new_handeye_policy_repair_20261023/results/9000/result.json'
        return json.loads(p.read_text())['guard_diagnostic']

    def call(self,current=-.000037739,predicted=(-.000027071,.000002165),z=(.055,.05502),**context):
        m=dict(supported=True,bilateral=True,unexpected_contact=False,unexpected_robot_contact=False)
        m.update(context)
        return unsafe_attached_prefix(current,np.array(predicted),.05499,np.array(z),
            measurement=m,airborne=False,simulation_only=True)

    def test_original_upward_prefix_can_leave_existing_supported_micro_contact(self):
        d=self.known_failure()
        indices=unsafe_attached_prefix(d['actual_attached_model_bottom_m'],
            np.array(d['predicted_cup_bottom_m']),d['current_tcp'][2],
            np.array(d['proposed_tcp'])[:,2],measurement=d['measured_physics'],
            airborne=False,simulation_only=True)
        self.assertEqual(indices.tolist(),[])

    def test_downward_or_worsening_rotation_or_stationary_prefix_still_rejected(self):
        self.assertEqual(self.call(predicted=(-.00004,.00001)).tolist(),[0])
        self.assertEqual(self.call(z=(.05498,.05502)).tolist(),[0])
        self.assertEqual(self.call(predicted=(-.000037739,-.000037739),z=(.05499,.05499)).tolist(),[0,1])

    def test_missing_physical_support_or_excess_penetration_cannot_get_exception(self):
        self.assertEqual(self.call(supported=False).tolist(),[0])
        self.assertEqual(self.call(bilateral=False).tolist(),[0])
        self.assertEqual(self.call(unexpected_contact=True).tolist(),[0])
        self.assertEqual(self.call(current=-.000101).tolist(),[0])

    def test_original_free_space_floor_and_hardware_rejection_unchanged(self):
        self.assertEqual(self.call(current=.001,predicted=(-.00002,.00001)).tolist(),[0])
        self.assertEqual(self.call(current=.001,predicted=(.00002,.00001)).tolist(),[])
        with self.assertRaises(ValueError):
            unsafe_attached_prefix(-.00003,np.array([-.00002]),.055,np.array([.0551]),
                measurement={},airborne=False,simulation_only=False)
        with self.assertRaises(ValueError):self.call(current=float('nan'))

    def test_once_clear_or_worsened_cannot_dip_back_under_floor(self):
        self.assertEqual(self.call(predicted=(.00001,-.00002)).tolist(),[1])

    def test_sweep_rejects_an_interior_dip_with_clear_endpoints(self):
        class Geometry:
            def pose(self,state):return {'tcp':np.array([0.,0.,.055+state[0]*.001])}
        state=np.zeros(8);row=state.copy();row[0]=.002
        def bottom(sample):return .0001-.001*np.sin(np.pi*sample[0]/.002)
        bad=unsafe_attached_rows(state,row[None],bottom,Geometry(),measurement={},airborne=True)
        self.assertEqual(bad.tolist(),[0])


if __name__=='__main__':unittest.main()
