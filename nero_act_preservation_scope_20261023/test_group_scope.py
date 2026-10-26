import unittest
import torch

try:
    from group_scope import preservation_scale
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'nero_act_position_response_repair_20261023'))
    from objective import preservation_scale as projection_scale
    def preservation_scale(task, replay, groups, **kwargs):
        return projection_scale(task, replay, groups[0]['params'], **kwargs)


class GroupScopeTests(unittest.TestCase):
    def test_a_quiet_input_projection_cannot_hide_a_dominant_arm_gradient(self):
        projection = torch.tensor(1., requires_grad=True)
        arm = torch.tensor(1., requires_grad=True)
        task, replay = projection+arm, projection+100*arm
        groups = [dict(name='projection', params=[projection]), dict(name='arm', params=[arm])]
        weight, _ = preservation_scale(task, replay, groups)
        self.assertLessEqual(weight*100, .5+1e-12)

    def test_cap_and_absent_preservation_gradient(self):
        arm = torch.tensor(1., requires_grad=True)
        unrelated = torch.tensor(1., requires_grad=True)
        weight, _ = preservation_scale(2*arm, unrelated, [dict(name='arm', params=[arm])])
        self.assertEqual(weight, .1)

    def test_unopposed_preservation_in_a_task_zero_group_is_suppressed(self):
        projection = torch.tensor(1., requires_grad=True)
        arm = torch.tensor(1., requires_grad=True)
        weight, _ = preservation_scale(projection, arm, [
            dict(name='projection', params=[projection]), dict(name='arm', params=[arm])])
        self.assertEqual(weight, 0.)


if __name__ == '__main__':
    unittest.main()
