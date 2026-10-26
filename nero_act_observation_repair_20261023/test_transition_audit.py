import importlib.util
from pathlib import Path
import unittest

import numpy as np


class TransitionAuditTests(unittest.TestCase):
    def test_near_identical_inputs_with_different_first_action_are_reported(self):
        path=Path(__file__).resolve().parent/'transition_audit.py'
        self.assertTrue(path.is_file(),'Missing observable state/action ambiguity audit')
        spec=importlib.util.spec_from_file_location('_transition_audit',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        first=np.zeros(15);second=first.copy();second[0]=1e-6
        action=np.zeros(8);moving=action.copy();moving[1]=.0015
        rgb=np.zeros((2,2,3),np.uint8)
        self.assertTrue(m.ambiguous_transition(first,second,action,moving,rgb,rgb))
        self.assertFalse(m.ambiguous_transition(first,second,action,action,rgb,rgb))
        second[8]=.0002
        self.assertFalse(m.ambiguous_transition(first,second,action,moving,rgb,rgb))


if __name__=='__main__':unittest.main()
