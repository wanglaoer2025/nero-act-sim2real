import unittest,numpy as np
class ReplayTraceTests(unittest.TestCase):
    def test_phase_reference_uses_elapsed_time_and_keeps_last_point(self):
        from executor.replay_trace import reference_at
        start=np.zeros(8);rows=np.ones((2,8))*.002
        self.assertTrue(np.array_equal(reference_at(start,rows,-1),start))
        self.assertTrue(np.allclose(reference_at(start,rows,.025),.001))
        self.assertTrue(np.array_equal(reference_at(start,rows,10),rows[-1]))
    def test_replay_rejects_jump_nonfinite_or_outside_original_limits(self):
        from executor.replay_trace import validate_reference
        rows=np.tile([0,0,0,1.5,0,0,0,.095],(20,1));validate_reference(rows)
        for change in ['jump','nan','limit','gripper']:
            bad=rows.copy()
            if change=='jump':bad[8,0]=.1
            elif change=='nan':bad[8,0]=np.nan
            elif change=='limit':bad[:,0]=3
            else:bad[8,7]=.08
            with self.assertRaises(ValueError):validate_reference(bad)
    def test_unknown_trajectory_cannot_be_authorized_by_replay_token(self):
        from executor.replay_trace import verify_replay
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((FileNotFoundError,ValueError)):
                verify_replay(Path(d),{},dict(confirmation='ACT_REPLAY'))
