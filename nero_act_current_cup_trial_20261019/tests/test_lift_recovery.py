import copy,json,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class LiftRecoveryTests(unittest.TestCase):
    def test_already_held_snapshot_recovery_starts_at_lift_without_unobservable_close_wait(self):
        from simulation.teacher import task_waypoints,run_teacher
        config=json.loads((ROOT/'scene/center.json').read_text())
        config['continuation']='lift_to_retreat'
        self.assertEqual([x[0] for x in task_waypoints(config)],['lift','carry','place','release','retreat'])
        with self.assertRaises(ValueError):run_teacher(config)
