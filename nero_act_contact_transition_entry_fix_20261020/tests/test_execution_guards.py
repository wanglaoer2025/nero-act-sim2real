import copy
import importlib.util
import unittest
import numpy as np


class ExecutionGuardsTests(unittest.TestCase):
    def test_stale_or_changed_scene_cannot_be_authorized_by_a_token(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.execution_guards'))
        from executor.execution_guards import require_current_scene
        scene = dict(initial_robot_state=dict(qpos_rad=[.1]*7, gripper_total_width_m=.095),
                     pick_xy_m=[-.5, .01], place_xy_m=[-.5, -.15], table_top_z_m=0)
        geometry = dict(capture_wall_s=100, geometry_fit_passed=True, table_z_base_mm=0,
                        targets=dict(source_bottom_mm=[-500, 10, 0], destination_mm=[-500, -150, 0]))
        q = [.1]*7+[.095]
        require_current_scene(scene, q, geometry, now=101)
        for changed in [dict(geometry, capture_wall_s=0), dict(geometry, geometry_fit_passed=False)]:
            with self.assertRaises(ValueError):require_current_scene(scene, q, changed, now=101)
        moved = copy.deepcopy(geometry);moved['targets']['source_bottom_mm'][0] += 10
        with self.assertRaises(ValueError):require_current_scene(scene, q, moved, now=101)
        q[1] += .7
        with self.assertRaises(ValueError):require_current_scene(scene, q, geometry, now=101)

    def test_controller_success_without_actual_motion_is_not_physical_acceptance(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.execution_guards'))
        from executor.execution_guards import actual_path_verified
        command = np.zeros((100, 8));command[:, 0] = np.linspace(0, .4, 100)
        command[:, 7] = .095
        static = np.tile(command[0], (100, 1))
        self.assertFalse(actual_path_verified(command, static))
        self.assertTrue(actual_path_verified(command, command.copy()))
        self.assertFalse(actual_path_verified(command[:2], command[:2]))


if __name__ == '__main__':unittest.main()
