"""Read-only scene and measured-motion guards; these functions cannot authorize motion."""
import math
import numpy as np


def require_current_scene(scene, actual, geometry, *, now):
    expected = np.r_[scene['initial_robot_state']['qpos_rad'],
                     scene['initial_robot_state']['gripper_total_width_m']]
    actual = np.asarray(actual, float)
    age = now-geometry['capture_wall_s']
    if (actual.shape != (8,) or expected.shape != (8,) or not np.isfinite(np.r_[actual, expected]).all()
            or not math.isfinite(age) or not 0 <= age <= 90
            or geometry.get('geometry_fit_passed') is not True
            or scene['table_top_z_m']*1000 != geometry.get('table_z_base_mm')
            or np.max(abs(actual[:7]-expected[:7])) > .005
            or abs(actual[7]-expected[7]) > .004):
        raise ValueError('Current actual start or fresh scene differs from the trained profile')
    for key, observed in [('pick_xy_m', 'source_bottom_mm'), ('place_xy_m', 'destination_mm')]:
        point = np.asarray(geometry['targets'][observed], float)/1000
        if (point.shape != (3,) or not np.isfinite(point).all()
                or np.linalg.norm(point[:2]-np.asarray(scene[key])) > .005
                or abs(point[2]-scene['table_top_z_m']) > .001):
            raise ValueError('Current cup/card geometry differs from the trained scene')


def actual_path_verified(commands, feedback):
    commands, feedback = np.asarray(commands, float), np.asarray(feedback, float)
    if (commands.ndim != 2 or feedback.ndim != 2 or commands.shape[1:] != (8,)
            or feedback.shape[1:] != (8,) or min(len(commands), len(feedback)) < 2
            or not np.isfinite(np.r_[commands.ravel(), feedback.ravel()]).all()):
        return False
    intended = np.ptp(commands[:, :7], axis=0)
    observed = np.ptp(feedback[:, :7], axis=0)
    moving = intended > .03
    return bool(intended.max() >= .1 and np.all(observed[moving] >= .8*intended[moving])
                and np.all(observed[~moving] <= intended[~moving]+.03)
                and np.max(abs(commands[-1, :7]-feedback[-1, :7])) <= .03)
