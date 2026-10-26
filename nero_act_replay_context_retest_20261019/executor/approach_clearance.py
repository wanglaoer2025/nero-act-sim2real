"""Offline finger/cup clearance for the complete frozen pre-close reference."""
import numpy as np

from executor.replay_trace import validate_reference


def approach_clearance(scene, manifest, rows):
    import mujoco
    from executor.ros_execution import checked_interpolation
    from simulation.scene import build_scene

    rows = validate_reference(rows)
    phase = manifest['phases'][0]
    if phase['name'] != 'approach' or phase['start'] != 0 or not 2 <= phase['end'] <= len(rows):
        raise ValueError('Complete pre-close approach required for clearance check')
    model = build_scene(scene)
    data = mujoco.MjData(model)
    body = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    fingers, cup = {body('gripper_link1'), body('gripper_link2')}, body('cup')
    if min(*fingers, cup) < 0:
        raise ValueError('Missing physical finger/cup bodies for clearance check')
    active = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    finger_geoms = [g for g in active if model.geom_bodyid[g] in fingers]
    cup_geoms = [g for g in active if model.geom_bodyid[g] == cup]
    if not finger_geoms or not cup_geoms:
        raise ValueError('Missing contact geometry for approach clearance check')
    # Use the supported cup, rather than the scene's 1mm spawn clearance.
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cup_free')
    data.qpos[model.jnt_qposadr[free]+2] = scene['table_top_z_m']+scene['cup']['height_m']/2
    initial = np.r_[scene['initial_robot_state']['qpos_rad'],
                    scene['initial_robot_state']['gripper_total_width_m']]
    checks = checked_interpolation(initial, rows[:phase['end']])
    minimum, closest = .05, None
    for index, row in enumerate(checks):
        data.qpos[:7] = row[:7]
        data.qpos[7:9] = [row[7]/2, -row[7]/2]
        mujoco.mj_forward(model, data)
        gap = min(mujoco.mj_geomDistance(model, data, a, b, .05, None)
                  for a in finger_geoms for b in cup_geoms)
        if gap < minimum:
            minimum, closest = float(gap), index
    return dict(minimum_gap_m=minimum, required_gap_m=.005,
                passed=minimum >= .005, interpolated_states=len(checks),
                closest_interpolated_state=closest,
                scope='nominal cylindrical cup and modeled fingers; protrusions and TCP remain unmeasured',
                hardware_connected=False)


def require_approach_clearance(scene, manifest, rows, report):
    result = approach_clearance(scene, manifest, rows)
    report['approach_clearance'] = result
    if not result['passed']:
        raise RuntimeError('Approach clearance %.2fmm is below the existing 5mm cup-position bound'
                           % (result['minimum_gap_m']*1000))
    return result
