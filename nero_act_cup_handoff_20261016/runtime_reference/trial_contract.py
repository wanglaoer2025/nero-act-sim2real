"""Permission for this newly trained, narrowly bounded air trial only."""
import numpy as np


def air_trial_allowed(record):
    if record.get('experimental_air_trial_allowed') is not True or record.get('execution_scope') != 'aligned_noncontact_air_sequence':
        return False
    profile = record.get('trial_profile', {})
    try:
        initial = np.asarray(profile['initial_state'], dtype=float)
        endpoint = np.asarray(profile['endpoint_state'], dtype=float)
        lower = np.asarray(profile['arm_lower_rad'], dtype=float)
        upper = np.asarray(profile['arm_upper_rad'], dtype=float)
        return bool(initial.shape == endpoint.shape == (8,) and lower.shape == upper.shape == (7,)
            and np.isfinite(np.r_[initial, endpoint, lower, upper]).all()
            and np.all(endpoint[:7] >= lower) and np.all(endpoint[:7] <= upper)
            and np.max(abs(endpoint[:7]-initial[:7])) <= .05
            and profile['gripper_lower_m'] <= endpoint[7] <= profile['gripper_upper_m']
            and np.all(lower < upper) and np.all(lower <= initial[:7]) and np.all(upper >= initial[:7])
            and np.max(upper-lower) <= .08 and np.max(abs(lower-initial[:7])) <= .065
            and np.max(abs(upper-initial[:7])) <= .065
            and 0 < profile['initial_tolerance_rad'] <= .005
            and 0 < profile['initial_gripper_tolerance_m'] <= .004
            and .05 <= profile['gripper_lower_m'] < profile['gripper_upper_m'] <= .08
            and profile['gripper_lower_m'] <= initial[7] <= profile['gripper_upper_m']
            and 0 < profile['maximum_seconds'] <= 20)
    except (KeyError, TypeError, ValueError):
        return False
