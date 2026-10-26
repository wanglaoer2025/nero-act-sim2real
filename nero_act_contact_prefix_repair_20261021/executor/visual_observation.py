"""Shared metric visual features and fail-closed observation checks."""
import numpy as np

from calibration.bundle import SCHEMA, calibration_hash, camera_metadata_hash

VISUAL_PROFILE = 'visual_relative_base_v1'


def relative_positions(state, geometry, robot_geometry):
    state = np.asarray(state,float)
    if not geometry.valid or state.shape != (8,) or not np.isfinite(state).all() or not 0 <= state[7] <= .1:
        raise ValueError('Valid visual observation and physical8D state required')
    cup,target,tcp = [np.asarray(x,float) for x in [geometry.cup_bottom_base_m,geometry.target_base_m,robot_geometry.pose(state)['tcp']]]
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in [cup,target,tcp]):
        raise ValueError('Finite base-frame cup/target/TCP positions required')
    return np.r_[cup-tcp,target-tcp].astype(np.float32)


def geometry_statistics(features, groups, training_groups):
    values = np.asarray(features,float)
    if values.shape != (len(groups),6) or not np.isfinite(values).all() or not training_groups:
        raise ValueError('Finite paired6D features and nonempty training groups required')
    mask = np.array([g in training_groups for g in groups])
    if not np.any(mask) or set(training_groups)-set(groups):
        raise ValueError('Training groups missing from visual observations')
    selected = values[mask]
    return dict(geometry_mean=selected.mean(0).astype(np.float32),
                geometry_std=np.maximum(selected.std(0),1e-6).astype(np.float32))


def require_visual_observation(frame, geometry, paired_feedback, *, now_s, calibration):
    if (calibration.get('schema') != SCHEMA or calibration.get('verified') is not True
            or calibration.get('synthetic') is not False
            or calibration.get('calibration_sha256') != calibration_hash(calibration)):
        raise ValueError('Verified real calibration and unchanged sample-bound artifact required')
    if (not geometry.valid or geometry.calibration_sha256 != calibration['calibration_sha256']
            or geometry.frame_id != frame.frame_id or paired_feedback['frame_id'] != frame.frame_id
            or not frame.clock_verified
            or frame.camera_metadata_hash != camera_metadata_hash(calibration['camera_metadata'])):
        raise ValueError('Visual geometry, frame, clock and current calibration do not agree')
    values = np.r_[geometry.cup_bottom_base_m,geometry.target_base_m,
                   geometry.error_bound_m,geometry.quality,geometry.cup_handle_yaw_rad]
    state = np.asarray(paired_feedback['state'],float)
    if (values.shape != (9,) or not np.isfinite(values).all() or geometry.quality < .9
            or geometry.error_bound_m < calibration['validation']['bounds_m']['vision_m']
            or state.shape != (8,) or not np.isfinite(state).all() or not 0 <= state[7] <= .1):
        raise ValueError('Invalid or under-reported visual uncertainty/feedback')
    low,high = calibration['geometry']['cup']['handle_yaw_range_rad']
    if not low <= geometry.cup_handle_yaw_rad <= high:
        raise ValueError('Cup handle outside measured orientation range')
    exposure,receipt,feedback,skew = (frame.exposure_monotonic_s,frame.received_monotonic_s,
                                   paired_feedback['monotonic_s'],paired_feedback['skew_s'])
    if (not np.isfinite([exposure,receipt,feedback,skew,now_s]).all()
            or not 0 <= now_s-exposure <= .25 or not exposure <= receipt <= now_s
            or not 0 <= now_s-feedback <= .15 or not 0 <= skew <= .05
            or not np.isclose(skew,abs(feedback-exposure),atol=1e-8,rtol=0)):
        raise ValueError('Stale, future-dated or mispaired visual observation/feedback')


def required_clearance_m(bounds, *, speed_m_s, detection_s, stopping_s):
    try:
        errors = [float(bounds[k]) for k in ['tcp_m','vision_m','shape_m','tracking_m']]
        speed,detection,stopping = map(float,[speed_m_s,detection_s,stopping_s])
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('Measured error and detection/stop bounds required') from exc
    if not np.isfinite([*errors,speed,detection,stopping]).all() or min(*errors,speed,detection,stopping) < 0:
        raise ValueError('Finite nonnegative measured error and stop bounds required')
    return max(.005,sum(errors)+speed*(detection+stopping)+.002)
