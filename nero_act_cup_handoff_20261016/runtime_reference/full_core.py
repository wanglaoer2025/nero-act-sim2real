"""Full arm/gripper execution gates. No action rebasing or projection."""
import numpy as np


class FullGate:
    def __init__(self, summary, deployment_allowed):
        self.summary = summary
        self.allowed = deployment_allowed is True
        self.profile = summary.get('trial_profile')
        self.limits = np.asarray(summary['settings']['joint_limits_rad'], dtype=float)
        self.lower = np.asarray(summary['train_qpos_min'], dtype=float)
        self.upper = np.asarray(summary['train_qpos_max'], dtype=float)
        if self.limits.shape != (7, 2) or self.lower.shape != (7,) or self.upper.shape != (7,):
            raise ValueError('invalid gate ranges')
        if not np.isfinite(np.r_[self.limits.ravel(), self.lower, self.upper]).all() or np.any(self.limits[:, 0] >= self.limits[:, 1]) or np.any(self.lower > self.upper):
            raise ValueError('nonfinite or inverted limits')

    def state(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (8,) or not np.isfinite(value).all():
            raise ValueError('expected finite feedback[8]')
        if np.any(value[:7] < self.limits[:, 0]) or np.any(value[:7] > self.limits[:, 1]) or not 0 <= value[7] <= .1:
            raise ValueError('feedback outside physical limits')
        if self.profile:
            if (np.any(value[:7] < np.asarray(self.profile['arm_lower_rad']))
                or np.any(value[:7] > np.asarray(self.profile['arm_upper_rad']))
                or not self.profile['gripper_lower_m'] <= value[7] <= self.profile['gripper_upper_m']):
                raise ValueError('feedback outside bounded noncontact trial envelope')
        return value

    def startup_issues(self, state, chunk=None):
        state = self.state(state)
        issues = []
        if not self.allowed:
            issues.append('checkpoint has no accepted permission for this execution scope')
        if self.profile:
            initial = np.asarray(self.profile['initial_state'])
            if np.max(abs(state[:7]-initial[:7])) > self.profile['initial_tolerance_rad'] or abs(state[7]-initial[7]) > self.profile['initial_gripper_tolerance_m']:
                issues.append('current pose does not match the trained air-sequence start; no automatic repositioning')
        outside = np.flatnonzero((state[:7] < self.lower-.1) | (state[:7] > self.upper+.1))
        if len(outside):
            issues.append('start outside training range: ' + ','.join(f'joint{i+1}' for i in outside))
        if chunk is not None:
            chunk = np.asarray(chunk, dtype=float)
            if chunk.shape != (100, 8) or not np.isfinite(chunk).all():
                issues.append('invalid finite [100,8] prediction')
            else:
                if np.any(chunk[:, :7] < self.limits[:, 0]) or np.any(chunk[:, :7] > self.limits[:, 1]):
                    issues.append('raw100 contains arm targets outside physical limits')
                if np.any(chunk[:, 7] < 0) or np.any(chunk[:, 7] > .095):
                    issues.append('raw100 contains gripper targets outside 0..95mm')
                if np.max(np.abs(chunk[:10, :7]-state[:7])) > .03:
                    issues.append('first action prefix discontinuous from real joints (>0.03rad)')
                if np.max(np.abs(chunk[:10, 7]-state[7])) > .01:
                    issues.append('first gripper prefix discontinuous from real width (>10mm)')
        return issues

    def point(self, raw, feedback, previous):
        raw = np.asarray(raw, dtype=float)
        feedback = self.state(feedback)
        previous = self.state(previous)
        if raw.shape != (8,) or not np.isfinite(raw).all():
            raise ValueError('invalid action')
        if not self.allowed:
            raise ValueError('checkpoint is not enabled for full execution')
        if np.any(raw[:7] < self.limits[:, 0]) or np.any(raw[:7] > self.limits[:, 1]) or not 0 <= raw[7] <= .095:
            raise ValueError('raw action outside physical arm/gripper limits')
        if np.max(np.abs(raw[:7]-feedback[:7])) > .03:
            raise ValueError('discontinuous raw arm target; no rebase or large-jump clipping')
        if abs(raw[7]-feedback[7]) > .01:
            raise ValueError('discontinuous raw gripper target')
        # Rate and feedback-lead constraints are intersected, not sequentially undone.
        lower = np.maximum.reduce([self.limits[:, 0], previous[:7]-.0075, feedback[:7]-.03])
        upper = np.minimum.reduce([self.limits[:, 1], previous[:7]+.0075, feedback[:7]+.03])
        if np.any(lower > upper):
            raise ValueError('incompatible feedback and command history')
        safe = raw.copy()
        safe[:7] = np.clip(raw[:7], lower, upper)
        grip_lower, grip_upper = max(0, previous[7]-.0005), min(.095, previous[7]+.0005)
        if grip_lower > grip_upper:
            raise ValueError('incompatible gripper range and command history')
        safe[7] = np.clip(raw[7], grip_lower, grip_upper)
        self.state(safe)
        return safe

    def feedback_check(self, state, target):
        state = self.state(state)
        target = self.state(target)
        if np.max(abs(state[:7]-target[:7])) > .04 or abs(state[7]-target[7]) > .012:
            raise RuntimeError('real feedback tracking error exceeds limit')
