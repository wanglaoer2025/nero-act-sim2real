"""Pure full-task action checks. This module cannot authorize or send hardware commands."""
import numpy as np

GRIPPER_CONTEXT='previous_gripper_target_delta_v1'
GRIPPER_CONTEXT_SCALE_M=.01
VISUAL_PROFILE='visual_relative_base_v1'


class FullTaskContract:
    def __init__(self, config):
        context=config.get('gripper_command_context')
        visual=config.get('visual_observation_profile')
        if visual not in (None,VISUAL_PROFILE):
            raise ValueError('Unknown visual observation profile')
        matched_state=config.get('state_dim')==8 and (
            (not context and not visual and config.get('policy_input_dim',8)==8) or
            (context==GRIPPER_CONTEXT and config.get('policy_input_dim')==(15 if visual else 9)))
        if (config.get('chunk_size') != 20 or not matched_state
                or config.get('physical_state_dim',8)!=8
                or config.get('action_dim') != 8
                or config.get('task_scope') != 'complete_pick_carry_place_retreat'):
            raise ValueError('Only the explicitly matched20x8 complete-task model is supported')
        self.lower = np.array([-2.70526, -1.74, -2.75, -1.01, -2.75, -.73, -1.5707963])
        self.upper = np.array([2.70526, 1.74, 2.75, 2.14, 2.75, .95, 1.5707963])
        self.last_raw = self.last_delta = None

    def validate_prefix(self, raw, observation, history):
        raw = np.asarray(raw, float)
        state, previous = np.asarray(observation['state'], float), np.asarray(history, float)
        age = float(observation['age_s'])
        if raw.shape != (20, 8) or state.shape != (8,) or previous.shape != (8,):
            raise ValueError('Expected prediction[20,8], feedback[8] and previous target[8]')
        if not np.isfinite(np.r_[raw.ravel(), state, previous, age]).all():
            raise ValueError('Nonfinite observation/action')
        if not 0 <= age <= .25 or age+.5 > .75:
            raise ValueError('Prediction observation is stale or future-dated')
        if (np.any(raw[:, :7] < self.lower) or np.any(raw[:, :7] > self.upper)
                or np.any(raw[:, 7] < 0) or np.any(raw[:, 7] > .095)):
            raise ValueError('Raw prediction exceeds physical arm/gripper limits')
        if np.max(abs(raw[:10, :7]-state[:7])) > .03:
            raise ValueError('Discontinuous raw arm prefix; no rebasing or large-jump clipping')
        if np.max(abs(raw[:10, 7]-state[7])) > .01:
            raise ValueError('Discontinuous raw gripper prefix')
        sent = []
        for target in raw[:10]:
            low = np.maximum.reduce([self.lower, previous[:7]-.002, state[:7]-.03])
            high = np.minimum.reduce([self.upper, previous[:7]+.002, state[:7]+.03])
            if np.any(low > high):
                raise ValueError('Incompatible measured feedback and target history')
            arm = np.clip(target[:7], low, high)
            grip_low, grip_high = max(0., previous[7]-.0005), min(.095, previous[7]+.0005)
            if grip_low > grip_high:
                raise ValueError('Incompatible gripper target history')
            previous = np.r_[arm, np.clip(target[7], grip_low, grip_high)]
            sent.append(previous)
        sent = np.asarray(sent)
        self.last_raw, self.last_delta = raw[:10].copy(), sent-raw[:10]
        return sent
