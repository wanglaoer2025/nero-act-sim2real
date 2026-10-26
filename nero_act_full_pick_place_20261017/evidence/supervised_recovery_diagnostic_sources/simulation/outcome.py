"""Physical task evidence only; this monitor never generates control actions."""
import numpy as np


class OutcomeMonitor:
    def __init__(self, pick_xy, place_xy, table_z, *, cup_height=.095):
        self.pick, self.place = np.asarray(pick_xy, float), np.asarray(place_xy, float)
        self.table_z, self.height = float(table_z), float(cup_height)
        self.max_lift = self.max_transfer = 0.
        self.lift_contact_frames = self.stable_released = self.unexpected = self.contract_failures = 0
        self.carried = False
        self.had_lifted = self.supported_before_release = self.premature_release = False
        self.last = None

    def update(self, measurement):
        cup = np.asarray(measurement['cup_xyz'], float)
        tcp = np.asarray(measurement['tcp_xyz'], float)
        if cup.shape != (3,) or tcp.shape != (3,) or not np.isfinite(np.r_[cup, tcp]).all():
            raise ValueError('Invalid finite cup/TCP observation')
        bottom = cup[2]-self.height/2-self.table_z
        transfer = float(np.linalg.norm(cup[:2]-self.pick))
        self.max_lift = max(self.max_lift, float(bottom))
        self.max_transfer = max(self.max_transfer, transfer)
        self.unexpected += int(measurement['unexpected_contact'])
        self.contract_failures += int(not measurement['contract_valid'])
        lifted = bottom > .06 and measurement['bilateral'] and measurement['cup_upright'] > .95
        if lifted:
            self.had_lifted = True
            self.lift_contact_frames += 1
            if transfer > .1 and self.lift_contact_frames >= 20:
                self.carried = True
        if self.had_lifted and measurement['bilateral']:
            self.supported_before_release = bool(measurement['supported'] and abs(bottom) < .002
                                                and np.linalg.norm(cup[:2]-self.place) < .02)
        if self.had_lifted and not measurement['bilateral'] and not self.supported_before_release:
            self.premature_release = True
        stable = (np.linalg.norm(cup[:2]-self.place) < .02
                  and abs(bottom) < .002 and measurement['supported']
                  and measurement['cup_upright'] > .95 and measurement['cup_speed'] < .05
                  and not measurement['bilateral'] and measurement['gripper_width'] > .085
                  and tcp[2] >= self.table_z+.245)
        self.stable_released = self.stable_released+1 if stable else 0
        self.last = dict(measurement)

    def result(self):
        physical = bool(self.carried and self.stable_released >= 100 and self.unexpected == 0
                        and self.supported_before_release and not self.premature_release)
        return dict(success=physical and self.contract_failures == 0,
                    physical_success=physical, max_lift_m=self.max_lift,
                    maximum_horizontal_transfer_m=self.max_transfer,
                    stable_released_frames=self.stable_released,
                    lift_contact_frames=self.lift_contact_frames,
                    unexpected_arm_cup_contact_frames=self.unexpected,
                    execution_contract_failures=self.contract_failures,
                    premature_release_before_support=self.premature_release,
                    constraint_attachment_used=False, last_observation=self.last)
