"""Pause/stop decisions only; all motion targets remain ACT outputs."""
import numpy as np


class PhaseGuard:
    def __init__(self, source_xy, destination_xy,grasp_height=.055):
        self.source, self.destination = np.asarray(source_xy), np.asarray(destination_xy)
        self.grasp_height=float(grasp_height)
        if not np.isfinite(self.grasp_height):raise ValueError('Finite grasp height required')
        self.lowest_before_hold = None
        self.retreat_confirmed = False

    def inspect(self, current, proposed, widths, previous_width, contact):
        current, proposed, widths = np.asarray(current), np.asarray(proposed), np.asarray(widths)
        if (current.shape != (3,) or proposed.ndim != 2 or proposed.shape[1] != 3
                or widths.shape != (len(proposed),) or not len(proposed)
                or not np.isfinite(np.r_[current, proposed.ravel(), widths, previous_width]).all()):
            raise RuntimeError('Invalid measured/predicted contact geometry')
        source_distance = np.linalg.norm(current[:2]-self.source)
        place_distance = np.linalg.norm(current[:2]-self.destination)
        if not contact.close_authorized and current[2] < .045:
            raise RuntimeError('ACT descended below the grasp region without GRASP')
        if contact.state == 'released' and widths.min() < .09:
            raise RuntimeError('ACT requested another closure after the confirmed release')
        first_close_at_depth=(abs(current[2]-self.grasp_height)<=.002
                              and widths.min()<previous_width-1e-6)
        if not contact.close_authorized and (widths.min() < .085 or first_close_at_depth):
            if source_distance > .02 or not .045 <= current[2] <= .085:
                raise RuntimeError('ACT requested closure outside the observed grasp region')
            return 'GRASP'
        if contact.close_authorized and contact.state == 'open':
            self.lowest_before_hold = min(current[2], self.lowest_before_hold or current[2])
            if source_distance > .02:
                raise RuntimeError('ACT left source before stable measured contact')
            if proposed[:, 2].max() > self.lowest_before_hold+.001:
                if contact.first_contact_stamp is not None:
                    return 'WAIT_CONTACT'
                raise RuntimeError('ACT requested lift without measured cup contact')
        if contact.state == 'holding' and not contact.carry_authorized:
            if source_distance > .03:
                raise RuntimeError('Cup transferred before actual-lift confirmation')
            if np.linalg.norm(proposed[:, :2]-self.source, axis=1).max() > .03:
                if current[2] < .20:
                    raise RuntimeError('ACT requested transfer below the accepted high region')
                return 'CARRY'
        if (contact.state == 'holding' and not contact.release_authorized
                and widths.max() > contact.contact_width-.003):
            if place_distance > .02 or current[2] > .085:
                raise RuntimeError('ACT requested opening outside supported-placement region')
            return 'RELEASE'
        if contact.state == 'released' and not self.retreat_confirmed:
            return 'RETREAT'
        return None
