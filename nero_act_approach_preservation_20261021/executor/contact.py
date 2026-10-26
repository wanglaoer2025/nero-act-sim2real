"""Explicit operator permissions and measured low-force contact health."""
import math


def contact_sync_allowed(*, command, feedback, force, age_s, authorized):
    return bool(authorized is True and all(math.isfinite(x) for x in [command, feedback, force, age_s])
                and 0 <= age_s <= .15 and .06 <= feedback <= .09
                and .0005 <= feedback-command <= .012 and .1 <= abs(force) <= 1.25)


class ContactMonitor:
    def __init__(self):
        self.state = 'open'
        self.close_authorized = self.release_authorized = self.carry_authorized = False
        self.last_stamp = self.first_contact_stamp = self.contact_width = None

    def authorize(self, event, token, observation):
        if event == 'close':
            if self.close_authorized or token != 'GRASP' or observation.get('aligned') is not True:
                raise ValueError('New aligned onsite GRASP confirmation required')
            self.close_authorized = True
        elif event == 'carry':
            if self.state != 'holding' or token != 'NEXT' or observation.get('cup_lifted') is not True:
                raise ValueError('Actual lifted stable cup must be confirmed before transfer')
            self.carry_authorized = True
        elif event == 'release':
            if (self.state != 'holding' or self.release_authorized or token != 'RELEASE'
                    or observation.get('supported') is not True or observation.get('upright') is not True):
                raise ValueError('Onsite supported upright placement must precede RELEASE')
            self.release_authorized = True
        else:
            raise ValueError('Unknown contact permission')

    def update(self, *, width, force, target, stamp):
        if not all(math.isfinite(x) for x in [width, force, target, stamp]) or not 0 <= width <= .1:
            raise RuntimeError('Invalid gripper feedback')
        if abs(force) > 1.25:
            raise RuntimeError('Low-force gripper threshold exceeded')
        if self.last_stamp is not None and stamp < self.last_stamp:
            raise RuntimeError('Gripper feedback timestamp moved backwards')
        if stamp == self.last_stamp:
            return self.state
        if self.last_stamp is not None and stamp-self.last_stamp > .15:
            self.first_contact_stamp = None
            if self.state == 'holding' and not self.release_authorized:
                raise RuntimeError('Continuous contact feedback was interrupted')
        self.last_stamp = stamp
        contact = .06 <= width <= .09 and .003 <= width-target <= .012 and abs(force) >= .1
        if self.state == 'holding' and not self.release_authorized:
            if not contact or width < self.contact_width-.003:
                raise RuntimeError('Cup contact lost or gripper width changed unexpectedly')
        if self.close_authorized and contact and not self.release_authorized:
            if self.first_contact_stamp is None:
                self.first_contact_stamp = stamp
            if stamp-self.first_contact_stamp >= .5 and self.state != 'holding':
                self.state, self.contact_width = 'holding', width
        elif not self.release_authorized:
            self.first_contact_stamp = None
        if self.release_authorized and width > .09 and abs(force) < .1:
            self.state = 'released'
        return self.state
