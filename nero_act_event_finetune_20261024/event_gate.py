"""Stable physical events for OFFLINE experts only; no policy inputs or hardware."""
import math


class StableEvent:
    def __init__(self,stage,control_hz):
        if stage not in ('close','release') or control_hz!=20:
            raise ValueError('Only declared20Hz offline contact events supported')
        self.stage=stage
        self.samples=self.stable=0

    def update(self,measurement,previous_gripper_target_m):
        required=('bilateral','supported','cup_upright','cup_speed',
                  'unexpected_contact','contract_valid','gripper_width')
        if any(k not in measurement for k in required):
            raise ValueError('Complete physical contact measurement required')
        upright,speed,width,target=(float(x) for x in (
            measurement['cup_upright'],measurement['cup_speed'],
            measurement['gripper_width'],previous_gripper_target_m))
        if (not all(math.isfinite(x) for x in [upright,speed,width,target])
                or not 0<=width<=.096 or not 0<=target<=.095001 or speed<0
                or measurement['unexpected_contact'] or not measurement['contract_valid']):
            raise ValueError('Invalid or unsafe offline contact state')
        stable=upright>.95 and speed<.01
        if self.stage=='close':
            stable &= bool(measurement['bilateral'] and .05<width<.085 and target<=width-.006)
        else:
            stable &= bool(measurement['supported'] and not measurement['bilateral'] and width>.09)
        self.samples+=1
        self.stable=self.stable+1 if stable else 0
        if self.stable>=10:return True
        if self.samples>=160:raise TimeoutError('Offline contact event did not stabilize within8seconds')
        return False
