"""Simulation safety geometry only; never generates policy features or targets."""
import numpy as np


class SupportedAttachment:
    def __init__(self,*,radius,half_height):
        if not np.isfinite([radius,half_height]).all() or min(radius,half_height)<=0:
            raise ValueError('Positive finite measured cup dimensions required')
        self.radius,self.half_height=radius,half_height
        self.center=self.rotation=None;self.airborne=False

    def observe(self,flange,cup_center,cup_rotation,*,supported):
        # Before lift-off, table support allows grip compliance. Once airborne,
        # freeze the measured transform permanently; never hide subsequent slip.
        if self.airborne:return
        p=np.asarray(flange['position']);r=np.asarray(flange['rotation'])
        c=np.asarray(cup_center);q=np.asarray(cup_rotation)
        if (p.shape!=(3,) or c.shape!=(3,) or r.shape!=(3,3) or q.shape!=(3,3)
                or not np.isfinite(np.r_[p,c,r.ravel(),q.ravel()]).all()
                or any(not np.allclose(x.T@x,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(x),1.) for x in [r,q])
                or type(supported) is not bool):
            raise ValueError('Current finite paired rigid poses and support status required')
        self.center=r.T@(c-p);self.rotation=r.T@q
        self.airborne=not supported

    def bottom(self,flange,table_z):
        if self.center is None:raise ValueError('Attachment requires a current supported/airborne observation')
        center=flange['position']+flange['rotation']@self.center
        axis=flange['rotation']@self.rotation[:,2]
        return float(center[2]-self.half_height*abs(axis[2])
            -self.radius*np.sqrt(max(0.,1-axis[2]**2))-table_z)
