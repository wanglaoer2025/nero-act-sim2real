"""Offline-only rejection of attached-cup descent, including supported escape sweeps.

The100um bound is a nominal MuJoCo micro-contact limit, never a real error bound.
No action is generated or modified here; callers execute only accepted ACT rows.
"""
import math
import numpy as np

FLOOR_M=-1e-6
MAX_EXISTING_CONTACT_M=1e-4


def unsafe_attached_prefix(current_bottom,predicted_bottom,current_z,predicted_z,*,
                           measurement,airborne,simulation_only):
    b,z=np.asarray(predicted_bottom,float),np.asarray(predicted_z,float)
    if (simulation_only is not True or type(airborne) is not bool
            or b.ndim!=1 or not len(b) or z.shape!=b.shape
            or not np.isfinite(np.r_[current_bottom,current_z,b,z]).all()):
        raise ValueError('Explicit offline scope and finite paired swept attached geometry required')
    escape=bool(-MAX_EXISTING_CONTACT_M<=current_bottom<FLOOR_M and not airborne
        and measurement.get('supported') is True and measurement.get('bilateral') is True
        and measurement.get('unexpected_contact') is False
        and measurement.get('unexpected_robot_contact') is False)
    previous_b,previous_z=float(current_bottom),float(current_z)
    unsafe=[]
    for i,(bottom,height) in enumerate(zip(b,z)):
        if bottom<FLOOR_M:
            if not (escape and bottom>previous_b+1e-10 and height>previous_z+1e-10):
                unsafe.append(i)
                escape=False
        else:
            escape=False
        previous_b,previous_z=float(bottom),float(height)
    return np.asarray(unsafe,dtype=np.int64)


def unsafe_attached_rows(state,rows,bottom,geometry,*,measurement,airborne):
    state,rows=np.asarray(state,float),np.asarray(rows,float)
    if (state.shape!=(8,) or rows.ndim!=2 or rows.shape[1]!=8 or not len(rows)
            or not np.isfinite(np.r_[state,rows.ravel()]).all()):
        raise ValueError('Finite paired ACT rows and current state required')
    previous=state;bottoms=[];heights=[];owners=[]
    for i,target in enumerate(rows):
        count=max(4,math.ceil(float(np.max(abs(target-previous)))/.00025))
        for alpha in np.linspace(0.,1.,count+1)[1:]:
            sampled=previous+alpha*(target-previous)
            bottoms.append(bottom(sampled));heights.append(geometry.pose(sampled)['tcp'][2]);owners.append(i)
        previous=target
    bad=unsafe_attached_prefix(bottom(state),bottoms,geometry.pose(state)['tcp'][2],heights,
        measurement=measurement,airborne=airborne,simulation_only=True)
    return np.unique(np.asarray(owners,dtype=np.int64)[bad])
