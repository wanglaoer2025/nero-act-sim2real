"""Zero-motion posture hold during RELEASE; task actions and gripper remain ACT."""
import numpy as np
from executor.contract import FullTaskContract


def require_support_pose(position, reference):
    position, reference=np.asarray(position,float),np.asarray(reference,float)
    if (position.shape!=(3,) or reference.shape!=(3,)
            or not np.isfinite(np.r_[position,reference]).all()
            or np.linalg.norm(position-reference)>.003):
        raise RuntimeError('Arm left the3mm supported-release envelope')


def hold_act_arm_during_release(rows, previous_act_target):
    rows, previous=np.asarray(rows,float),np.asarray(previous_act_target,float)
    if (rows.ndim!=2 or rows.shape[1]!=8 or previous.shape!=(8,)
            or not np.isfinite(np.r_[rows.ravel(),previous]).all()):
        raise ValueError('Finite ACT prefix and its previous validated target required')
    # An unchanged goal is safe to retain even with a constant servo tracking offset.
    # The separate measured-pose guard bounds actual movement, not commanded FK error.
    held=rows.copy();held[:,:7]=previous[:7]
    return held


def release_reference_transfer(previous,measured):
    previous,measured=np.asarray(previous,float),np.asarray(measured,float)
    if (previous.shape!=(8,) or measured.shape!=(8,)
            or not np.isfinite(np.r_[previous,measured]).all()):
        raise ValueError('Finite reference and measured posture required')
    guard=FullTaskContract(dict(task_scope='complete_pick_carry_place_retreat',chunk_size=20,state_dim=8,action_dim=8))
    if (max(abs(previous[:7]-measured[:7]))>.03 or not 0<=previous[7]<=.095
            or np.any(measured[:7]<guard.lower) or np.any(measured[:7]>guard.upper)):
        raise RuntimeError('Reference transfer exceeds original tracking/physical limits')
    goal=measured.copy();goal[7]=previous[7]
    count=max(1,int(np.ceil(max(abs(goal[:7]-previous[:7]))/.002)))
    return np.array([previous+(goal-previous)*index/count for index in range(1,count+1)])
