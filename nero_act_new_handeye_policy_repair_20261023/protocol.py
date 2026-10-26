"""Offline training protocol and pure startup checks; no hardware connections."""
import copy
import math

import numpy as np

STAGES = ('approach','descend','close','lift','carry','place','release','retreat')


def plan_scenes(scene, snapshots):
    rows = []
    for split, offsets in [('train',[(0,0),(.015,0),(-.015,0),(0,.015),(0,-.015)]),
                           ('val',[(.01,.01),(-.01,-.01)])]:
        for offset in offsets:
            config = copy.deepcopy(scene)
            config['initial_robot_state']['gripper_total_width_m'] = .095
            config['pick_xy_m'] = (np.asarray(scene['pick_xy_m'])+offset).tolist()
            rows.append(dict(role='center' if offset==(0,0) else 'offset',split=split,
                offset_xy_m=list(offset),scene=config))
    for query in [100,273,450]:
        config = copy.deepcopy(scene)
        saved = copy.deepcopy(snapshots[query])
        config['initial_simulation_snapshot'] = saved
        config['initial_robot_state'] = dict(qpos_rad=saved['qpos'][:7],
            gripper_total_width_m=saved['qpos'][7]-saved['qpos'][8])
        rows.append(dict(role='recovery',split='train',parent_query=query,scene=config))
    for index, row in enumerate(rows,9000):
        row.update(id=index,source_group=f'new_handeye_{row["split"]}_{index}')
        row['scene']['generalization'] = dict(index=index,split=row['split'],
            source_group=row['source_group'],seed=231023,
            label_source='complete_physical_teacher_not_failed_ACT_actions')
        row['scene']['hypothetical_preparation_required'] = True
        row['scene']['hardware_deployment_allowed'] = False
    return rows


def balanced_batches(pools, *, steps=1000):
    keys = [f'{kind}:{stage}' for kind in ['center','other'] for stage in STAGES]
    if (type(steps) is not int or not 0<steps<=1000 or set(pools)!=set(keys)
            or any(not len(pools[key]) for key in keys)):
        raise ValueError('All eight center and other training-stage pools required')
    flat = [int(x) for key in keys for x in pools[key]]
    if len(set(flat))!=len(flat) or min(flat)<0:
        raise ValueError('Each training sample belongs to one declared pool')
    rng = np.random.default_rng(231023)
    return np.asarray([[int(x) for key in keys for x in rng.choice(pools[key],2)]
                       for _ in range(steps)],dtype=np.int64)


def initial_reference(observation, preparation, *, now):
    """Accept an actual completed reference, never infer commands from measured width.

    This validates a context only; it does not grant motion permission or execute
    the high, empty-gripper preparation whose result the caller must supply.
    """
    state = np.asarray(observation.get('state'),float)
    target = np.asarray(preparation.get('target'),float)
    receipt = float(observation.get('receipt_monotonic_s',float('nan')))
    finished = float(preparation.get('finished_monotonic_s',float('nan')))
    if (state.shape!=(8,) or target.shape!=(8,)
            or not np.isfinite(np.r_[state,target,receipt,finished,now]).all()
            or preparation.get('accepted') is not True
            or preparation.get('result_status')!=4 or preparation.get('error_code')!=0
            or preparation.get('arm_held_verified') is not True
            or not finished<=receipt<=now or not 0<=now-receipt<=.15
            or not 0<=now-finished<=5
            or abs(target[7]-.095)>1e-7
            or np.max(abs(state[:7]-target[:7]))>.002
            or abs(state[7]-target[7])>.002):
        raise ValueError('Completed95mm preparation and fresh settled feedback required; no reference clipping')
    return target.copy()
