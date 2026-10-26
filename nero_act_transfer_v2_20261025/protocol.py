"""Predeclared spatial groups, independent of trial outcomes."""
import numpy as np


def plan_positions():
    train=[[-30,-30],[-30,0],[-30,30],[-10,-10],[-10,10],[10,-10],[10,10],
           [30,-30],[30,0],[30,30],[0,-30],[0,0]]
    dev=[[-20,-20],[-20,20],[20,-20],[20,20],[-20,0],[20,0]]
    test=[[x,y] for x in [-15,-5,5,15] for y in [-15,-5,5,15]]
    test += [[0,-15],[0,15],[-15,0],[15,0]]
    result=dict(schema='nero_act_transfer_v2_positions',coordinate_frame='robot_base_XY',
                positions_mm=dict(train=train,dev=dev,test=test),
                episode_counts=dict(train=120,dev=30,test=40),
                cross_group_exclusion_mm=5.,target_card_fixed=True,
                geometric_reachability_verified=False,final_test_consumed=False)
    validate_positions(result)
    return result


def validate_positions(protocol):
    groups=protocol['positions_mm']
    if set(groups)!={'train','dev','test'} or {k:len(v) for k,v in groups.items()}!={'train':12,'dev':6,'test':20}:
        raise ValueError('Declared12/6/20 spatial groups required')
    for key,points in groups.items():
        a=np.asarray(points,float)
        limit=30 if key=='train' else 20
        if a.shape!=(len(points),2) or not np.isfinite(a).all() or abs(a).max()>limit:
            raise ValueError('Spatial bounds invalid')
        if len(np.unique(a,axis=0))!=len(a):
            raise ValueError('Duplicated position')
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        distances=np.linalg.norm(np.asarray(groups[a])[:,None]-np.asarray(groups[b])[None],axis=2)
        if distances.min()<5.-1e-8:
            raise ValueError('Position leakage within5mm exclusion')
    return True


def validate_final_positions(final,collection,historical):
    points=np.asarray(final['positions_mm'],float)
    if (points.shape!=(20,2) or not np.isfinite(points).all() or abs(points).max()>20
            or len(np.unique(points,axis=0))!=20 or final['consumed'] is not False):
        raise ValueError('Twenty unconsumed final positions within20mm required')
    forbidden=np.asarray(collection['positions_mm']['train']+collection['positions_mm']['dev']+
        [[1000*x for x in r['offset_xy_m']] for r in historical['cases']])
    if np.linalg.norm(points[:,None]-forbidden[None],axis=2).min()<5.-1e-8:
        raise ValueError('Final positions must exclude all development/history within5mm')
    return True
