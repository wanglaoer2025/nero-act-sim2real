"""Audit perceptually near-identical expert states with different submitted targets."""
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent


def ambiguous_transition(first,second,action_first,action_second,rgb_first,rgb_second):
    first,second,a,b=[np.asarray(x,float) for x in [first,second,action_first,action_second]]
    if first.shape!=(15,) or second.shape!=(15,) or a.shape!=(8,) or b.shape!=(8,):
        raise ValueError('Physical15D descriptors and8D submitted expert actions required')
    tolerance=np.array([2e-5]*7+[1e-5,1e-5]+[1e-4]*6)
    return bool(np.all(abs(first-second)<=tolerance)
        and np.max(abs(np.asarray(rgb_first,int)-np.asarray(rgb_second,int)))<=1
        and np.max(abs(a[:7]-b[:7]))>=.001)


def main():
    sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
    from executor.ros_execution import RobotGeometry
    source=ROOT.parent/'nero_act_new_handeye_policy_repair_20261023/data/manifest.json'
    manifest=json.loads(source.read_text())
    rows=[];hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]}
    for row in manifest['episodes']:
        if row['split']!='train':continue
        path=Path(row['path']);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with h5py.File(path) as f:
            scene=json.loads(f.attrs['scene_config']);geometry=RobotGeometry(scene)
            q=f['observations/qpos'][:];width=f['observations/gripper_state'][:,0]
            actions=f['action'][:];visual=f['observations/visual_relative_m'][:]
            valid=f['observations/visual_valid'][:];stage=f['simulation/stage'].asstr()[:]
            close=np.flatnonzero((stage=='close')&valid)
            lift=np.flatnonzero((stage=='lift')&valid)
            t=next(int(i) for i in lift if geometry.pose(actions[i])['tcp'][2]-geometry.pose(np.r_[q[i],width[i]])['tcp'][2]>1e-5)
            def observed(i):return np.r_[q[i],width[i],actions[i-1,7]-width[i],visual[i]]
            pairs=[]
            rgb_t=f['observations/images/camera_0'][t]
            for c in close[-80:]:
                rgb_c=f['observations/images/camera_0'][c]
                if not ambiguous_transition(observed(c),observed(t),actions[c],actions[t],rgb_c,rgb_t):continue
                pairs.append(dict(close=int(c),lift=t,time_difference_s=float(f['timestamp'][t]-f['timestamp'][c]),
                    maximum_joint_difference_rad=float(np.max(abs(q[c]-q[t]))),
                    gripper_difference_m=float(abs(width[c]-width[t])),
                    previous_gripper_command_difference_m=float(abs(actions[c-1,7]-actions[t-1,7])),
                    geometry_difference_m=float(np.max(abs(visual[c]-visual[t]))),
                    rgb_max_difference=int(np.max(abs(rgb_c.astype(int)-rgb_t.astype(int)))),
                    first_arm_target_difference_rad=float(np.max(abs(actions[c,:7]-actions[t,:7])))))
            rows.append(dict(id=row['id'],pairs=pairs,near_ambiguous_pairs=len(pairs),
                first_lift_index=t,close_dwell_seconds=scene['dwell_seconds']['close']))
    for p,h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Expert data source changed')
    report=dict(complete=True,episodes=rows,source_sha256=hashes,
        episodes_with_near_ambiguous_transitions=sum(bool(r['pairs']) for r in rows),
        total_near_ambiguous_pairs=sum(len(r['pairs']) for r in rows),
        limitations='Engineering-resolution near-aliasing, not a proof of exact mathematical equality or sole causal attribution',
        data_modified=False,training_updates=0,hardware_deployment_allowed=False)
    (ROOT/'transition_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['episodes','source_sha256']}))


if __name__=='__main__':main()
