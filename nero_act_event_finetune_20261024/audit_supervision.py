"""Measure and remove engineering-resolution WAIT/LIFT ambiguity from query sampling."""
import json
from pathlib import Path
import h5py
import numpy as np
from common import ROOT,historical
from executor.ros_execution import RobotGeometry
from supervision import remove_wait_aliases


def audit_episode(path):
    with h5py.File(path) as f:
        scene=json.loads(f.attrs['scene_config']);geometry=RobotGeometry(scene)
        q=f['observations/qpos'][:];width=f['observations/gripper_state'][:,0]
        actions=f['action'][:];features=f['observations/visual_relative_m'][:]
        valid=f['observations/visual_valid'][:];stages=f['simulation/stage'].asstr()[:]
        previous=np.r_[f.attrs['initial_previous_gripper_target_m'],actions[:-1,7]]
        observed=np.c_[q,width,previous-width,features]
        lift=next(int(i) for i in np.flatnonzero((stages=='lift')&valid)
            if geometry.pose(actions[i])['tcp'][2]-geometry.pose(np.r_[q[i],width[i]])['tcp'][2]>1e-5)
        mask,removed=remove_wait_aliases(observed,actions,f['observations/images/camera_0'],stages,valid,lift)
    return dict(first_lift_index=lift,removed_close_query_starts=removed,
        training_query_starts=np.flatnonzero(mask).tolist(),
        selected_stage_counts={s:int(np.count_nonzero(mask&(stages==s))) for s in set(stages)})


def main():
    source=ROOT/'data/manifest.json';manifest=json.loads(source.read_text())
    if not manifest['complete'] or manifest['rejected']:raise ValueError('Physical and RGB data gate failed')
    rows=[]
    for row in manifest['episodes']:
        report=audit_episode(row['path'])
        if any(report['selected_stage_counts'].get(s,0)==0 for s in ['approach','descend','close','lift','carry','place','release','retreat']):
            raise ValueError('Ambiguity exclusion removed an entire stage')
        rows.append(dict(id=row['id'],path=row['path'],**report))
    report=dict(complete=True,episodes=rows,physical_labels_modified=False,
        selection='exclude_only_close_query_starts_matching_first_lift_under_original_ambiguity_test',
        raw_ambiguous_pairs=sum(len(r['removed_close_query_starts']) for r in rows),
        retained_ambiguous_close_query_starts=0,hardware_deployment_allowed=False,
        source_sha256=historical.snapshot([source,*[Path(r['path']) for r in rows],
            Path(__file__),ROOT/'supervision.py',ROOT.parent/'nero_act_observation_repair_20261023/transition_audit.py']))
    (ROOT/'supervision_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','episodes']}),flush=True)


if __name__=='__main__':main()
