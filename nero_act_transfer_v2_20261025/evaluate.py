"""Complete-task evaluation using the retained protected physics runner."""
import copy
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from protocol import plan_positions

ROOT=Path(__file__).resolve().parent


def development_cases():
    rows=[dict(id=f'centre_{i}',offset_mm=[0,0],seed=251900+i) for i in range(3)]
    for i,xy in enumerate(plan_positions()['positions_mm']['dev'][:4]):
        rows.append(dict(id=f'dev_{i}',offset_mm=xy,seed=251910+i))
    return rows


def full_task_passed(r):
    return bool(r.get('success') is True and r.get('error') is None
                and r.get('teacher_free') is True and r.get('full_task_from_fresh_start') is True
                and not r.get('resumed_partial_not_full',False)
                and r.get('max_lift_m',0)>=.18 and r.get('maximum_horizontal_transfer_m',0)>.1
                and r.get('stable_released_frames',0)>=100
                and r.get('unexpected_arm_cup_contact_frames',1)==0
                and r.get('execution_contract_failures',1)==0
                and not r.get('premature_release_before_support',True)
                and not r.get('constraint_attachment_used',False)
                and r.get('active_policy_points',4000)<=3900
                and [e['kind'] for e in r.get('simulated_events',[])]==['GRASP','CARRY','RELEASE','RETREAT'])


def load_runner():
    sys.path.insert(0,str(ROOT/'vendor'))
    from supervised_contact_escape import rollout
    return rollout


def development_suite(manifest):
    cases=[dict(**r,role='readiness') for r in development_cases()]
    dev=[r for r in manifest['episodes'] if r['split']=='dev']
    if len(dev)!=30:raise ValueError('All30 development demonstration conditions required')
    for i,row in enumerate(dev):
        with h5py.File(row['path']) as f:scene=json.loads(f.attrs['scene_config'])
        cases.append(dict(id=f'full_dev_{i:02d}',scene=scene,seed=252000+i,role='full_development'))
    old=json.loads((ROOT/'historical_development_cases.json').read_text())
    for row in old['cases']:
        cases.append(dict(id=f'historical_{row["id"]}',scene=row['scene'],seed=191899,role='historical_regression'))
    return cases


def evaluate_development(policy,stats,manifest,output):
    rollout=load_runner()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with h5py.File(manifest['source_scene']) as f:base=json.loads(f.attrs['scene_config'])
    records=[]
    for row in development_suite(manifest):
        scene=copy.deepcopy(row.get('scene',base))
        if row['role']=='readiness':
            scene['pick_xy_m']=(np.asarray(base['pick_xy_m'])+np.asarray(row['offset_mm'])/1000).tolist()
            rng=np.random.default_rng(row['seed'])
            scene['initial_robot_state']['qpos_rad']=(np.asarray(base['initial_robot_state']['qpos_rad'])+rng.uniform(-.0002,.0002,7)).tolist()
        scenario=dict(seed=row['seed'],query_stride=2,servo_delay_steps=0,
                      observation_age_s=.05,jitter=False,force=False)
        folder=output/row['id']
        result=rollout(scene,policy,stats,scenario,folder,runtime_profile='history_common_v1')
        records.append(dict(**{k:v for k,v in row.items() if k!='scene'},passed=full_task_passed(result),result=result))
        (output/'progress.json').write_text(json.dumps(dict(complete=False,records=records),indent=2)+'\n')
    centre=sum(r['passed'] for r in records[:3])
    nearby=int(records[0]['passed'])+sum(r['passed'] for r in records[3:7])
    phase_progress=[sum(any(e['kind']==kind for e in r['result'].get('simulated_events',[])) for r in records)
                    for kind in ['RETREAT','RELEASE','CARRY','GRASP']]
    report=dict(complete=True,records=records,passed=sum(r['passed'] for r in records),total=len(records),
                phase_progress=phase_progress,
                development_only=True,independent_holdout_consumed=False,
                adaptation_gate_passed=centre==3 and nearby>=4,hardware_commands_sent=False)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
