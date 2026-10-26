"""Exact center-scene contact recoveries and fresh independent physical cases."""
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nero_act_new_handeye_policy_repair_20261023'
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from simulation.collect import collect_episode,verify_replay,verify_action_contract
from training.legacy_visual_data import sha


def main():
    output=ROOT/'physical'
    output.mkdir(exist_ok=False)
    source=PREVIOUS/'results/9000'
    scene=json.loads((source/'scene_configuration.json').read_text())
    snapshots=json.loads((source/'physical_snapshots.json').read_text())['snapshots']
    result=json.loads((source/'result.json').read_text())
    grasp=next(e['physics_step'] for e in result['simulated_events'] if e['kind']=='GRASP')
    with np.load(source/'rollout.npz') as f:
        query_steps=f['query_steps']
    first=int(np.searchsorted(query_steps,grasp,side='right'))
    rows=[]
    for index,(query,mode) in enumerate([(first,'grasp_to_retreat'),(len(snapshots)-10,'lift_to_retreat'),(len(snapshots)-1,'lift_to_retreat')],9100):
        config=copy.deepcopy(scene);saved=copy.deepcopy(snapshots[query])
        config.update(initial_simulation_snapshot=saved,continuation=mode,
            initial_robot_state=dict(qpos_rad=saved['qpos'][:7],gripper_total_width_m=saved['qpos'][7]-saved['qpos'][8]),
            runtime_physics_offset=int(query_steps[query]))
        role='grasp' if mode=='grasp_to_retreat' else 'lift'
        rows.append(dict(id=index,role=role,split='train',parent_query=query,scene=config))
    for index,offset in enumerate([(.007,-.012),(-.012,.007)],9200):
        config=copy.deepcopy(scene)
        config['pick_xy_m']=(np.asarray(config['pick_xy_m'])+offset).tolist()
        rows.append(dict(id=index,role='independent',split='val',offset_xy_m=list(offset),scene=config))
    for row in rows:
        row['source_group']=f'new_handeye_contact_{row["id"]}'
        row['scene']['generalization']=dict(index=row['id'],split=row['split'],source_group=row['source_group'],seed=231023)
    (ROOT/'physical_protocol.json').write_text(json.dumps(rows,indent=2)+'\n')
    report=dict(complete=False,episodes=[],hardware_commands_sent=False,failed_ACT_actions_used_as_labels=False,
        source_sha256={str(p):sha(p) for p in [Path(__file__),source/'scene_configuration.json',source/'physical_snapshots.json',source/'rollout.npz',PREVIOUS/'results/summary.json']})
    for row in rows:
        path=output/f'episode_{row["id"]}.hdf5'
        result=collect_episode(row['scene'],path)
        replay,contract=verify_replay(path),verify_action_contract(path)
        if not result['success'] or not replay['passed'] or not contract['passed']:
            raise ValueError('Physical contact recovery/replay/contract failed')
        report['episodes'].append(dict(row,path=str(path),sha256=sha(path),replay=replay,
            action_contract=contract,physical_success=True,full_task_verified=result['full_task_verified']))
        (ROOT/'physical_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(id=row['id'],role=row['role'],accepted=True,frames=result['frames'])),flush=True)
    report['complete']=True
    (ROOT/'physical_manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
