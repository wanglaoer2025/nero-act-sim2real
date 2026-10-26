"""Offline physical labels at the restored start and ACT-visited approach poses."""
import copy
import importlib.util
import json
from pathlib import Path
import select
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'nero_act_position_response_repair_20261023'
sys.path.insert(0,str(PRIOR))
import prepare as prefix_source
from simulation.collect import collect_episode,verify_replay,verify_action_contract
from training.legacy_visual_data import derive_episode,training_profile,sha


def main():
    output=ROOT/'data';output.mkdir(exist_ok=False)
    capture_path=ROOT/'current_capture/preflight_01/report.json'
    capture=json.loads(capture_path.read_text())
    if capture.get('error') or capture['can_tx_before']!=capture['can_tx_after']:raise ValueError('Clean capture required')
    previous=ROOT.parent/'nero_motion_trial_20261023/full_flow_request_02/rollout'
    scene_path=previous/'scene_configuration.json';scene=json.loads(scene_path.read_text())
    worker_path=ROOT.parent/'nero_motion_request_20261022/existing_calibration_transfer_01/perception_worker_table.py'
    with (output/'perception.log').open('w') as log:
        worker=subprocess.Popen(['/home/user/nero/bin/python','-u',str(worker_path),str(scene_path),str(output)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            def receive():
                if not select.select([worker.stdout],[],[],45)[0]:raise TimeoutError('Saved-frame observer timeout')
                value=json.loads(worker.stdout.readline())
                if value.get('type')=='error':raise ValueError(value['error'])
                return value
            if receive()!={'type':'ready'}:raise ValueError('Observer handshake failed')
            worker.stdin.write(json.dumps(dict(index=0,camera_frame=capture['camera']['frame'],
                camera_received_monotonic_s=capture['camera']['received_monotonic']))+'\n');worker.stdin.flush()
            observed=receive()
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                try:worker.wait(timeout=3)
                except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=3)
    (output/'scene_observation.json').write_text(json.dumps(observed,indent=2)+'\n')
    scene.update(pick_xy_m=observed['cup'][:2],place_xy_m=observed['target'][:2],visual_marker_xy_m=observed['target'][:2],
        initial_robot_state=dict(qpos_rad=capture['final_SDK_health']['state'][:7],
            gripper_total_width_m=capture['final_SDK_health']['state'][7]),
        action_delay_steps=0,joint_speed_limit_rad_s=.03,teacher_command_lead_limit_rad=.016,
        gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008)
    scene['dwell_seconds'].update(close=8,release=8)
    scene.pop('initial_simulation_snapshot',None);scene.pop('arm_disturbance',None)
    scene['generalization']=dict(index=4000,split='train',source_group='full_repair_restored_current',seed=231019,
        source_capture_sha256=sha(capture_path),geometry_independently_verified=False)
    (ROOT/'current_scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    snapshots=json.loads((previous/'physical_snapshots.json').read_text())['snapshots']
    specs=[(4000,'complete',copy.deepcopy(scene))]
    for index,query in enumerate([0,70,150,220,262,350,430,515]):
        config=copy.deepcopy(scene);config['initial_robot_state']=dict(qpos_rad=snapshots[query]['qpos'][:7],gripper_total_width_m=.095)
        config['generalization']=dict(index=4010+index,split='train',source_group=f'full_repair_query_{query}',
            parent_query=query,parent_snapshots_sha256=sha(previous/'physical_snapshots.json'),
            label_source='physical_teacher_not_failed_ACT_actions',synthetic_stationary_initialization=True)
        specs.append((4010+index,'correction',config))
    (output/'protocol.json').write_text(json.dumps([dict(id=eid,role=role,scene=config) for eid,role,config in specs],indent=2)+'\n')
    manifest=dict(complete=False,episodes=[],rejected=[],hardware_deployment_allowed=False,failed_ACT_actions_used_as_labels=False)
    for eid,role,config in specs:
        started=time.monotonic();physical=output/'physical'/f'episode_{eid}.hdf5';physical.parent.mkdir(exist_ok=True)
        try:
            result=collect_episode(config,physical) if role=='complete' else prefix_source.collect_prefix(config,physical)
            if role=='complete' and not result.get('full_task_verified'):raise ValueError('Incomplete full teacher')
            replay=verify_replay(physical);contract=verify_action_contract(physical)
            if not replay['passed'] or not contract['passed']:raise ValueError('Teacher replay/contract failed')
            report=derive_episode(physical,output/'visual'/physical.name,expected_sha=sha(physical),profile=training_profile(config))
            manifest['episodes'].append(dict(report,id=eid,role=role,split='train',
                source_group=config['generalization']['source_group'],physical_path=str(physical),physical_sha256=sha(physical),
                replay=replay,prefix_contract_audit=contract,full_task_verified=role=='complete'))
            print(json.dumps(dict(id=eid,role=role,accepted=True,elapsed_s=time.monotonic()-started)),flush=True)
        except (ValueError,RuntimeError) as exc:
            manifest['rejected'].append(dict(id=eid,role=role,error=repr(exc)))
            print(json.dumps(manifest['rejected'][-1]),flush=True)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not any(r['role']=='complete' for r in manifest['episodes']) or sum(r['role']=='correction' for r in manifest['episodes'])<4:
        raise ValueError('One full verified demonstration and at least4 safe visited-state corrections required')
    manifest['complete']=True
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
