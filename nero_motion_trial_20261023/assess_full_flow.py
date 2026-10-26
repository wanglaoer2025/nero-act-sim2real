"""One offline diagnostic from the last measured stop; never opens a robot connection."""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import select
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'nero_act_paired_response_20261023'
sys.path.insert(0,str(SOURCE))
from experiment import load_latest,require_unchanged,snapshot
from simulation.supervised_rollout import rollout
from simulation.visual_evaluation import full_task_passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'full_flow_request_01')
    output=parser.parse_args().output;output.mkdir(exist_ok=False)
    source=ROOT/'postflight_02/preflight_01/report.json'
    capture=json.loads(source.read_text())
    if capture.get('error') or capture['can_tx_before']!=capture['can_tx_after']:
        raise ValueError('Clean receive-only stopped-state capture required')
    measured=capture['final_SDK_health']
    if measured['feedback']['motion_status']!=0 or measured['feedback']['err_code']!=0:
        raise ValueError('Capture was not a healthy stationary stop')
    scene_path=ROOT/'preparation_01/ACT_source/scene_configuration.json'
    scene=json.loads(scene_path.read_text())
    worker_path=ROOT.parent/'nero_motion_request_20261022/existing_calibration_transfer_01/perception_worker_table.py'
    with (output/'perception.log').open('w') as log:
        worker=subprocess.Popen(['/home/user/nero/bin/python','-u',str(worker_path),str(scene_path),str(output)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            def receive():
                if not select.select([worker.stdout],[],[],45)[0]:raise TimeoutError('Saved-frame observer timeout')
                result=json.loads(worker.stdout.readline())
                if result.get('type')=='error':raise ValueError(result['error'])
                return result
            if receive()!={'type':'ready'}:raise ValueError('Observer handshake failed')
            worker.stdin.write(json.dumps(dict(index=0,camera_frame=capture['camera']['frame'],
                camera_received_monotonic_s=capture['camera']['received_monotonic']))+'\n');worker.stdin.flush()
            observation=receive()
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                try:worker.wait(timeout=3)
                except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=3)
    (output/'scene_observation.json').write_text(json.dumps(observation,indent=2)+'\n')
    scene['pick_xy_m']=observation['cup'][:2];scene['place_xy_m']=observation['target'][:2]
    scene['visual_marker_xy_m']=scene['place_xy_m']
    scene['initial_robot_state']=dict(qpos_rad=measured['state'][:7],gripper_total_width_m=measured['state'][7])
    checkpoint=SOURCE/'training/pilot_A/policy_last.ckpt'
    proof=json.loads((checkpoint.parent/'integrity.json').read_text())
    require_unchanged(proof['source_sha256']);require_unchanged(proof['checkpoint_sha256'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['profile']!='paired_position_response_v1' or payload['actual_updates']!=2000:
        raise ValueError('Wrong frozen ACT checkpoint')
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    policy,stats,_,_=load_latest();policy.load_state_dict(payload['model'],strict=True);policy.requires_grad_(False).eval()
    hashes=snapshot([Path(__file__),source,Path(capture['camera']['frame']),scene_path,worker_path,checkpoint,
        checkpoint.parent/'integrity.json',Path(sys.modules[rollout.__module__].__file__)])
    result=rollout(copy.deepcopy(scene),policy,stats,dict(name='last_real_stop_full_flow_diagnostic_only',seed=191899,
        query_stride=2,servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False),
        output/'rollout',runtime_profile='history_common_v1')
    require_unchanged(hashes)
    summary=dict(diagnostic_only=True,complete_task_qualification=False,full_flow_in_this_nominal_case_passed=full_task_passed(result),
        source_real_capture_time=capture['time_local'],camera_live_during_simulation=False,
        initial_state=measured['state'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        calibration_verified=False,hardware_deployment_allowed=False,hardware_connected=False,hardware_commands_sent=False,
        source_sha256=hashes,result={k:result.get(k) for k in ['success','error','frames','active_policy_points','queries',
            'max_lift_m','maximum_horizontal_transfer_m','stable_released_frames','unexpected_arm_cup_contact_frames',
            'execution_contract_failures','premature_release_before_support','simulated_events','runtime_s']})
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['result']),flush=True)


if __name__=='__main__':main()
