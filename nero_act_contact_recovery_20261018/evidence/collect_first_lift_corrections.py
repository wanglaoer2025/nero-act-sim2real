"""Bounded offline dataset job: exact failed run14 states, no hardware imports."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay
from executor.ros_execution import RobotGeometry

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'nero_act_generalized_pick_place_20261018/evaluation/run14_generalization16'
OUT=ROOT/'training_data/recoveries_v1'
CASES=[(1001,'cup_+0_+0','train','run14_nominal_failure'),
 (1002,'last_site_geometry_prepared_open_0','train','run14_historical_site_failure'),
 (1003,'last_site_geometry_prepared_open_1','train','run14_historical_site_failure'),
 (1004,'last_site_geometry_prepared_open_2','train','run14_historical_site_failure'),
 (1005,'cup_+0_-20','val','run14_cup_0_minus20_failure'),
 (1006,'cup_+20_-20','val','run14_cup_plus20_minus20_failure')]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest=dict(episodes=[],rejected=[],role='partial_first_lift_recovery_not_full_tasks',
        source_evaluation=str(BASE),source_protocol_sha256=sha(BASE/'protocol.json'),
        source_checkpoint_sha256=json.loads((BASE/'protocol.json').read_text())['checkpoint_sha256'],
        original_16_cases_now_development_only=True)
    for index,name,split,group in CASES:
        origin=BASE/name;path=OUT/f'episode_{index}.hdf5';start=time.monotonic()
        try:
            if index==1001:
                config=json.loads(path.with_suffix('.scene.json').read_text())
                result=json.loads(path.with_suffix('.result.json').read_text())
                from simulation.collect import verify_action_contract
                result['prefix_contract_audit']=verify_action_contract(path)
            else:
                assert not path.exists() and not path.with_suffix('.result.json').exists()
                config=json.loads((origin/'scene_configuration.json').read_text())
                record=json.loads((origin/'physical_snapshots.json').read_text());snap=record['snapshots'][-1]
                state=np.r_[np.array(snap['qpos'])[:7],snap['qpos'][7]-snap['qpos'][8]]
                config['initial_robot_state']=dict(qpos_rad=state[:7].tolist(),gripper_total_width_m=float(state[7]))
                config.update(initial_simulation_snapshot=snap,continuation='lift_to_retreat',
                    joint_speed_limit_rad_s=.02,teacher_command_lead_limit_rad=.015,
                    gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
                    ik_goal_rotation_link7=RobotGeometry(config).pose(state)['rotation'].tolist(),
                    correction_source=dict(origin=str(origin),query_step=record['query_steps'][-1],split=split,
                        source_group=group,role='partial_not_full',reason='correct first downward prefix after measured grasp'))
                config['dwell_seconds'].update(close=8,release=8)
                path.with_suffix('.scene.json').write_text(json.dumps(config,indent=2))
                result=collect_episode(config,path)
            replay=verify_replay(path)
            assert result['success'] and result['full_task_verified'] is False and replay['passed']
            assert result['prefix_contract_audit']['passed']
            row=dict(id=index,path=str(path),sha256=sha(path),status='valid',split=split,source_group=group,
                teacher_success=True,full_task_verified=False,correction_verified=True,task='lift_to_retreat_correction',
                origin=str(origin),query_step=config['correction_source']['query_step'],
                snapshot_sha256=sha(origin/'physical_snapshots.json'),scene_sha256=sha(origin/'scene_configuration.json'),
                replay=replay,prefix_contract_audit=result['prefix_contract_audit'],elapsed_s=time.monotonic()-start)
            manifest['episodes'].append(row)
            print(json.dumps(dict(id=index,accepted=True,frames=result['frames'],elapsed_s=row['elapsed_s'])),flush=True)
        except Exception as e:
            manifest['rejected'].append(dict(id=index,name=name,error=str(e)))
            print(json.dumps(dict(id=index,accepted=False,error=str(e))),flush=True)
    manifest['complete']=len(manifest['episodes'])==6 and not manifest['rejected']
    target=OUT/'manifest.json'
    with target.open('x') as f:json.dump(manifest,f,indent=2)
    if not manifest['complete']:raise SystemExit('Not all corrections passed; do not train')

if __name__=='__main__':main()
