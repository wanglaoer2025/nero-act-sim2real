"""Offline expert recovery from exact failed ACT state, not runtime takeover."""
import copy,json,time
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay
from executor.ros_execution import RobotGeometry
root=Path(__file__).resolve().parents[1]
origin=root.parent/'nero_act_generalized_pick_place_20261018/evaluation/run14_generalization16/cup_+0_+0'
out=root/'training_data/recoveries_v1';out.mkdir(exist_ok=False)
config=json.loads((origin/'scene_configuration.json').read_text());record=json.loads((origin/'physical_snapshots.json').read_text())
snapshot=record['snapshots'][-1];q=np.array(snapshot['qpos']);state=np.r_[q[:7],q[7]-q[8]]
config['initial_robot_state']=dict(qpos_rad=state[:7].tolist(),gripper_total_width_m=float(state[7]))
config.update(initial_simulation_snapshot=snapshot,continuation='grasp_to_retreat',joint_speed_limit_rad_s=.02,
 teacher_command_lead_limit_rad=.015,gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
 ik_goal_rotation_link7=RobotGeometry(config).pose(state)['rotation'].tolist(),
 correction_source=dict(origin=str(origin),query_step=record['query_steps'][-1],split='train',source_group='run14_nominal_failure',
 role='partial_not_full',reason='remove erroneous first downward prefix after measured grasp'))
config['dwell_seconds'].update(close=8,release=8)
(out/'episode_1000.scene.json').write_text(json.dumps(config,indent=2));started=time.monotonic()
result=collect_episode(config,out/'episode_1000.hdf5');replay=verify_replay(out/'episode_1000.hdf5')
report=dict(success=result['success'],full_task_verified=result['full_task_verified'],frames=result['frames'],replay=replay,
 prefix_contract_audit=result['prefix_contract_audit'],elapsed_s=time.monotonic()-started)
(out/'pilot_acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
