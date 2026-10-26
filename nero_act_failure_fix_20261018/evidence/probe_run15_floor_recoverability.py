"""Two exact-state physical probes; no changes to current training or guards."""
import json,time
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay
from executor.ros_execution import RobotGeometry
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'nero_act_contact_recovery_20261018/evaluation'
OUT=ROOT/'evidence/run15_floor_recoverability';OUT.mkdir(exist_ok=False)
reports=[]
for suite,name in [('run15_development16','last_site_geometry_prepared_open_1'),('run15_independent16','fresh_initial_pose_3')]:
 origin=OLD/suite/name;folder=OUT/name;folder.mkdir()
 c=json.loads((origin/'scene_configuration.json').read_text());r=json.loads((origin/'physical_snapshots.json').read_text())
 snap=r['snapshots'][-1];q=np.array(snap['qpos']);state=np.r_[q[:7],q[7]-q[8]]
 c['initial_robot_state']=dict(qpos_rad=state[:7].tolist(),gripper_total_width_m=float(state[7]))
 c.update(initial_simulation_snapshot=snap,continuation='lift_to_retreat',joint_speed_limit_rad_s=.02,
  teacher_command_lead_limit_rad=.015,gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
  ik_goal_rotation_link7=RobotGeometry(c).pose(state)['rotation'].tolist(),
  correction_source=dict(origin=str(origin),query_step=r['query_steps'][-1],role='diagnostic_only_excluded_from_training'))
 c['dwell_seconds'].update(close=8,release=8);(folder/'config.json').write_text(json.dumps(c,indent=2))
 try:
  result=collect_episode(c,folder/'diagnostic.hdf5');replay=verify_replay(folder/'diagnostic.hdf5')
  report=dict(case=name,teacher_success=result['success'],full_task_verified=False,included_in_training=False,
   replay=replay,prefix_contract_audit=result['prefix_contract_audit'],max_lift_m=result['max_lift_m'],transfer_m=result['maximum_horizontal_transfer_m'])
 except Exception as e:report=dict(case=name,teacher_success=False,error=str(e),included_in_training=False)
 reports.append(report);(folder/'diagnostic_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
(OUT/'summary.json').write_text(json.dumps(reports,indent=2))
