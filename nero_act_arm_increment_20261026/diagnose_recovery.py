"""One offline expert recovery from B's saved stall; never ACT acceptance/data."""
from pathlib import Path
import copy,json,sys
import numpy as np
from experiment import BASE
from support import save_json,digest_file
from simulation.teacher import run_teacher

ROOT=Path(__file__).resolve().parent
case=ROOT/'evaluation/B/short_centre_0'
scene=json.loads((case/'scene_configuration.json').read_text())
saved=json.loads((case/'physical_snapshots.json').read_text())
snap=copy.deepcopy(saved['snapshots'][-1]);scene=copy.deepcopy(scene)
scene['initial_robot_state']['qpos_rad']=snap['qpos'][:7]
scene['initial_robot_state']['gripper_total_width_m']=snap['qpos'][7]-snap['qpos'][8]
scene['initial_simulation_snapshot']=snap
scene['runtime_physics_offset']=saved['query_steps'][-1]
result=run_teacher(scene)
trace=result.pop('trace')
report=dict(complete=True,diagnostic_only=True,source_query_step=saved['query_steps'][-1],
    source_snapshot_sha256=digest_file(case/'physical_snapshots.json'),
    source_scene_sha256=digest_file(case/'scene_configuration.json'),
    teacher_recovery_success=bool(result['success'] and result['full_task_verified']),
    physical_result=result,recorded_trace_points=len(trace),
    ACT_policy_used=False,ACT_full_task_passed=False,train_data_admitted=False,
    development_scene_diagnostic_only=True,independent_holdout_consumed=False,
    optimizer_updates=0,hardware_commands_sent=False)
save_json(ROOT/'expert_recovery_diagnosis.json',report)
print(json.dumps(report),flush=True)
