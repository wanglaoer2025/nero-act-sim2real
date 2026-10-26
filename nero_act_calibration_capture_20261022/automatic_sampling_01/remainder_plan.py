"""Offline remainder of the authorized route, with no new goal or device access."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automatic_sampling import flange_fk,validate_route
from capture import dump

parent=ROOT/'automatic_sampling_01/nominal_plan.json'
plan=json.loads(parent.read_bytes())
log=[json.loads(line) for line in (ROOT/'automatic_sampling_01/execution_01/control_events.jsonl').read_text().splitlines()]
sent=[v for v in log if v['kind']=='arm' and v['state']=='returned']
assert len(sent)==71 and sent[-1]['details']['stage']==1 and sent[-1]['details']['waypoint']==34
health_path=ROOT/'automatic_sampling_01/stop_stream_01/report.json'
health=json.loads(health_path.read_text())['final_health']
state=np.asarray(health['state'])
assert np.max(abs(state[:7]-sent[-1]['details']['target']))<.0007
assert abs(state[7]-plan['initial_state'][7])<.000001

new=copy.deepcopy(plan)
remaining=plan['stages'][1]['waypoints_rad'][35:]
new['initial_state']=state.tolist()
new['stages'][0]=dict(index=0,role='prepare',goal_q_rad=remaining[0],
    goal_transform=flange_fk(remaining[0]).tolist(),waypoints_rad=remaining[:1])
new['stages'][1]['waypoints_rad']=remaining[1:]
new['waypoint_count']=sum(len(s['waypoints_rad']) for s in new['stages'])
original_flat=np.array([q for s in plan['stages'] for q in s['waypoints_rad']])
remaining_flat=np.array([q for s in new['stages'] for q in s['waypoints_rad']])
assert np.array_equal(remaining_flat,original_flat[71:])

# Reuse the exact checked geometry functions without its route-generation/output tail.
route_source=ROOT/'auto_plan_probe_01/route.py'
prefix=route_source.read_text().partition("\npool=result['pool'];endpoint_bad={}")[0]
assert 'def edge(a,b):' in prefix
scope={'__file__':str(route_source)}
exec(compile(prefix,str(route_source),'exec'),scope)
previous=state[:7];checks=0;errors=[]
for stage in new['stages']:
    # Stages retain the original straight joint interpolation; include measured start.
    bad,count=scope['edge'](previous,np.array(stage['waypoints_rad'][0]));checks+=count
    if bad:errors.append(dict(stage=stage['index'],edge='measured_or_previous_to_first',reason=bad))
    bad,count=scope['edge'](np.array(stage['waypoints_rad'][0]),np.array(stage['goal_q_rad']));checks+=count
    if bad:errors.append(dict(stage=stage['index'],edge='remaining_stage',reason=bad))
    previous=np.array(stage['goal_q_rad'])
new.update(checks=checks,route_errors=errors,nominal_route_passed=not errors,
    continuation_of_plan_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    completed_parent_arm_targets=71,exact_remaining_parent_targets=True,
    current_pose_source=str(health_path),actual_motion_sent=False)
for source in [parent,health_path,Path(__file__)]:
    new['source_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
validate_route(new)
dump(ROOT/'automatic_sampling_01/remainder_plan.json',new)
print(json.dumps(dict(nominal_route_passed=not errors,checks=checks,remaining_targets=new['waypoint_count'],
    new_goals=0,automatic_return=False,errors=errors,hardware_connected=False)))
