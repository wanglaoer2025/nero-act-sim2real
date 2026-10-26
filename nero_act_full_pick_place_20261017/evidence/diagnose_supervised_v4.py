import json,numpy as np
from pathlib import Path
from executor.ros_execution import RobotGeometry
from executor.release_hold import first_lift_index
from simulation.supervised_rollout import PhysicsContact
p=Path('evidence/run9_supervised_development_v4/nominal');t=np.load(p/'rollout.npz');m=json.loads((p/'measurements.json').read_text());cfg=json.loads((p/'scene_configuration.json').read_text());g=RobotGeometry(cfg)
c=PhysicsContact();c.close_authorized=True
held=None
for i,x in enumerate(m):
 c.update(x,t['sent_targets'][i,7],(i+1)*.05)
 if held is None and c.state=='holding' and i+1 in t['query_steps']:
  k=list(t['query_steps']).index(i+1);s=t['feedback_at_query'][k];rows=t['raw_chunks'][k,:10];first=first_lift_index(g.pose(s)['tcp'],np.array([g.pose(r)['tcp'] for r in rows]))
  if first==0:
   z=g.pose(s);held=z['rotation'].T@(np.r_[cfg['pick_xy_m'],.0475]-z['position']);rot=z['rotation'].T
   print('ATTACH',i+1,'tcp',z['tcp'],'actualcup',x['cup_xyz']);break
for k in [-1,-2,-3,-5,-10,-20,-50,-100]:
 s=t['states'][k];z=g.pose(s);center=z['position']+z['rotation']@held;axis=z['rotation']@rot[:,2];bottom=center[2]-.0475*abs(axis[2])-.0375*np.sqrt(max(0,1-axis[2]**2))
 print(k,'tcp',z['tcp'],'modelbottom',bottom,'physics', {a:m[k][a] for a in ['cup_xyz','supported','cup_upright','bilateral','gripper_width']})
print('RAW last',np.array([g.pose(r)['tcp'] for r in t['raw_chunks'][-1,:10]]))
