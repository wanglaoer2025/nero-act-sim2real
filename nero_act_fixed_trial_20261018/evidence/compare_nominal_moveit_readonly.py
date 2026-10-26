"""Historical MuJoCo nominal vs current matching MoveIt model; no scene writes/goals."""
from pathlib import Path
import json,copy,time,numpy as np,rclpy
from moveit_msgs.srv import GetPlanningScene,GetStateValidity
from executor.ros_execution import NAMES,RobotGeometry
from executor.fixed_permission import MODEL
source=Path('/home/user/wq/nero_act_fixed_scene_heads_20261018/evaluation/run27_current_scene14/nominal');result=json.loads((source/'result.json').read_text());stop=next(e['physics_step'] for e in result['simulated_events'] if e['kind']=='GRASP');states=np.load(source/'rollout.npz')['states'];config=json.loads((MODEL/'scene_config.json').read_text());geometry=RobotGeometry(config)
before=int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());rclpy.init();node=rclpy.create_node('act_nominal_collision_receive_only')
def wait(f):
 end=time.monotonic()+5
 while not f.done():
  if time.monotonic()>end:raise TimeoutError('read-only service timeout')
  rclpy.spin_once(node,timeout_sec=.005)
 if f.exception():raise f.exception()
 return f.result()
report=dict(motion_sent=False,scene_modified=False,original_mujoco_success=result['success'],checked=[])
try:
 sc=node.create_client(GetPlanningScene,'/get_planning_scene');valid=node.create_client(GetStateValidity,'/check_state_validity');assert sc.wait_for_service(timeout_sec=3) and valid.wait_for_service(timeout_sec=3)
 req=GetPlanningScene.Request();req.components.components=1023;scene=wait(sc.call_async(req)).scene
 cup=next(o for o in scene.world.collision_objects if o.id=='act_task_cup')
 positions=[]
 for pose in [cup.pose,cup.primitive_poses[0]]:
  q=pose.orientation;assert np.allclose([q.x,q.y,q.z,abs(q.w)],[0,0,0,1],atol=1e-6)
  positions.append([pose.position.x,pose.position.y,pose.position.z])
 center=np.sum(positions,axis=0);report['planning_model_cup_center']=center.tolist()
 assert np.allclose(center[:2],config['pick_xy_m'],atol=1e-6)
 for i in list(range(0,stop,20))+[stop-1]:
  req=GetStateValidity.Request();req.group_name='';req.robot_state=copy.deepcopy(scene.robot_state);req.robot_state.is_diff=False;req.robot_state.joint_state.name=NAMES;req.robot_state.joint_state.position=states[i].tolist();req.robot_state.joint_state.velocity=[];req.robot_state.joint_state.effort=[]
  res=wait(valid.call_async(req));item=dict(index=i,valid=res.valid,tcp_xyz_mm=(geometry.pose(states[i])['tcp']*1000).tolist(),contacts=[dict(body1=c.contact_body_1,body2=c.contact_body_2,depth_m=c.depth) for c in res.contacts]);report['checked'].append(item)
  if not res.valid:report['first_invalid']=item;break
 report.update(can_tx_before=before,can_tx_after=int(Path('/sys/class/net/can0/statistics/tx_packets').read_text()))
 Path('evidence/nominal_simulation_moveit_pregrasp_comparison_01.json').open('x').write(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checked'}))
finally:node.destroy_node();rclpy.shutdown()
