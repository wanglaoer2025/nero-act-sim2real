"""Offline state queries with current cup geometry; restores planning object; no robot goals."""
from pathlib import Path
import copy,json,time,sys
import numpy as np
import rclpy
from moveit_msgs.srv import GetPlanningScene,GetStateValidity,ApplyPlanningScene
from moveit_msgs.msg import PlanningScene
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from executor.ros_execution import NAMES,checked_interpolation,RobotGeometry
from executor.fixed_permission import MODEL
out=ROOT/'evidence/current_cup_pregrasp_moveit_01';out.mkdir(exist_ok=False)
fit=json.loads((ROOT/'evidence/retry_restored_geometry_01/report.json').read_text())['measured']
case=ROOT/'evidence/retry_current_cup_sim_01/rollout'
result=json.loads((case/'result.json').read_text());scene_config=json.loads((case/'scene_configuration.json').read_text());geometry=RobotGeometry(scene_config)
stop=next(e['physics_step'] for e in result['simulated_events'] if e['kind']=='GRASP')
z=np.load(case/'rollout.npz');targets=z['sent_targets'][:stop];states=z['states'][:stop]
report=dict(mode='offline_MoveIt_current_cup_pregrasp',motion_sent=False,robot_goals_sent=False,
 source_cup_xyz=fit['cup_xyz'],full_mujoco_task_success=result['success'],success=False,
 original_scene_restored=False,checked_states=0)
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
before=tx();rclpy.init();node=rclpy.create_node('act_current_cup_collision_validation_only')
sc=node.create_client(GetPlanningScene,'/get_planning_scene');valid=node.create_client(GetStateValidity,'/check_state_validity');apply=node.create_client(ApplyPlanningScene,'/apply_planning_scene')
original=None;updated=False

def guard():
 if node.count_publishers('/control/gated_joint_states') or tx()!=before:
  raise RuntimeError('Physical control route/TX appeared during offline collision validation')
def wait(f,timeout=5):
 end=time.monotonic()+timeout
 while not f.done():
  if time.monotonic()>end:raise TimeoutError('Planning service timeout')
  rclpy.spin_once(node,timeout_sec=.002)
 if f.exception():raise f.exception()
 return f.result()
def get_scene():
 req=GetPlanningScene.Request();req.components.components=1023
 return wait(sc.call_async(req)).scene

def apply_cup(cup):
 diff=PlanningScene(is_diff=True);diff.robot_state.is_diff=True
 diff.world.collision_objects=[cup]
 if not wait(apply.call_async(ApplyPlanningScene.Request(scene=diff))).success:
  raise RuntimeError('Planning cup update rejected')
def query(row):
 request=GetStateValidity.Request();request.group_name=''
 request.robot_state=copy.deepcopy(original.robot_state);request.robot_state.is_diff=False
 request.robot_state.joint_state.name=NAMES;request.robot_state.joint_state.position=list(row)
 request.robot_state.joint_state.velocity=[];request.robot_state.joint_state.effort=[]
 return valid.call_async(request)
def check(rows,label):
 for offset in range(0,len(rows),8):
  guard();batch=rows[offset:offset+8];futures=[query(row) for row in batch]
  for index,(row,f) in enumerate(zip(batch,futures)):
   response=wait(f);report['checked_states']+=1
   if not response.valid:
    report['first_invalid']=dict(path=label,index=offset+index,state=list(row),
     tcp_xyz_mm=(geometry.pose(row)['tcp']*1000).tolist(),
     contacts=[dict(body1=c.contact_body_1,body2=c.contact_body_2,depth_m=c.depth) for c in response.contacts])
    return False
 return True
try:
 assert sc.wait_for_service(timeout_sec=3) and valid.wait_for_service(timeout_sec=3) and apply.wait_for_service(timeout_sec=3)
 deadline=time.monotonic()+.5
 while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.01)
 guard();original=get_scene();cup=copy.deepcopy(next(o for o in original.world.collision_objects if o.id=='act_task_cup'))
 assert len(cup.primitives)==len(cup.primitive_poses)==1 and not original.robot_state.attached_collision_objects
 assert cup.header.frame_id in ['world','']
 for pose in [cup.pose,cup.primitive_poses[0]]:
  q=pose.orientation;assert np.allclose([q.x,q.y,q.z,abs(q.w)],[0,0,0,1],atol=1e-6)
 original_cup=copy.deepcopy(cup)
 p=cup.pose.position;pp=cup.primitive_poses[0].position
 p.x=float(fit['cup_xyz'][0])-pp.x;p.y=float(fit['cup_xyz'][1])-pp.y
 cup.operation=cup.ADD;updated=True;apply_cup(cup)
 checks=checked_interpolation(states[0],targets)
 report['full_sim_pregrasp_command_path_passed']=check(checks,'current_cup_simulated_pregrasp_targets')
 if report['full_sim_pregrasp_command_path_passed']:
  report['full_sim_pregrasp_physical_states_passed']=check(states,'current_cup_simulated_physical_states')
 if report.get('full_sim_pregrasp_physical_states_passed'):
  prior=json.loads((ROOT/'fixed_trial/attempt_02/task/blocks.json').read_text())[-1]
  report['previous_rejected_block_passed_at_current_cup']=check(checked_interpolation(prior['actual_before'],prior['protected_targets']),'prior_rejected_real_ACT_block')
 report['success']=all(report.get(k) is True for k in ['full_sim_pregrasp_command_path_passed','full_sim_pregrasp_physical_states_passed','previous_rejected_block_passed_at_current_cup'])
except BaseException as exc:
 report['error']=f'{type(exc).__name__}: {exc}'
finally:
 if updated:
  try:
   apply_cup(original_cup)
   current=get_scene();restored=next(o for o in current.world.collision_objects if o.id=='act_task_cup')
   def center(o):return np.array([o.pose.position.x+o.primitive_poses[0].position.x,o.pose.position.y+o.primitive_poses[0].position.y,o.pose.position.z+o.primitive_poses[0].position.z])
   report['original_scene_restored']=bool(np.allclose(center(restored),center(original_cup),atol=1e-9) and current.allowed_collision_matrix==original.allowed_collision_matrix)
  except Exception as exc:report['restore_error']=str(exc)
 if not report['original_scene_restored']:report['success']=False
 report.update(can_tx_before=before,can_tx_after=tx())
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 node.destroy_node();rclpy.shutdown()
