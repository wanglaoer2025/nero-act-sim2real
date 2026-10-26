"""Read-only MoveIt contact queries for the rejected ACT block; no goals or scene writes."""
import copy,json,time
from pathlib import Path
import sys
import rclpy
from moveit_msgs.srv import GetPlanningScene,GetStateValidity
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from executor.ros_execution import NAMES,checked_interpolation,RobotGeometry
from executor.fixed_permission import MODEL

output=ROOT/'evidence/current_trial_no_close_collision_diagnosis.json'
before=int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
blocked=json.loads((ROOT/'fixed_trial/attempt_01/task/blocks.json').read_text())[-1]
rows=checked_interpolation(blocked['actual_before'],blocked['protected_targets'])
rclpy.init();node=rclpy.create_node('act_rejected_path_receive_only')
def wait(f):
    end=time.monotonic()+5
    while not f.done():
        if time.monotonic()>end:raise TimeoutError('read-only scene query timeout')
        rclpy.spin_once(node,timeout_sec=.005)
    if f.exception():raise f.exception()
    return f.result()
report=dict(motion_sent=False,scene_modified=False,contacts=[])
try:
    scene_client=node.create_client(GetPlanningScene,'/get_planning_scene')
    validity=node.create_client(GetStateValidity,'/check_state_validity')
    assert scene_client.wait_for_service(timeout_sec=3) and validity.wait_for_service(timeout_sec=3)
    req=GetPlanningScene.Request();req.components.components=1023
    scene=wait(scene_client.call_async(req)).scene
    geometry=RobotGeometry(json.loads((MODEL/'scene_config.json').read_text()))
    report['world_objects']=[o.id for o in scene.world.collision_objects]
    for index,row in enumerate([blocked['actual_before']]+rows.tolist()):
        req=GetStateValidity.Request();req.group_name=''
        req.robot_state=copy.deepcopy(scene.robot_state);req.robot_state.is_diff=False
        req.robot_state.joint_state.name=NAMES
        req.robot_state.joint_state.position=list(row)
        req.robot_state.joint_state.velocity=[];req.robot_state.joint_state.effort=[]
        result=wait(validity.call_async(req))
        record=dict(index=index,valid=result.valid,state=row,tcp_xyz_m=geometry.pose(row)['tcp'].tolist(),
          contacts=[dict(body1=c.contact_body_1,body2=c.contact_body_2,depth_m=c.depth,
                         position=[c.position.x,c.position.y,c.position.z]) for c in result.contacts])
        report['contacts'].append(record)
        if not result.valid:break
    report.update(can_tx_before=before,can_tx_after=int(Path('/sys/class/net/can0/statistics/tx_packets').read_text()))
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:
    node.destroy_node();rclpy.shutdown()
