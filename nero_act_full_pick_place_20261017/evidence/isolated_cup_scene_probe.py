"""Use real MoveIt messages/services in isolated domain43; no robot or action clients."""
import copy
import argparse
import json
import os
from pathlib import Path
import sys
import time

if os.environ.get('ROS_DOMAIN_ID') != '43':raise RuntimeError('Isolated domain43 required')
import h5py
import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters
from moveit_msgs.msg import CollisionObject
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetPlanningScene, GetStateValidity
from rosidl_runtime_py.set_message import set_message_fields

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from executor.ros_execution import CupSceneMixin, RobotGeometry, validate_in_batches

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'evidence/isolated_cup_scene_probe')
output=parser.parse_args().output;output.mkdir(exist_ok=False)
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
before=tx();rclpy.init(args=[]);node=rclpy.create_node('act_isolated_cup_scene_probe')
report=dict(mode='isolated_planning_scene_integration_only',domain=43,
    real_hardware_connected=False,trajectory_clients_created=False,control_publishers_created=False,
    physical_task_verified=False,transitions=[])

class Session(CupSceneMixin):
    def __init__(self):
        self.output=output;self.clients={};self.held_center=self.held_rotation=None
        with h5py.File(ROOT/'training_data_current_source/episode_0.hdf5','r') as f:
            self.scene_config=json.loads(f.attrs['scene_config'])
            stages=f['simulation/stage'].asstr()[:]
            states=np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]]
        self.stage_states={name:states[np.flatnonzero(stages==name)[-1]] for name in set(stages)}
        self.current=np.r_[self.scene_config['initial_robot_state']['qpos_rad'],.095]
        self.geometry=RobotGeometry(self.scene_config)
        (output/'robot.urdf').write_text((ROOT/'evidence/current_ros_readonly_inventory/robot.urdf').read_text())
    def actual(self):return self.current
    def cached_client(self,name,kind):
        if name not in self.clients:
            client=node.create_client(kind,'/act_collision_preflight'+name)
            if not client.wait_for_service(timeout_sec=10):raise RuntimeError('Isolated service absent: '+name)
            self.clients[name]=client
        return self.clients[name]
    def wait(self,future,timeout=5):
        deadline=time.monotonic()+timeout
        while not future.done():
            if time.monotonic()>=deadline:raise TimeoutError('Isolated service timeout')
            rclpy.spin_once(node,timeout_sec=.002)
        if future.exception():raise future.exception()
        return future.result()
    def scene(self):
        request=GetPlanningScene.Request();request.components.components=1023
        return self.wait(self.cached_client('/get_planning_scene',GetPlanningScene).call_async(request)).scene
    def check(self,name,expected_valid=True):
        scene=self.scene();world=[o.id for o in scene.world.collision_objects]
        attached=[o.object.id for o in scene.robot_state.attached_collision_objects]
        if world.count('act_task_cup')+attached.count('act_task_cup')!=1:
            raise RuntimeError('Cup collision representation missing or duplicated')
        request=GetStateValidity.Request();request.group_name='';request.robot_state=copy.deepcopy(scene.robot_state)
        request.robot_state.is_diff=False
        request.robot_state.joint_state.name=[f'joint{i}' for i in range(1,8)]+['gripper']
        request.robot_state.joint_state.position=self.current.tolist()
        request.robot_state.joint_state.velocity=[];request.robot_state.joint_state.effort=[]
        result=self.wait(self.cached_client('/check_state_validity',GetStateValidity).call_async(request))
        report['transitions'].append(dict(name=name,world_cup='act_task_cup' in world,
            attached_cup='act_task_cup' in attached,valid=result.valid,
            contacts=[dict(first=c.contact_body_1,second=c.contact_body_2,depth=c.depth) for c in result.contacts]))
        if result.valid!=expected_valid:raise RuntimeError('Unexpected isolated collision result: '+name)

try:
    names=[name for name,_ in node.get_node_names_and_namespaces()]
    if any(n in names for n in ['agx_arm_ctrl_single_node','control_gate']):
        raise RuntimeError('Physical driver/gate unexpectedly present in isolated domain')
    params=node.create_client(GetParameters,'/act_collision_preflight/move_group/get_parameters')
    if not params.wait_for_service(timeout_sec=5):raise RuntimeError('Isolated parameter service absent')
    session=Session()
    value=session.wait(params.call_async(GetParameters.Request(names=['allow_trajectory_execution']))).values[0]
    if value.type!=1 or value.bool_value:raise RuntimeError('Trajectory execution must be disabled')
    report['trajectory_execution_disabled']=True
    original=json.loads((ROOT/'evidence/current_ros_readonly_inventory/planning_scene.json').read_text())
    source=next(obj for obj in original['world']['collision_objects'] if obj['id']=='nero_table')
    table=CollisionObject(id='nero_table',operation=CollisionObject.ADD)
    table.header.frame_id=source['header']['frame_id'];set_message_fields(table.pose,source['pose'])
    table.primitives=[SolidPrimitive(type=1,dimensions=source['primitives'][0]['dimensions'])]
    pose=Pose();set_message_fields(pose,source['primitive_poses'][0]);table.primitive_poses=[pose]
    session.apply_scene([table]);session.configure_initial_scene();session.check('source_high')
    session.current=session.stage_states['close'];session.allow_finger_contact(True);session.check('closing_world_cup')
    session.attach_actual_source();session.current=session.stage_states['lift'];session.check('holding_attached_cup')
    session.current=session.stage_states['place'];session.detach_for_supported_release(session.scene_config['place_xy_m'])
    session.check('supported_release_world_cup')
    session.current=session.stage_states['release'];session.finish_supported_release(session.scene_config['place_xy_m'])
    session.check('fingers_clear_all_contacts_restored')
    session.current=session.stage_states['retreat'];session.check('retreat_world_cup')
    report['success']=True
except BaseException as exc:
    report.update(success=False,error=f'{type(exc).__name__}: {exc}')
finally:
    node.destroy_node();rclpy.try_shutdown()
    report.update(can_tx_before=before,can_tx_after=tx())
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if not report['success']:raise SystemExit(1)
