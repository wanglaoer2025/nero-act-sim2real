"""Full ACT replay path in isolated MoveIt scene; no trajectory/control clients."""
import copy,json,os,time
from pathlib import Path
import numpy as np
assert os.environ.get('ROS_DOMAIN_ID')=='43'
import rclpy
from rcl_interfaces.srv import GetParameters
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import GetPlanningScene,GetStateValidity
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from rosidl_runtime_py.set_message import set_message_fields
from executor.ros_execution import CupSceneMixin,RobotGeometry,checked_interpolation,validate_in_batches
from executor.replay_trace import verify_replay
from executor.replay_session import protect_gripper,join_reference
from executor.release_hold import release_reference_transfer
from executor.bundle import verify_model_bundle
ROOT=Path(__file__).resolve().parents[1];prior=ROOT.parent/'nero_act_full_pick_place_20261017'
output=ROOT/'evidence/replay_collision_preflight';output.mkdir(exist_ok=False)
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());before=tx()
rclpy.init(args=[]);node=rclpy.create_node('replay_isolated_collision_probe')
report=dict(mode='isolated_full_robot_table_cup_collision',hardware_connected=False,trajectory_clients_created=False,checks=[])
class Session(CupSceneMixin):
    def __init__(self):
        self.output=output;self.clients={};self.held_center=self.held_rotation=None
        self.scene_config=json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text())
        self.geometry=RobotGeometry(self.scene_config)
        self.current=np.r_[self.scene_config['initial_robot_state']['qpos_rad'],self.scene_config['initial_robot_state']['gripper_total_width_m']]
        (output/'robot.urdf').write_text((prior/'evidence/current_ros_readonly_inventory/robot.urdf').read_text())
    def actual(self):return self.current
    def cached_client(self,name,kind):
        if name not in self.clients:
            client=node.create_client(kind,'/act_collision_preflight'+name)
            if not client.wait_for_service(timeout_sec=8):raise RuntimeError('Isolated collision service absent: '+name)
            self.clients[name]=client
        return self.clients[name]
    def wait(self,future,timeout=8):
        end=time.monotonic()+timeout
        while not future.done():
            if time.monotonic()>end:raise TimeoutError('Isolated service timeout')
            rclpy.spin_once(node,timeout_sec=.001)
        if future.exception():raise future.exception()
        return future.result()
    def scene(self):
        request=GetPlanningScene.Request();request.components.components=1023
        return self.wait(self.cached_client('/get_planning_scene',GetPlanningScene).call_async(request)).scene
    def check_rows(self,name,rows):
        state=copy.deepcopy(self.scene().robot_state);state.is_diff=False
        state.joint_state.name=[f'joint{i}' for i in range(1,8)]+['gripper'];state.joint_state.velocity=[];state.joint_state.effort=[]
        checks=checked_interpolation(self.current,rows);bad=[]
        def submit(row):
            request=GetStateValidity.Request();request.group_name='';request.robot_state=copy.deepcopy(state)
            request.robot_state.joint_state.position=row.tolist()
            return self.cached_client('/check_state_validity',GetStateValidity).call_async(request)
        def wait(future):
            result=self.wait(future)
            if not result.valid:bad.extend([dict(first=c.contact_body_1,second=c.contact_body_2,depth=c.depth) for c in result.contacts])
            return result
        try:count=validate_in_batches(checks,submit,wait)
        except Exception:
            report['checks'].append(dict(name=name,success=False,contacts=bad));raise
        report['checks'].append(dict(name=name,success=True,intermediate_checks=count));print('COLLISION_PASS '+name+' '+str(count),flush=True)
try:
    names=[n for n,_ in node.get_node_names_and_namespaces()]
    if any(n in names for n in ['agx_arm_ctrl_single_node','control_gate']):raise RuntimeError('Unexpected driver/gate in isolated collision domain')
    session=Session();params=node.create_client(GetParameters,'/act_collision_preflight/move_group/get_parameters')
    if not params.wait_for_service(timeout_sec=8):raise RuntimeError('Isolated move_group unavailable')
    value=session.wait(params.call_async(GetParameters.Request(names=['allow_trajectory_execution']))).values[0]
    if value.type!=1 or value.bool_value:raise RuntimeError('Trajectory execution not disabled')
    original=json.loads((prior/'evidence/current_ros_readonly_inventory/planning_scene.json').read_text())
    source=next(o for o in original['world']['collision_objects'] if o['id']=='nero_table')
    table=CollisionObject(id='nero_table',operation=CollisionObject.ADD);table.header.frame_id=source['header']['frame_id']
    set_message_fields(table.pose,source['pose']);table.primitives=[SolidPrimitive(type=1,dimensions=source['primitives'][0]['dimensions'])]
    pose=Pose();set_message_fields(pose,source['primitive_poses'][0]);table.primitive_poses=[pose]
    session.apply_scene([table]);session.configure_initial_scene()
    manifest,trace=verify_replay(ROOT,verify_model_bundle(ROOT/'model/frozen_run11'))
    # Closed-cup widths are protective openings, not changes to any ACT arm target.
    for phase in manifest['phases']:
        name=phase['name'];rows=trace[phase['start']:phase['end']].copy()
        if name=='close':session.allow_finger_contact(True)
        if name=='lift':session.attach_actual_source()
        if name=='release':session.detach_for_supported_release(session.scene_config['place_xy_m'])
        if name=='retreat':session.finish_supported_release(session.scene_config['place_xy_m'])
        session.check_rows(name,rows)
        if name in ['close','lift','carry_place']:
            for width in [.074,.083,.09]:session.check_rows(name+'_protect_'+str(width),protect_gripper(rows,width))
        session.current=rows[-1].copy()
    report['success']=True
except BaseException as exc:report.update(success=False,error=f'{type(exc).__name__}: {exc}')
finally:
    node.destroy_node();rclpy.try_shutdown();report.update(can_tx_before=before,can_tx_after=tx())
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if not report['success']:raise SystemExit(1)
