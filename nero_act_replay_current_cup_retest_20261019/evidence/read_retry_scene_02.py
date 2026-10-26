"""Read-only retest inventory. No action clients, motion services or SDK sends."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from executor.camera import SceneCamera
from executor.runtime import SceneObserver,json_write
from executor.scene_acceptance import execution_scene
from executor.execution_guards import require_current_scene

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'evidence/retry_readonly_02';out.mkdir(exist_ok=False)
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
before=tx();report=dict(mode='receive_only_retest_preflight',motion_sent=False,ready=False)
camera=health=node=None
try:
    for name,expected in json.loads((ROOT/'package_sha256.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Sealed source changed: '+name)
    manifest=json.loads((ROOT/'replay/manifest.json').read_text())
    scene=execution_scene(ROOT,json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text()),manifest,require_evidence=True)
    sensor=SceneObserver(scene,out);camera=SceneCamera(out/'camera')
    sensor.measure(camera.capture())
    observations=[]
    for _ in range(8):
        details=camera.capture();row=sensor.measure(details)
        row['capture_wall_s']=time.time()-(time.monotonic()-details['camera_received_monotonic_s'])
        observations.append(row);time.sleep(.25)
    report['observations']=observations
    camera.close();camera=None
    sys.path.insert(0,'/home/user/wq/nero_full_task_20261016')
    from task_replay import observer
    health=observer();measured=health.read_health(require_enabled=True)
    report['sdk_health']=measured
    import rclpy
    from sensor_msgs.msg import JointState
    rclpy.init();node=rclpy.create_node('act_retest_receive_only_inventory')
    names=[f'joint{i}' for i in range(1,8)]+['gripper'];latest={}
    def receive(topic,msg):
        index={n:i for i,n in enumerate(msg.name)}
        if all(n in index for n in names):latest[topic]=dict(position=[msg.position[index[n]] for n in names],receipt_monotonic_s=time.monotonic())
    subs=[node.create_subscription(JointState,topic,lambda msg,t=topic:receive(t,msg),10)
          for topic in ['/feedback/joint_states','/control/joint_states']]
    deadline=time.monotonic()+5
    while len(latest)<2 and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.01)
    report['ros_feedback']=latest;report['gated_command_publishers']=node.count_publishers('/control/gated_joint_states')
    actual=np.asarray(health.read_state()['state']);report['actual_state']=actual
    report['joint_degrees']=np.rad2deg(actual[:7]);report['gripper_width_mm']=actual[7]*1000
    expected=np.r_[scene['initial_robot_state']['qpos_rad'],scene['initial_robot_state']['gripper_total_width_m']]
    report['expected_start_joint_degrees']=np.rad2deg(expected[:7])
    report['maximum_start_joint_error_rad']=float(abs(actual[:7]-expected[:7]).max())
    report['source_offset_mm']=float(np.linalg.norm(observations[-1]['cup_xyz'][:2]-scene['pick_xy_m'])*1000)
    geometry=dict(capture_wall_s=observations[-1]['capture_wall_s'],geometry_fit_passed=True,
                  table_z_base_mm=scene['table_top_z_m']*1000,actual_state=actual,
                  targets=dict(source_bottom_mm=observations[-1]['cup_xyz']*1000,
                               destination_mm=observations[-1]['target_xyz']*1000),observations=observations)
    report['geometry']=geometry
    if len(latest)!=2 or report['gated_command_publishers']!=0:raise RuntimeError('Read-only feedback/routes are not ready')
    report['virtual_state_needs_isolated_sync']=bool(np.max(abs(np.asarray(latest['/control/joint_states']['position'])-actual))>.001)
    if np.max(abs(np.asarray(latest['/feedback/joint_states']['position'])-actual))>.001:raise RuntimeError('SDK/physical ROS feedback mismatch')
    cups=np.array([r['cup_xyz'] for r in observations]);cards=np.array([r['target_xyz'] for r in observations])
    if not np.isfinite(np.r_[cups.ravel(),cards.ravel()]).all() or max(np.ptp(cups,axis=0).max(),np.ptp(cards,axis=0).max())>.005:raise RuntimeError('Cup/card measurement window unstable')
    require_current_scene(scene,actual,geometry,now=time.time());report['ready']=True
except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}'
finally:
    if camera is not None:camera.close()
    if health is not None:health.disconnect()
    if node is not None:node.destroy_node();rclpy.try_shutdown()
    report.update(can_tx_before=before,can_tx_after=tx());json_write(out/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ['ready','error','joint_degrees','expected_start_joint_degrees','gripper_width_mm','maximum_start_joint_error_rad','source_offset_mm','can_tx_before','can_tx_after']},default=lambda a:a.tolist()),flush=True)
