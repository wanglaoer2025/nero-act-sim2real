"""Real camera/SDK warmup; isolated synthetic ROS feedback/gate, no motion goals."""
import json,os,sys,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
assert os.environ.get('ROS_DOMAIN_ID')=='43'
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'evidence/startup_receive_only_02';out.mkdir(exist_ok=False)
sys.path.insert(0,'/home/user/wq/nero_full_task_20261016')
from task_replay import observer
from executor.replay_session import replay_session_class
from executor.gate import gate_class
from executor.contact import ContactMonitor
from executor.camera import SceneCamera
from executor.runtime import SceneObserver,wait_background,json_write
from executor.scene_acceptance import execution_scene
sys.argv=['probe','--ros-args','-p','arm_joint_names:=[joint1,joint2,joint3,joint4,joint5,joint6,joint7]','-p','status_topics:=[/arm_controller/follow_joint_trajectory/_action/status,/gripper_controller/follow_joint_trajectory/_action/status]','-p','arm_sync_tolerance_rad:=0.001','-p','gripper_sync_tolerance_m:=0.0005','-p','state_timeout_s:=0.15']
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());before=tx()
scene=execution_scene(ROOT,json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text()),json.loads((ROOT/'replay/manifest.json').read_text()),require_evidence=True)
report=dict(success=False,real_camera=True,SDK_receive_only=True,ROS_domain=43,ROS_feedback_is_synthetic=True,gate_driver_service_is_fake=True,motion_goals_sent=False,events=[])
health=session=gate=fake=executor=pool=camera=None;thread=None
try:
    health=observer();initial=health.read_health(require_enabled=True)
    session=replay_session_class()(out,report,health,scene,ContactMonitor())
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_srvs.srv import SetBool
    fake=rclpy.create_node('synthetic_feedback_no_CAN');pubs=[fake.create_publisher(JointState,t,10) for t in ['/feedback/joint_states','/control/joint_states']]
    def publish():
        msg=JointState();msg.header.stamp=fake.get_clock().now().to_msg();msg.name=[f'joint{i}' for i in range(1,8)]+['gripper']
        msg.position=list(map(float,initial['state']));msg.effort=[0.]*7+[float(initial['gripper_details']['force_n'])]
        for p in pubs:p.publish(msg)
    def set_gate(request,response):
        if request.data:raise RuntimeError('No enable is allowed in startup-only test')
        response.success=True;return response
    service=fake.create_service(SetBool,'/control_enable',set_gate);timer=fake.create_timer(.005,publish)
    gate=gate_class()(ttl_s=90);executor=MultiThreadedExecutor(num_threads=2);executor.add_node(fake);executor.add_node(gate)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start();session.spin(.5)
    pool=ThreadPoolExecutor(max_workers=1)
    report['events'].append(dict(stage='camera_warm_start',monitoring=session.monitoring,stamp=time.monotonic()))
    camera=wait_background(session,pool.submit(SceneCamera,out/'camera'))
    report['events'].append(dict(stage='perception_warm_start',monitoring=session.monitoring,stamp=time.monotonic()))
    sensor=wait_background(session,pool.submit(SceneObserver,scene,out))
    frame=wait_background(session,pool.submit(camera.capture));measurement=wait_background(session,pool.submit(sensor.measure,frame))
    assert not session.monitoring
    report['events'].append(dict(stage='cold_inputs_ready',monitoring=session.monitoring,stamp=time.monotonic()))
    session.enable_monitoring();session.spin(1.)
    assert session.monitoring and not gate.ever_opened and not gate.latch.open and not gate.latch.failed
    assert time.monotonic()-session.gate_health['received']<.15
    assert tx()==before
    report.update(success=True,measured_source_xy_m=measurement['cup_xyz'][:2].tolist(),healthy_samples=len(session.health_trace),gate_opened=False)
except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}'
finally:
    if session is not None:
        session.monitoring=False
        try:session.cancel_all()
        except Exception as exc:report['cleanup_error']=str(exc)
    if camera is not None:camera.close()
    if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
    if executor is not None:executor.shutdown()
    if thread is not None:thread.join(timeout=2)
    if gate is not None:gate.destroy_node()
    if fake is not None:fake.destroy_node()
    if session is not None:session.node.destroy_node();session.ros.try_shutdown()
    if health is not None:health.disconnect()
    report.update(can_tx_before=before,can_tx_after=tx());json_write(out/'report.json',report);print(json.dumps(report),flush=True)
if not report['success']:raise SystemExit(1)
