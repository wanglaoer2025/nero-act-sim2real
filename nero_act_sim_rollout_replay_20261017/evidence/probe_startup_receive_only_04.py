"""Real camera+SDK startup with separate-process isolated ROS fixture; no goals."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
assert os.environ.get('ROS_DOMAIN_ID')=='43'
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'evidence/startup_receive_only_04';out.mkdir(exist_ok=False)
sys.path.insert(0,'/home/user/wq/nero_full_task_20261016')
from task_replay import observer
from executor.replay_session import replay_session_class
from executor.contact import ContactMonitor
from executor.camera import SceneCamera
from executor.runtime import SceneObserver,wait_background,json_write
from executor.scene_acceptance import execution_scene
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());before=tx()
scene=execution_scene(ROOT,json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text()),json.loads((ROOT/'replay/manifest.json').read_text()),require_evidence=True)
report=dict(success=False,real_camera=True,SDK_receive_only=True,ROS_domain=43,ROS_feedback_is_synthetic=True,gate_driver_service_is_fake=True,fixture_is_separate_process=True,motion_goals_sent=False,events=[])
health=session=pool=camera=fixture=log=None
try:
    pool=ThreadPoolExecutor(max_workers=1)
    report['events'].append(dict(stage='perception_warm_start_before_SDK_ROS',stamp=time.monotonic()))
    sensor=SceneObserver(scene,out);camera=pool.submit(SceneCamera,out/'camera').result(timeout=45)
    sensor.measure(pool.submit(camera.capture).result(timeout=10))
    report['events'].append(dict(stage='cold_inputs_ready_before_SDK_ROS',stamp=time.monotonic()))
    health=observer();initial=health.read_health(require_enabled=True)
    state=out/'fixture_state.json';json_write(state,dict(q=initial['state'],force=initial['gripper_details']['force_n']))
    log=(out/'fixture.log').open('x');fixture=subprocess.Popen(['/usr/bin/python3','-u',str(ROOT/'evidence/startup_ros_fixture.py'),str(state)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    session=replay_session_class()(out,report,health,scene,ContactMonitor());session.spin(1.)
    frame=wait_background(session,pool.submit(camera.capture));measurement=wait_background(session,pool.submit(sensor.measure,frame))
    session.spin(.1);session.enable_monitoring();session.spin(1.)
    assert fixture.poll() is None and time.monotonic()-session.gate_health['received']<.15
    assert session.gate_health['driver_open'] is False and not session.gate_health['failed']
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
    if fixture is not None and fixture.poll() is None:
        fixture.send_signal(signal.SIGINT)
        try:fixture.wait(timeout=5)
        except subprocess.TimeoutExpired:fixture.terminate();fixture.wait(timeout=5)
    if log is not None:log.close()
    if session is not None:session.node.destroy_node();session.ros.try_shutdown()
    if health is not None:health.disconnect()
    report.update(can_tx_before=before,can_tx_after=tx());json_write(out/'report.json',report);print(json.dumps(report),flush=True)
if not report['success']:raise SystemExit(1)
