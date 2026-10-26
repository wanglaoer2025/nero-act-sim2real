"""Reproduce ACT startup with send-blocked ROS driver; submits no actions/services."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys,time,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from executor import runtime
from executor.ros_execution import session_class
from executor.contact import ContactMonitor
from executor.fixed_permission import MODEL
sys.path.insert(0,'/home/user/wq/nero_full_task_20261016')
from task_replay import observer

out=ROOT/'evidence'/sys.argv[1] if len(sys.argv)>1 else ROOT/'evidence/startup_readonly_diagnostic_01'
out.mkdir(exist_ok=False)
events=[];report={};session=worker=health=pool=None
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
before=tx();context='initial'
def event(name,**values):
    row=dict(event=name,monotonic=time.monotonic(),**values);events.append(row);print(json.dumps(row),flush=True)
try:
    readonly=json.loads((ROOT/'fixed_trial/attempt_01/readonly_restored.json').read_text())
    assert readonly['runtime_send_blocked'] is True
    scene=json.loads((MODEL/'scene_config.json').read_text())
    if '--prewarm' in sys.argv:
        event('prewarm_before_ros')
        runtime.SceneObserver(scene,out)
        event('prewarm_complete')
    health=observer()
    session=session_class()(out,report,health,scene,ContactMonitor())
    actual=session.inventory()
    assert session.node.count_publishers('/control/gated_joint_states')==0
    assert report['driver']['speed_percent']==0 and report['driver']['auto_enable'] is False
    base_actual=session.actual
    def checked_actual():
        try:return base_actual()
        except Exception:
            event('actual_failed',context=context,stack=traceback.format_exc(),
                  last_receipt=session.latest.get('/feedback/joint_states',{}).get('receipt'))
            raise
    session.actual=checked_actual
    original_verify=runtime.verify_model_bundle
    def timed_verify(*args,**kwargs):
        start=time.monotonic();result=original_verify(*args,**kwargs)
        event('bundle_verify',duration_s=time.monotonic()-start);return result
    runtime.verify_model_bundle=timed_verify
    session.enable_monitoring()
    context='LiveWorker constructor';event('begin_worker')
    worker=runtime.LiveWorker(MODEL,out,session);event('worker_ready')
    context='SceneObserver constructor';event('begin_scene_observer')
    pool=ThreadPoolExecutor(max_workers=1)
    sensor=runtime.wait_background(session,pool.submit(runtime.SceneObserver,scene,out))
    event('scene_observer_ready')
    context='first current input prediction'
    worker.predict();event('prediction_ready')
    report['diagnostic_success']=True
except BaseException:
    report['error_traceback']=traceback.format_exc();event('failed',error=report['error_traceback'])
finally:
    if session is not None:session.monitoring=False
    if worker is not None:worker.close()
    if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
    if session is not None:
        runtime.json_write(out/'actual_feedback.json',session.trace)
        session.node.destroy_node();session.ros.try_shutdown()
    if health is not None:health.disconnect()
    report.update(can_tx_before=before,can_tx_after=tx(),motion_sent=False,action_goals_submitted=False)
    runtime.json_write(out/'events.json',events);runtime.json_write(out/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ['diagnostic_success','can_tx_before','can_tx_after','motion_sent']}))
