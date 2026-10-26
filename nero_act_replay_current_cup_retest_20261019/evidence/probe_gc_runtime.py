"""Offline GC comparison under replay imports. Opens no ROS nodes/CAN/camera."""
import copy,gc,json,time
from pathlib import Path
from executor.runtime import SceneObserver
from executor.ros_execution import session_class

ROOT=Path(__file__).resolve().parents[1]
sensor=SceneObserver(json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text()),ROOT/'evidence')
sensor.measure(dict(camera_frame=str(ROOT/'evidence/authorized_scene/camera/camera_00008.png'),camera_received_monotonic_s=time.monotonic()))
session_class()  # Same class imports; never instantiate a robot session.
health=json.loads((ROOT/'evidence/real_attempt_01/independent_health.json').read_text());sample=health[-1];del health
feedback=json.loads((ROOT/'evidence/real_attempt_01/actual_feedback.json').read_text());point=feedback[-1];del feedback
results=[]
for compact in [False,True]:
    gc.collect();events=[];active={};records=[];trace=[]
    def callback(phase,info):
        gen=info['generation']
        if phase=='start':active[gen]=time.perf_counter()
        elif gen in active:events.append(dict(generation=gen,duration_s=time.perf_counter()-active.pop(gen),samples=len(records)))
    gc.callbacks.append(callback);started=time.perf_counter()
    try:
        for _ in range(20000):
            records.append(json.dumps(sample,separators=(',',':')) if compact else copy.deepcopy(sample))
            trace.extend([copy.deepcopy(point),copy.deepcopy(point)])
    finally:gc.callbacks.remove(callback)
    full=[e for e in events if e['generation']==2]
    results.append(dict(compact_json_health=compact,full_gc_events=full,maximum_full_gc_pause_s=max([e['duration_s'] for e in full],default=0),elapsed_s=time.perf_counter()-started,samples=len(records),feedback_samples=len(trace)))
    del records,trace;gc.collect()
report=dict(mode='offline_health_record_GC_comparison',hardware_connected=False,motion_sent=False,
            scope='Recorded data with actual replay imports; not proof of the exact previous event cause',results=results)
(ROOT/'evidence/heartbeat_runtime_gc_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
