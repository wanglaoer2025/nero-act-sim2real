"""Receive-only live alignment window; accepts only consecutive stable valid frames."""
import json,sys,time
from pathlib import Path
import numpy as np
from executor.camera import SceneCamera
from executor.runtime import SceneObserver,json_write
from executor.execution_guards import require_current_scene
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'evidence/live_alignment';out.mkdir(exist_ok=False)
sys.path.insert(0,'/home/user/wq/nero_full_task_20261016')
from task_replay import observer
scene=json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text());tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());before=tx()
camera=health=None;report=dict(success=False,mode='receive_only_alignment',hardware_motion_sent=False);rows=[];window=[]
try:
    health=observer();camera=SceneCamera(out/'camera');sensor=SceneObserver(scene,out);deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        frame=camera.capture();m=sensor.measure(frame);actual=health.read_health(require_enabled=True)['state']
        g=dict(capture_wall_s=time.time()-(time.monotonic()-m['camera_monotonic_s']),geometry_fit_passed=bool(m['fit_rmse_px']<=3),table_z_base_mm=scene['table_top_z_m']*1000,targets=dict(source_bottom_mm=np.asarray(m['cup_xyz'])*1000,destination_mm=np.asarray(m['target_xyz'])*1000),measurement=m)
        rows.append(g);error=float(np.linalg.norm(m['cup_xyz'][:2]-scene['pick_xy_m'])*1000)
        print('SOURCE_ERROR_MM',round(error,2),flush=True)
        try:require_current_scene(scene,actual,g,now=time.time())
        except ValueError:window=[]
        else:window.append(g)
        if len(window)>=6:
            recent=window[-6:];points=np.array([x['measurement']['cup_xyz'] for x in recent]);cards=np.array([x['measurement']['target_xyz'] for x in recent])
            if np.max(np.ptp(points,axis=0))<=.005 and np.max(np.ptp(cards,axis=0))<=.005:
                json_write(out/'geometry.json',g);report.update(success=True,actual=actual);break
        if tx()!=before:raise RuntimeError('CAN changed during receive-only alignment')
        time.sleep(.5)
except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}'
finally:
    if camera is not None:camera.close()
    if health is not None:health.disconnect()
    report.update(can_tx_before=before,can_tx_after=tx(),observations=rows)
    if tx()!=before:report.update(success=False,unexpected_can_tx=True)
    json_write(out/'report.json',report);print('ALIGNMENT_SUCCESS',report['success'],flush=True)
if not report['success']:raise SystemExit(1)
