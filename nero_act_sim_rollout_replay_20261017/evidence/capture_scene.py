"""New live images and SDK feedback only; all SDK sends blocked, no control ROS clients."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from executor.camera import SceneCamera
from executor.runtime import SceneObserver,json_write
from executor.execution_guards import require_current_scene
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,'/home/user/wq/nero_full_task_20261016')
from task_replay import observer
scene=json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text())
camera=health=None;tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());before=tx()
report=dict(mode='receive_only_current_scene_and_start',motion_sent=False,enable_sent=False)
try:
    health=observer();camera=SceneCamera(args.output/'camera');sensor=SceneObserver(scene,args.output);rows=[]
    for _ in range(3):
        details=camera.capture();m=sensor.measure(details);state=health.read_health(require_enabled=True)['state']
        geometry=dict(capture_wall_s=time.time()-(time.monotonic()-m['camera_monotonic_s']),
            geometry_fit_passed=bool(m['fit_rmse_px']<=3),table_z_base_mm=scene['table_top_z_m']*1000,
            targets=dict(source_bottom_mm=np.asarray(m['cup_xyz'])*1000,destination_mm=np.asarray(m['target_xyz'])*1000),measurement=m)
        require_current_scene(scene,state,geometry,now=time.time());rows.append(geometry)
    report.update(success=True,actual=state,observations=rows,geometry_assumptions_not_independent_metrology=True)
    json_write(args.output/'geometry.json',rows[-1])
except BaseException as exc:report.update(success=False,error=f'{type(exc).__name__}: {exc}')
finally:
    if camera is not None:camera.close()
    if health is not None:health.disconnect()
    report.update(can_tx_before=before,can_tx_after=tx())
    if tx()!=before:report.update(success=False,unexpected_can_tx=True)
    json_write(args.output/'report.json',report);print(json.dumps(dict(success=report.get('success'),error=report.get('error'),output=str(args.output))),flush=True)
if not report.get('success'):raise SystemExit(1)
