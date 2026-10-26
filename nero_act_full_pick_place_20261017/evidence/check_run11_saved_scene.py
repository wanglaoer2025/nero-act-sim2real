"""Measure captured readonly inference frames; never refresh their capture timestamps."""
import json,time
from pathlib import Path
import numpy as np
from executor.runtime import SceneObserver
root=Path(__file__).resolve().parents[1];out=root/'evidence/run11_saved_scene';out.mkdir(exist_ok=False)
scene=json.loads((root/'model/candidate_run11_release_deadband/scene_config.json').read_text());records=json.loads((root/'evidence/run11_live_shadow/predictions.json').read_text());observer=SceneObserver(scene,out);rows=[]
for record in records:
 details=record['prediction']['details'];measured=observer.measure(details)
 measured={key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in measured.items()}
 measured['capture_wall_s']=time.time()-(time.monotonic()-details['camera_received_monotonic_s']);rows.append(measured)
pick=np.array([r['cup_xyz'] for r in rows]);place=np.array([r['target_xyz'] for r in rows]);report=dict(readonly=True,physical_motion_sent=False,geometry_assumptions_not_metrology=True,rows=rows,
 pick_max_error_from_training_m=float(max(np.linalg.norm(pick[:,:2]-scene['pick_xy_m'],axis=1))),place_max_error_from_training_m=float(max(np.linalg.norm(place[:,:2]-scene['place_xy_m'],axis=1))),
 cup_max_spread_m=float(max(np.linalg.norm(pick-pick.mean(axis=0),axis=1))),target_max_spread_m=float(max(np.linalg.norm(place-place.mean(axis=0),axis=1))),
 max_fit_rmse_px=max(r['fit_rmse_px'] for r in rows),measured_wall_s=time.time(),onsite_confirmation=False,hardware_permission=False)
report['saved_frame_scene_check_passed']=bool(report['pick_max_error_from_training_m']<=.005 and report['place_max_error_from_training_m']<=.005 and report['cup_max_spread_m']<.003 and report['target_max_spread_m']<.003)
(out/'report.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='rows'});assert report['saved_frame_scene_check_passed']
