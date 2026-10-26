"""Read-only localization of actual ACT input frames; timestamps retain capture age."""
import argparse,json,time
from pathlib import Path
import numpy as np
from executor.runtime import SceneObserver
from executor.execution_guards import require_current_scene
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--shadow',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
scene=json.loads((root/'model/frozen_run11/scene_config.json').read_text())
records=json.loads((a.shadow/'predictions.json').read_text())
assert len(records)>0 and all(x['prefix_contract_issue'] is None for x in records)
observer=SceneObserver(scene,a.output);rows=[]
for record in records[-3:]:
 details=record['prediction']['details'];m=observer.measure(details)
 m={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()}
 geometry=dict(capture_wall_s=time.time()-(time.monotonic()-details['camera_received_monotonic_s']),geometry_fit_passed=bool(m['fit_rmse_px']<=3),table_z_base_mm=scene['table_top_z_m']*1000,targets=dict(source_bottom_mm=(np.asarray(m['cup_xyz'])*1000).tolist(),destination_mm=(np.asarray(m['target_xyz'])*1000).tolist()),measurement=m)
 require_current_scene(scene,record['actual_feedback']['state'],geometry,now=time.time())
 rows.append(geometry)
(a.output/'geometry.json').write_text(json.dumps(rows[-1],indent=2))
(a.output/'report.json').write_text(json.dumps(dict(success=True,readonly=True,hardware_permission=False,geometry_assumptions_not_metrology=True,rows=rows),indent=2))
print(json.dumps(dict(success=True,checked_frames=len(rows),geometry=str(a.output/'geometry.json'),fit_rmse_px=rows[-1]['measurement']['fit_rmse_px'])))
