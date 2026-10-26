"""Render observable depth at the same saved exposure; no body truth in estimator input."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'nero_act_contact_escape_fix_20261023'
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from simulation.scene import build_scene
from simulation.snapshot import restore
from simulation.nero_domain_randomization import NeroServo
from rgbd_geometry import world_points_from_depth


def main():
    source=ROOT/'native_frame_audit/manifest.json';original=json.loads(source.read_text())
    output=ROOT/'rgbd_frame_audit';output.mkdir(exist_ok=False)
    sources=dict(original['source_sha256'])
    for p in [source,Path(__file__),ROOT/'rgbd_geometry.py']:
        sources[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    records=[]
    for case in sorted({r['case'] for r in original['records']}):
        selected=[r for r in original['records'] if r['case']==case]
        scene=selected[0]['scene_config'];model=build_scene(scene);data=mujoco.MjData(model)
        model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
        servo=NeroServo(model,data,scene)
        saved=json.loads((BASE/'results'/str(case)/'physical_snapshots.json').read_text())['snapshots']
        with mujoco.Renderer(model,height=720,width=1280) as renderer:
            for row in selected:
                restore(model,data,servo,saved[row['query']])
                renderer.update_scene(data,camera='calibrated_nero_camera')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                rgb=renderer.render().copy()
                reference=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB)
                if not np.array_equal(rgb,reference):raise ValueError('RGB-D exposure does not match the frozen native image')
                renderer.enable_depth_rendering();depth=renderer.render().copy();renderer.disable_depth_rendering()
                points=world_points_from_depth(depth,scene['camera_calibration'])
                hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
                blue=(hsv[:,:,0]>=95)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>90)&(hsv[:,:,2]>25)
                median_table_z=float(np.median(points[:,:,2][blue]))
                if abs(median_table_z-.001)>.002:
                    raise ValueError('Rendered depth backprojection disagrees with known card plane')
                path=output/f'{case}_{row["query"]:04d}.npz'
                np.savez_compressed(path,depth_m=depth)
                records.append(dict(row,depth_path=str(path.resolve()),
                    depth_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    card_plane_depth_check_m=median_table_z,depth_is_observable_render_not_object_pose=True))
    for p,h in sources.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('RGB-D exposure source changed')
    (output/'manifest.json').write_text(json.dumps(dict(complete=True,records=records,source_sha256=sources,
        truth_used_as_observer_input=False,ideal_rendered_depth=True,real_sensor_error_bound_verified=False,
        hardware_deployment_allowed=False),indent=2)+'\n')
    print(json.dumps(dict(rgbd_frames=len(records),RGB_bit_identical=True)))


if __name__=='__main__':main()
