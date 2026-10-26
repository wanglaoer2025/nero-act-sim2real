"""Native-resolution geometry exposures, paired with the unchanged ACT RGB fixtures."""
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


def main():
    source=ROOT/'frame_audit/manifest.json';original=json.loads(source.read_text())
    output=ROOT/'native_frame_audit';output.mkdir(exist_ok=False)
    rows=[];sources=dict(original['source_sha256'])
    sources[str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
    sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
                rgb=renderer.render().copy();image=output/f'{case}_{row["query"]:04d}.png'
                if not cv2.imwrite(str(image),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise RuntimeError('Cannot save native image')
                rows.append(dict(row,image=str(image.resolve()),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                    act_rgb_image_unchanged=row['image']))
    for p,h in sources.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Bound exposure source changed')
    (output/'manifest.json').write_text(json.dumps(dict(complete=True,records=rows,source_sha256=sources,
        native_image_size_wh=[1280,720],geometry_only=True,act_rgb_modified=False,
        truth_used_as_observer_input=False,hardware_qualified=False),indent=2)+'\n')
    print(json.dumps(dict(native_frames=len(rows),act_rgb_modified=False)))


if __name__=='__main__':main()
