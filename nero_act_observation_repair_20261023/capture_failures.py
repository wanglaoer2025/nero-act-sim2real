"""Reconstruct bound saved exposures; ground truth is for estimator auditing only."""
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
from training.legacy_visual_data import training_profile
sys.path.insert(0,str(BASE))
from card_edges import CardEdgesEstimator


def main():
    output=ROOT/'frame_audit';output.mkdir(exist_ok=False)
    cases=[(9005,'fit_development'),(9000,'fit_development'),(9006,'observer_frame_validation'),
           (9200,'observer_frame_validation'),(9300,'observer_frame_validation')]
    records=[];sources={}
    for case_id,role in cases:
        folder=BASE/'results'/str(case_id)
        paths=[folder/'scene_configuration.json',folder/'physical_snapshots.json',folder/'rollout.npz']
        sources.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        scene=json.loads(paths[0].read_text());snapshots=json.loads(paths[1].read_text())['snapshots']
        model=build_scene(scene);data=mujoco.MjData(model);servo=NeroServo(model,data,scene)
        cup_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cup')
        with np.load(paths[2]) as trace:
            query_steps=trace['query_steps'];qstates=trace['feedback_at_query']
        result=json.loads((folder/'result.json').read_text())
        event=next(x for x in result['simulated_events'] if x['kind']=='GRASP')['physics_step']
        indices=sorted({int(np.clip(np.searchsorted(query_steps,event+offset),0,len(snapshots)-1))
            for offset in [-100,0,80,150,220,400,900]}|{len(snapshots)-1})
        estimator=CardEdgesEstimator(training_profile(scene),(180,320))
        with mujoco.Renderer(model,height=180,width=320) as renderer:
            for index in indices:
                restore(model,data,servo,snapshots[index])
                feedback=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
                if not np.array_equal(feedback,qstates[index]):raise ValueError('Saved exposure/feedback differ')
                renderer.update_scene(data,camera='calibrated_nero_camera')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                rgb=renderer.render().copy();truth=data.xpos[cup_id]-[0,0,.0475]
                path=output/f'{case_id}_{index:04d}.png'
                if not cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise RuntimeError('Frame save failed')
                row=dict(case=case_id,query=index,role=role,image=str(path.resolve()),
                    scene_config=scene,true_cup_bottom_for_diagnostic_m=truth.tolist(),
                    truth_used_as_observer_input=False,policy_acceptance=False)
                try:
                    cup,card,diagnostic=estimator.measure(rgb)
                    row.update(legacy_estimate_m=cup.tolist(),legacy_error_mm=float(np.linalg.norm(cup-truth)*1000),legacy_diagnostics=diagnostic)
                except ValueError as exc:row['legacy_error']=str(exc)
                records.append(row)
    manifest=dict(complete=True,records=records,source_sha256=sources,
        method='exact pre-action saved exposure; no motion replay or synthetic image editing',
        source_scenes_are_policy_development_only=True,real_hardware_qualified=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(frames=len(records),legacy_maximum_error_mm=max(x.get('legacy_error_mm',0) for x in records))))


if __name__=='__main__':main()
