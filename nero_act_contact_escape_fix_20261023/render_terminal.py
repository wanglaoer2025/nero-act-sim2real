"""Reconstruct terminal exposure from a bound physics snapshot; diagnostic only."""
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from simulation.scene import build_scene
from simulation.snapshot import restore
from simulation.nero_domain_randomization import NeroServo
from training.legacy_visual_data import NominalImageEstimator,training_profile


def main():
    case=ROOT/'results_initial_review_01/9000'
    scene=json.loads((case/'scene_configuration.json').read_text())
    saved=json.loads((case/'physical_snapshots.json').read_text())['snapshots'][-1]
    model=build_scene(scene);data=mujoco.MjData(model);servo=NeroServo(model,data,scene)
    restore(model,data,servo,saved)
    with np.load(case/'rollout.npz') as trace:
        start=int(trace['query_steps'][-1]);commands=trace['sent_targets'][start:]
        mismatch=0.
        for i,target in enumerate(commands,start):
            current=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
            mismatch=max(mismatch,float(np.max(abs(current-trace['states'][i]))))
            servo.command(target[:7],target[7])
            for _ in range(25):servo.apply();mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data)
    assert mismatch<1e-9,mismatch
    with mujoco.Renderer(model,height=180,width=320) as renderer:
        renderer.update_scene(data,camera='calibrated_nero_camera')
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        rgb=renderer.render().copy()
    output=ROOT/'release_visual_diagnostic';output.mkdir(exist_ok=True)
    cv2.imwrite(str(output/'failure_frame.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    estimator=NominalImageEstimator(training_profile(scene),(180,320))
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    blue=(hsv[:,:,0]>=95)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>90)&(hsv[:,:,2]>25)
    y,x=np.nonzero(blue)
    points=np.asarray([estimator.ray_at_table([u,v])[:2] for u,v in zip(x,y)])
    low,high=points.min(0),points.max(0)
    corners=np.array([[low[0],low[1]],[low[0],high[1]],[high[0],low[1]],[high[0],high[1]]])
    distances=np.array([np.linalg.norm(points-corner,axis=1).min() for corner in corners])
    report=dict(exact_precommand_state_reconstruction_error_rad_m=mismatch,
        visible_card_extents_m=(high-low).tolist(),visible_corner_distances_m=distances.tolist(),
        estimated_card_center_m=((high+low)/2).tolist(),
        true_target_for_diagnostic_only_m=scene['place_xy_m'],truth_used_as_policy_input=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
