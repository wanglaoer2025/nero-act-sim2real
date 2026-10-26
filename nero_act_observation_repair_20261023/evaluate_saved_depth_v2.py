"""Evaluate saved real camera exposures; never communicates with camera or robot."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from training.legacy_visual_data import training_profile
from rgbd_observer_v2 import RgbdCupEstimator


def main():
    manifest=ROOT/'rgbd_frame_audit/manifest.json'
    row=json.loads(manifest.read_text())['records'][0]
    capture=ROOT/'depth_capture_neural_light_fresh20'
    camera=json.loads((capture/'report.json').read_text())
    profile=training_profile(row['scene_config'])
    k=np.asarray(profile['camera']['intrinsic_matrix'])
    intrinsic=camera['rectified_left_intrinsics']
    if not np.allclose([k[0,0],k[1,1],k[0,2],k[1,2]],
        [intrinsic[x] for x in ['fx','fy','cx','cy']],atol=1e-8):
        raise ValueError('Saved camera intrinsics differ from the candidate observer profile')
    paths=[Path(__file__),ROOT/'rgbd_observer_v2.py',ROOT/'rgbd_geometry.py',
        ROOT.parent/'nero_act_contact_escape_fix_20261023/card_edges.py',
        ROOT.parent/'nero_act_grouped_gripper_20261021/training/legacy_visual_data.py',
        manifest,capture/'report.json',*[capture/f'frame_{n:02}.npz' for n in [0,9,19]]]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    records=[]
    for n in [0,9,19]:
        with np.load(capture/f'frame_{n:02}.npz') as f:
            rgb,xyz,confidence=[f[x].copy() for x in ['rgb','xyz','confidence']]
        before=[x.copy() for x in [rgb,xyz,confidence]]
        record=dict(frame=n,valid=False);start=time.monotonic()
        try:
            cup,target,diagnostic=RgbdCupEstimator(profile,rgb.shape[:2]).measure_camera_xyz(rgb,xyz,confidence)
            record.update(valid=True,cup_bottom_m=cup.tolist(),target_m=target.tolist(),diagnostic=diagnostic)
        except ValueError as exc:record['error']=str(exc)
        record['elapsed_ms']=(time.monotonic()-start)*1000
        if not all(np.array_equal(a,b,equal_nan=True) for a,b in zip(before,[rgb,xyz,confidence])):
            raise ValueError('Saved real exposure was modified')
        records.append(record)
    for p,h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Saved exposure audit source changed')
    result=dict(complete=True,records=records,source_sha256=hashes,
        full_cup_and_target_valid=sum(r['valid'] for r in records),
        stored_camera_only=True,hardware_commands_sent=False,absolute_accuracy_verified=False,
        full_task_passed=False,hardware_deployment_allowed=False)
    output=ROOT/'saved_real_depth_observer_results_v2.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
