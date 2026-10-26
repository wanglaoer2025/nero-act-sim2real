"""Sealed frame-only mechanism audit; truth scores outputs, never selects them."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from training.legacy_visual_data import training_profile
from rgbd_observer_v2 import RgbdCupEstimator


def main():
    source=ROOT/'rgbd_frame_audit/manifest.json';manifest=json.loads(source.read_text())
    paths=[source,Path(__file__),ROOT/'rgbd_observer_v2.py',ROOT/'rgbd_geometry.py',
        ROOT.parent/'nero_act_contact_escape_fix_20261023/card_edges.py',
        ROOT.parent/'nero_act_grouped_gripper_20261021/training/legacy_visual_data.py']
    hashes=dict(manifest['source_sha256'])
    hashes.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    records=[]
    for row in manifest['records']:
        for key,hashkey in [('image','image_sha256'),('depth_path','depth_sha256')]:
            path=Path(row[key]);expected=row[hashkey]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Exposure changed')
            hashes[str(path)]=expected
        rgb=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB)
        with np.load(row['depth_path']) as f:original_depth=f['depth_m']
        for mode in ['ideal_rendered','declared_noise_stress']:
            depth=original_depth.copy()
            if mode=='declared_noise_stress':
                rng=np.random.default_rng(1023+row['case']*10000+row['query'])
                depth+=rng.normal(0,.00035,depth.shape)
                depth[rng.random(depth.shape)<.15]=np.nan
            before_rgb=rgb.copy();before_depth=depth.copy()
            record=dict(case=row['case'],query=row['query'],role=row['role'],mode=mode,valid=False)
            started=time.monotonic()
            try:
                cup,target,diagnostic=RgbdCupEstimator(training_profile(row['scene_config']),rgb.shape[:2]).measure_depth(
                    rgb,depth,np.zeros(depth.shape))
                record.update(valid=True,error_mm=float(np.linalg.norm(cup-row['true_cup_bottom_for_diagnostic_m'])*1000),
                    position_m=cup.tolist(),target_m=target.tolist(),diagnostic=diagnostic)
            except ValueError as exc:record['error']=str(exc)
            record['elapsed_ms']=(time.monotonic()-started)*1000
            if not np.array_equal(before_rgb,rgb) or not np.array_equal(before_depth,depth,equal_nan=True):
                raise ValueError('Observer mutated exposure')
            records.append(record)
    splits={}
    for mode in ['ideal_rendered','declared_noise_stress']:
        for role in ['fit_development','observer_frame_validation']:
            rows=[r for r in records if r['mode']==mode and r['role']==role]
            valid=[r for r in rows if r['valid']];errors=[x['error_mm'] for x in valid]
            splits[mode+'/'+role]=dict(total=len(rows),valid=len(valid),valid_fraction=len(valid)/len(rows),
                within_2mm=sum(x<=2 for x in errors),maximum_error_mm=max(errors) if errors else None,
                median_error_mm=float(np.median(errors)) if errors else None,
                latency_p95_ms=float(np.percentile([r['elapsed_ms'] for r in rows],95)),
                mechanism_passed=len(valid)/len(rows)>=.9 and bool(errors) and max(errors)<=2)
    for p,h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Audit source changed')
    report=dict(complete=True,records=records,splits=splits,source_sha256=hashes,
        mechanism_passed=all(x['mechanism_passed'] for x in splits.values()),
        synthetic_noise=dict(camera_Z_std_m=.00035,dropout_fraction=.15,matched_real_sensor_distribution=False),
        ground_truth_only_used_for_error_audit=True,truth_used_as_observer_input=False,
        frame_validation_is_policy_development=True,frozen_policy_replaced=False,
        absolute_real_accuracy_verified=False,real_time_latency_qualified=False,
        training_updates=0,full_task_passed=False,hardware_deployment_allowed=False)
    output=ROOT/'rgbd_observer_results_v2.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(splits=splits,mechanism_passed=report['mechanism_passed'])),flush=True)


if __name__=='__main__':main()
