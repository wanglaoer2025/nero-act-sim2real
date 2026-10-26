"""Frame-level nominal pose audit; no frozen policy or real acceptance is changed."""
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
sys.path.insert(0,str(ROOT.parent/'nero_act_contact_escape_fix_20261023'))
from card_edges import CardEdgesEstimator
from cup_observer import VisibleBoundaryCupEstimator


def main():
    source=ROOT/'native_frame_audit/manifest.json';manifest=json.loads(source.read_text())
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [source,ROOT/'cup_observer.py',Path(__file__)]}
    hashes.update(manifest['source_sha256'])
    records=[]
    for row in manifest['records']:
        path=Path(row['image'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:raise ValueError('Native exposure changed')
        hashes[str(path)]=row['image_sha256']
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);original=rgb.copy()
        profile=training_profile(row['scene_config'])
        record=dict(case=row['case'],query=row['query'],role=row['role'],valid=False,
            low_resolution_legacy_error_mm=row.get('legacy_error_mm'))
        for name,kind in [('native_legacy',CardEdgesEstimator),('candidate',VisibleBoundaryCupEstimator)]:
            started=time.monotonic()
            try:
                cup,target,diagnostic=kind(profile,rgb.shape[:2]).measure(rgb)
                error=float(np.linalg.norm(cup-row['true_cup_bottom_for_diagnostic_m'])*1000)
                record[name]=dict(valid=True,error_mm=error,position_m=cup.tolist(),diagnostic=diagnostic,
                    elapsed_ms=(time.monotonic()-started)*1000)
            except ValueError as exc:
                record[name]=dict(valid=False,error=str(exc),elapsed_ms=(time.monotonic()-started)*1000)
        if not np.array_equal(rgb,original):raise ValueError('Observer changed current RGB')
        records.append(record)
    splits={}
    for role in ['fit_development','observer_frame_validation']:
        rows=[r for r in records if r['role']==role]
        valid=[r['candidate'] for r in rows if r['candidate']['valid']]
        errors=[x['error_mm'] for x in valid]
        splits[role]=dict(total=len(rows),valid=len(valid),valid_fraction=len(valid)/len(rows),
            within_2mm=sum(x<=2 for x in errors),maximum_error_mm=max(errors) if errors else None,
            median_error_mm=float(np.median(errors)) if errors else None,
            mechanism_passed=len(valid)/len(rows)>=.9 and bool(errors) and max(errors)<=2)
    for p,h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Observer audit source changed')
    report=dict(complete=True,records=records,splits=splits,source_sha256=hashes,
        mechanism_passed=all(x['mechanism_passed'] for x in splits.values()),
        geometry_truth_policy_input=False,ground_truth_only_used_for_error_audit=True,
        frozen_policy_replaced=False,full_task_passed=False,hardware_deployment_allowed=False)
    (ROOT/'observer_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(splits=splits,mechanism_passed=report['mechanism_passed'])),flush=True)


if __name__=='__main__':main()
