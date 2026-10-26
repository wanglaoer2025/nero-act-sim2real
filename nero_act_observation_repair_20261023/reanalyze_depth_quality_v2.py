"""Recompute the revised motion check on retained camera arrays, without hardware."""
import hashlib
import json
from pathlib import Path

import numpy as np

from depth_quality_v2 import sequence_quality

ROOT=Path(__file__).resolve().parent


def main():
    directory=ROOT/'depth_capture_neural_light_fresh20'
    paths=[Path(__file__),ROOT/'depth_quality_v2.py',directory/'report.json',
        *[directory/f'frame_{n:02}.npz' for n in [0,9,19]]]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    samples=[]
    for n in [0,9,19]:
        with np.load(directory/f'frame_{n:02}.npz') as f:
            samples.append({k:f[k].copy() for k in ['mask','xyz','confidence']})
    result=dict(complete=True,retained_frames=[0,9,19],sequence=sequence_quality(samples),source_sha256=hashes,
        all_20_raw_frames_available=False,original_20_frame_statistic_independently_recomputed=False,
        original_report_superseded_for_motion_claims=True,hardware_commands_sent=False,
        absolute_accuracy_verified=False,hardware_deployment_allowed=False)
    for p,h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Retained camera data changed')
    output=ROOT/'depth_quality_reanalysis_v2.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
