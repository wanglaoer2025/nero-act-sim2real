"""Audit actual action timing conventions across all 68 immutable demonstrations."""
from pathlib import Path
import sys,json
from collections import Counter
import numpy as np
import h5py

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'nero_act_optimizer_fixed_20261021';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from training.integrity import snapshot,require_unchanged

manifest=ROOT/'training_data/filtered_v1/manifest.json';rows=json.loads(manifest.read_text())['episodes']
records=[];sources=[Path(__file__),manifest,ROOT/'simulation/history_teacher.py',ROOT/'simulation/teacher.py',ROOT/'executor/model.py']
for row in rows:
    path=Path(row['path']);sources.append(path)
    with h5py.File(path,'r') as f:
        scene=json.loads(f.attrs['scene_config']);stages=f['simulation/stage'].asstr()[:];targets=f['action'][:,7]
        pairs=np.flatnonzero((stages[:-1]=='close')&(stages[1:]=='close'))
        diff=(targets[pairs+1]-targets[pairs])*1000;changing=diff[diff<-.05]
        records.append(dict(episode=row['id'],split=row['split'],role=str(f.attrs.get('recovery_role','base')),
            control_hz=scene['control_hz'],gripper_speed_m_s=scene.get('gripper_speed_limit_m_s',.01),
            configured_step_mm=scene.get('gripper_speed_limit_m_s',.01)/scene['control_hz']*1000,
            observed_median_closing_step_mm=None if not len(changing) else float(np.median(changing)),
            active_close_pairs=len(changing),joint_speed_rad_s=scene['joint_speed_limit_rad_s']))
counts=Counter((r['role']=='base',r['gripper_speed_m_s'],r['control_hz']) for r in records)
assert len(records)==68 and counts[(True,.008,20)]==48 and counts[(False,.004,20)]==20
proof=dict(complete=True,actual_optimizer_updates=0,records=records,
    groups=[dict(base=base,speed_m_s=speed,control_hz=hz,episodes=n) for (base,speed,hz),n in counts.items()],
    policy_input_has_explicit_speed_or_time_scale=False,source_sha256=snapshot(sources),hardware_commands_sent=False)
require_unchanged(proof['source_sha256']);(OUT/'timebase_audit.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(complete=True,groups=proof['groups'],actual_optimizer_updates=0)))
