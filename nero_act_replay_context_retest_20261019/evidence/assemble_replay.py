"""Package the complete successful ACT trace unchanged, with phase/source checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from executor.bundle import verify_model_bundle
from executor.replay_trace import validate_reference, verify_replay
from executor.ros_execution import RobotGeometry
from executor.scene_acceptance import scene_hash

ROOT = Path(__file__).resolve().parents[1]
metadata = verify_model_bundle(ROOT/'model/current_run27')
source = ROOT/'evidence/bound_scene_generation/current_cup_nominal'
result = json.loads((source/'result.json').read_text())
assert result['success'] and result['physical_success'] and result['teacher_free']
assert result['error'] is None and not result['constraint_attachment_used']
trace = np.load(source/'rollout.npz', allow_pickle=False)
rows = validate_reference(trace['sent_targets'])
events = {kind: [e['physics_step'] for e in result['simulated_events'] if e['kind'] == kind]
          for kind in ['GRASP', 'CARRY', 'RELEASE', 'RETREAT']}
assert all(len(points) == 1 for points in events.values())
grasp, carry, release, retreat = [events[kind][0] for kind in events]
scene = json.loads((ROOT/'model/current_run27/scene_config.json').read_text())
geometry = RobotGeometry(scene)
height = geometry.pose(rows[grasp])['tcp'][2]
lift = next(i for i in range(grasp+1, carry)
            if trace['target_sources'][i] == 'ACT' and geometry.pose(rows[i])['tcp'][2] > height+.0001)
boundaries = [0, grasp, lift, carry, release, retreat, len(rows)]
assert all(a < b for a, b in zip(boundaries, boundaries[1:]))
output = ROOT/'replay'
output.mkdir(exist_ok=False)
shutil.copy2(source/'rollout.npz', output/'act_rollout.npz')
shutil.copy2(source/'result.json', output/'simulation_result.json')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
original = ROOT.parent/'nero_act_replay_current_cup_retest_20261019/replay/manifest.json'
manifest = json.loads(original.read_text())
manifest.update({k: metadata[k] for k in ['checkpoint_sha256', 'stats_sha256', 'scene_sha256', 'training_config_sha256']})
manifest.update(source_rollout=str(source/'rollout.npz'), source_rollout_sha256=sha(output/'act_rollout.npz'),
    simulation_result_sha256=sha(output/'simulation_result.json'), reference_points=len(rows),
    reference_sha256=hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
    sources=dict(Counter(trace['target_sources'].tolist())),
    phases=[dict(name=name, start=start, end=end) for name, start, end in zip(
        ['approach', 'close', 'lift', 'carry_place', 'release', 'retreat'], boundaries, boundaries[1:])],
    derivation='Complete frozen context-aware run27 ACT simulation at measured current cup; trained checkpoint/statistics/config unchanged. No IK, teacher fallback or spatial rewriting.')
(output/'manifest.json').write_text(json.dumps(manifest, indent=2))
acceptance = dict(user_statement='有一点偏差也没关系',
    current_user_choice='保持当前杯位，先修复并验证ACT完整流程',
    source_xy_m=scene['pick_xy_m'], maximum_accepted_source_offset_m=.01,
    trace_sha256=manifest['source_rollout_sha256'], authorized_scene_sha256=scene_hash(scene),
    note='Zero offset from the newly bound current-cup deployment scene; original10mm scope unchanged')
(ROOT/'evidence/source_offset_acceptance.json').write_text(json.dumps(acceptance, ensure_ascii=False, indent=2))
verified, replay = verify_replay(ROOT, metadata)
assert np.array_equal(replay, rows)
print(json.dumps(dict(reference_points=len(rows), duration_s=len(rows)*.05,
    trace_sha256=manifest['source_rollout_sha256'], phases=manifest['phases'])), flush=True)
