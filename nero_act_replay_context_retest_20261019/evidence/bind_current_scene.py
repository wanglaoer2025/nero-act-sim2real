"""Derive deployment config; preserve trained tensors and source training proof."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import torch

from executor.bundle import verify_model_bundle
from executor.policy_worker import FrozenAct

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'model/source_run27'
DEST = ROOT/'model/current_run27'
metadata = verify_model_bundle(SOURCE)
prototype = ROOT.parent/'nero_act_replay_ready_retest_20261019/evidence/context_comparison_run27'
protocol = json.loads((prototype/'protocol.json').read_text())
summary = json.loads((prototype/'summary.json').read_text())
assert summary['success'] and summary['physical_success'] and summary['error'] is None
scene = protocol['scene']
shutil.copytree(SOURCE, DEST)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
binding = dict(mode='posttraining_deployment_scene_only_no_optimization',
               source_model=str(SOURCE.resolve()), source_scene_sha256=metadata['scene_sha256'],
               source_checkpoint_sha256=metadata['checkpoint_sha256'],
               prototype_sha256=sha(prototype/'summary.json'),
               geometry_sha256=sha(ROOT.parent/'nero_act_replay_current_cup_retest_20261019/evidence/retry_readonly_02/report.json'),
               trained_weights_statistics_and_source_training_proof_unchanged=True,
               deployment_config_derived=True,
               additional_optimization_steps=0)
(DEST/'original_deployment_scene.json').write_bytes((SOURCE/'scene_config.json').read_bytes())
(DEST/'source_training_config.json').write_bytes((SOURCE/'training_config.json').read_bytes())
(DEST/'source_training_integrity.json').write_bytes((SOURCE/'training_integrity.json').read_bytes())
payload = torch.load(SOURCE/'policy.ckpt', map_location='cpu', weights_only=False)
config = copy.deepcopy(payload['config'])
config.update(scene_config=scene, source_training_scene_config=copy.deepcopy(payload['config']['scene_config']),
              deployment_scene_binding=binding)
derived = dict(payload, config=config, additional_optimization_steps=0,
               source_checkpoint_sha256=metadata['checkpoint_sha256'], deployment_config_only=True)
torch.save(derived, DEST/'policy.ckpt')
(DEST/'training_config.json').write_text(json.dumps(config, indent=2))
(DEST/'scene_config.json').write_text(json.dumps(scene, indent=2))
updated = dict(metadata, checkpoint_sha256=sha(DEST/'policy.ckpt'),
               training_config_sha256=sha(DEST/'training_config.json'),
               scene_sha256=sha(DEST/'scene_config.json'), initial_state=scene['initial_robot_state'],
               deployment_scene_binding=binding, hardware_deployment_allowed=False,
               source_checkpoint_sha256=metadata['checkpoint_sha256'],
               training_integrity_scope='source_run27_checkpoint_only; derived deployment config is not a new training output',
               real_full_task_verified=False, simulation_evaluation_completed=False, simulation_evaluation_passed=False)
(DEST/'runtime_metadata.json').write_text(json.dumps(updated, indent=2))
for name in ['dataset_stats.pkl', 'training_integrity.json']:
    assert sha(SOURCE/name) == sha(DEST/name)
restored = torch.load(DEST/'policy.ckpt', map_location='cpu', weights_only=False)
assert restored['model'].keys() == payload['model'].keys()
assert all(torch.equal(restored['model'][k], payload['model'][k]) for k in payload['model'])
assert json.loads((DEST/'source_training_config.json').read_text()) == payload['config']
verify_model_bundle(DEST)
original, bound = FrozenAct(SOURCE), FrozenAct(DEST)
observed = json.loads((ROOT.parent/'nero_act_replay_current_cup_retest_20261019/evidence/retry_readonly_02/report.json').read_text())
maximum = 0.
for row in observed['observations'][::3]:
    rgb = cv2.cvtColor(cv2.imread(row['frame']), cv2.COLOR_BGR2RGB)
    a = original.predict(rgb, observed['actual_state'], previous_gripper_target_m=.095)
    b = bound.predict(rgb, observed['actual_state'], previous_gripper_target_m=.095)
    maximum = max(maximum, float(abs(a-b).max()))
assert maximum == 0.
derivation = dict(mode='deployment_config_only_no_optimization', additional_optimization_steps=0,
    source_model_path=str(SOURCE.resolve()),
    source_checkpoint_sha256=metadata['checkpoint_sha256'],
    source_training_config_sha256=metadata['training_config_sha256'],
    source_training_integrity_sha256=metadata['training_integrity_sha256'],
    deployment_checkpoint_sha256=updated['checkpoint_sha256'],
    deployment_config_sha256=updated['training_config_sha256'],
    deployment_scene_sha256=updated['scene_sha256'], stats_sha256=updated['stats_sha256'],
    all_trained_tensors_bit_identical=True, statistics_file_identical=True,
    real_image_predictions=3, maximum_prediction_difference=maximum,
    source_integrity_is_not_a_direct_training_output_proof_for_derived_checkpoint=True)
(DEST/'deployment_derivation.json').write_text(json.dumps(derivation, indent=2))
updated['deployment_derivation_sha256'] = sha(DEST/'deployment_derivation.json')
(DEST/'runtime_metadata.json').write_text(json.dumps(updated, indent=2))
(ROOT/'evidence/scene_binding_audit.json').write_text(json.dumps(dict(binding, passed=True,
    real_image_predictions=3, maximum_prediction_difference=maximum,
    source_training_config_preserved=True, all_trained_tensors_bit_identical=True,
    deployment_derivation_sha256=updated['deployment_derivation_sha256'],
    deployment_scene=scene, checkpoint_sha256=updated['checkpoint_sha256'],
    execution_ready=False, hardware_motion_sent=False), indent=2))
print(json.dumps(dict(passed=True, maximum_prediction_difference=maximum,
    checkpoint_sha256=updated['checkpoint_sha256'], scene_sha256=updated['scene_sha256'])), flush=True)
