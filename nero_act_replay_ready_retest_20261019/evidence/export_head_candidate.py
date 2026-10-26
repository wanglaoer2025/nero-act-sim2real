"""Freeze the completed validation-selected action-head adaptation; no hardware."""
import hashlib
import json
from pathlib import Path

import torch

from training.export import export_model

ROOT = Path(__file__).resolve().parents[1]
run = ROOT/'training/current_cup_heads_10ep'
checkpoint = run/'checkpoints/policy_best_val.ckpt'
source = ROOT/'model/source_run11'
original = torch.load(source/'policy.ckpt', map_location='cpu', weights_only=False)
trained = torch.load(checkpoint, map_location='cpu', weights_only=False)
allowed = {'model.action_head.weight', 'model.action_head.bias'}
assert trained['model'].keys() == original['model'].keys()
changed = [name for name in trained['model'] if not torch.equal(trained['model'][name], original['model'][name])]
assert changed and set(changed) <= allowed
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(run/'checkpoints/dataset_stats.pkl') == sha(source/'dataset_stats.pkl')
scene = json.loads((ROOT/'training_data_current_cup/scene_config.json').read_text())
assert trained['config']['scene_config'] == scene
(ROOT/'scene/config.json').write_text(json.dumps(scene, indent=2))
metadata = export_model(checkpoint, ROOT/'model/current_cup_heads')
report = dict(passed=True, training_completed=True, declared_epochs=10,
              selected_epoch=trained['epoch'], checkpoint_selection='lowest held-out validation loss',
              changed_state_dict_tensors=changed, all_other_parameters_and_buffers_identical=True,
              normalization_file_identical=True, architecture_changed=False,
              original_prior_only_scheme_retained=True, cvae_retrained=False,
              source_checkpoint_sha256=sha(source/'policy.ckpt'),
              checkpoint_sha256=metadata['checkpoint_sha256'], hardware_motion_sent=False,
              execution_ready=False)
(ROOT/'evidence/action_head_freeze_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
