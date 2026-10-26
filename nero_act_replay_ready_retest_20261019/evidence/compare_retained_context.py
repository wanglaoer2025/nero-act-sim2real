"""Development comparison of existing approved context ACTs at the measured cup."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('version', choices=['run26', 'run27'])
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
locations = dict(
    run26=('nero_act_gripper_contact_focus_20261018', 'run26_current_fixed_scene'),
    run27=('nero_act_current_cup_trial_20261019', 'current_cup_run27'))
name, model_name = locations[args.version]
SOURCE = ROOT.parent/name
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from executor.bundle import verify_model_bundle
from executor.model import BoundedActPolicy, verify_paired_readout_statistics
from simulation.supervised_rollout import rollout

OUT = ROOT/'evidence'/('context_comparison_'+args.version)
OUT.mkdir(exist_ok=False)
model_dir = SOURCE/'model'/model_name
metadata = verify_model_bundle(model_dir)
bundle = torch.load(model_dir/'policy.ckpt', map_location='cpu', weights_only=False)
config = dict(bundle['config'], device='cuda', pretrained_backbone=False)
with (model_dir/'dataset_stats.pkl').open('rb') as stream:
    stats = pickle.load(stream)
verify_paired_readout_statistics(config, stats)
torch.set_num_threads(2)
policy = BoundedActPolicy(config).cuda()
policy.load_state_dict(bundle['model'], strict=True)
policy.eval()
policy.config, policy.device = config, torch.device('cuda')
scene = json.loads((model_dir/'scene_config.json').read_text())
observed = json.loads((ROOT.parent/'nero_act_replay_current_cup_retest_20261019/evidence/retry_readonly_02/report.json').read_text())
scene['pick_xy_m'] = np.median([r['cup_xyz'][:2] for r in observed['observations']], axis=0).tolist()
scene['place_xy_m'] = np.median([r['target_xyz'][:2] for r in observed['observations']], axis=0).tolist()
case = dict(name='current_cup_nominal_context_'+args.version, query_stride=10,
            servo_delay_steps=0, observation_age_s=.05, jitter=False, force=False, seed=20261019)
(OUT/'protocol.json').write_text(json.dumps(dict(mode='development_comparison_not_independent_acceptance',
    hardware_connected=False, motion_sent=False, original_model=str(model_dir), metadata=metadata,
    scene=scene, scenario=case, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
result = rollout(scene, policy, stats, case, OUT/case['name'])
verify_model_bundle(model_dir)
(OUT/'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
