"""Offline diagnostic: frozen ACT on the recorded post-stop source, no hardware."""
import hashlib
import json
import pickle
import sys
from pathlib import Path

SOURCE = Path('/home/user/wq/nero_act_replay_current_scene_20261019')
OUTPUT = Path(__file__).resolve().parent/'fresh_run11_recorded_scene'
sys.path.insert(0, str(SOURCE))

import torch
from executor.bundle import verify_model_bundle
from executor.model import BoundedActPolicy, verify_paired_readout_statistics
from simulation.supervised_rollout import rollout

if OUTPUT.exists():
    raise FileExistsError(OUTPUT)
OUTPUT.mkdir()
model_dir = SOURCE/'model/frozen_run11'
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
scene['pick_xy_m'] = [-0.528298610038193, 0.00993258465659119]
case = dict(name='recorded_post_stop_source', query_stride=10, servo_delay_steps=0,
            observation_age_s=.05, jitter=False, force=False, seed=20261019)
protocol = dict(mode='offline_diagnostic_not_hardware_acceptance',
                hardware_connected=False, motion_sent=False,
                recorded_source_frame='camera_00016.png',
                scene=scene, model_metadata=metadata, scenario=case,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
result = rollout(scene, policy, stats, case, OUTPUT/case['name'])
verify_model_bundle(model_dir)
(OUTPUT/'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
