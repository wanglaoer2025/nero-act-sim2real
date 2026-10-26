import hashlib,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import torch
from executor.bundle import verify_model_bundle
from executor.model import BoundedActPolicy, verify_paired_readout_statistics
from simulation.supervised_rollout import rollout

OUT = ROOT/'evidence/bound_scene_generation'
OUT.mkdir(exist_ok=False)
model_dir = ROOT/'model/current_run27'
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
case = dict(name='current_cup_nominal', query_stride=10,
            servo_delay_steps=0, observation_age_s=.05, jitter=False, force=False, seed=20261019)
(OUT/'protocol.json').write_text(json.dumps(dict(mode='frozen_full_ACT_current_scene_generation_not_hardware',
    hardware_connected=False, motion_sent=False, original_model=str(model_dir), metadata=metadata,
    scene=scene, scenario=case, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
result = rollout(scene, policy, stats, case, OUT/case['name'])
verify_model_bundle(model_dir)
(OUT/'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
