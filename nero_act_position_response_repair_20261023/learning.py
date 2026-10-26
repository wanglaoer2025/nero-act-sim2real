"""Keep the exact existing actor; change only offline data and objectives."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'nero_act_position_recovery_20261022'
spec = importlib.util.spec_from_file_location('sealed_position_learning', SOURCE/'learning.py')
sealed = importlib.util.module_from_spec(spec); spec.loader.exec_module(sealed)
from pipeline import require_checkpoint


def load_for_learning():
    checkpoint=SOURCE/'training/pilot_A/policy_last.ckpt'
    proof=json.loads((checkpoint.parent/'integrity.json').read_text())
    require_checkpoint(proof,checkpoint)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('profile')!='position_recovery_pilot_v2':
        raise ValueError('Exact fixed-last2000-update candidate initializer required')
    policy,stats,groups=sealed.load_for_learning()
    policy.load_state_dict(payload['model'],strict=True)
    return policy,stats,groups
