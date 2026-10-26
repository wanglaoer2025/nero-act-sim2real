"""Continue the sealed position-response experiment without changing its inference."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'nero_act_position_response_repair_20261023'
sys.path.insert(0,str(SOURCE))
import train as prior
from pipeline import require_checkpoint,require_unchanged,snapshot

INITIALIZER=SOURCE/'training/pilot_A/policy_last.ckpt'
PROFILE='position_response_continuation_v1'

def load_initial():
    proof=json.loads((INITIALIZER.parent/'integrity.json').read_text())
    require_checkpoint(proof,INITIALIZER)
    expected=json.loads((ROOT/'plan.json').read_text())['initializer_sha256']
    if snapshot([INITIALIZER])[str(INITIALIZER.resolve())]!=expected:
        raise ValueError('Continuation requires the declared latest fixed-last weights')
    payload=torch.load(INITIALIZER,map_location='cpu',weights_only=False)
    if payload.get('profile')!='position_response_loss_repair_v1' or payload.get('actual_updates')!=1000:
        raise ValueError('Wrong initializer profile or completed update count')
    policy,stats,groups=prior.load_for_learning()
    if payload['config']!=policy.config or any(not np.array_equal(payload['stats'][k],stats[k]) for k in stats):
        raise ValueError('Paired unchanged ACT configuration and statistics required')
    policy.load_state_dict(payload['model'],strict=True)
    return policy,stats,groups
