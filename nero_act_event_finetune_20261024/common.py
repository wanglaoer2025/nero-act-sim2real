"""Load the exact approved architecture, statistics and trainable scope."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nero_act_new_handeye_policy_repair_20261023'
CHECKPOINT=PREVIOUS/'training/pilot_A/policy_last.ckpt'
CHECKPOINT_SHA='1a28f4fddd3095791403028b639b99f9775b91caab45c950c2d0f518955bc0f0'
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
spec=importlib.util.spec_from_file_location('_event_finetune_base',PREVIOUS/'train_adaptation.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
historical=base.historical


def load_initial():
    if historical.snapshot([CHECKPOINT])[str(CHECKPOINT)]!=CHECKPOINT_SHA:
        raise ValueError('Frozen1a28 initializer required')
    policy,stats,groups=base.load_initial()
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    if payload['config']!=policy.config or any(not np.array_equal(payload['stats'][k],stats[k]) for k in stats):
        raise ValueError('Do not change architecture or paired statistics')
    policy.load_state_dict(payload['model'],strict=True)
    return policy.eval(),stats,groups
