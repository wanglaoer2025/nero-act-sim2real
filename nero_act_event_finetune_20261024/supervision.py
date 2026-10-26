"""Exclude conflicting WAIT query starts; retain every physical target unchanged."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('_original_transition_audit',
    Path(__file__).resolve().parent.parent/'nero_act_observation_repair_20261023/transition_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def remove_wait_aliases(observed,actions,rgb,stages,valid,lift_index):
    observed,actions,stages,valid=[np.asarray(x) for x in [observed,actions,stages,valid]]
    n=len(actions)
    if (observed.shape!=(n,15) or actions.shape!=(n,8) or stages.shape!=(n,)
            or valid.shape!=(n,) or not 0<=lift_index<n or not valid[lift_index]
            or stages[lift_index]!='lift' or not np.isfinite(observed[valid]).all()):
        raise ValueError('Complete valid physical descriptors and first moving lift query required')
    removed=[int(i) for i in np.flatnonzero(valid & (stages=='close')) if audit.ambiguous_transition(
        observed[i],observed[lift_index],actions[i],actions[lift_index],rgb[i],rgb[lift_index])]
    mask=valid.copy();mask[removed]=False
    return mask,removed
