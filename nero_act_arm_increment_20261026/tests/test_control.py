from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import claim_training,configure_runtime
from experiment import ArmPolicy


def test_training_output_can_only_be_claimed_once(tmp_path):
    claim_training(tmp_path)
    with pytest.raises(FileExistsError):claim_training(tmp_path)


def test_cuda_runtime_does_not_require_unsupported_strict_cumsum():
    configure_runtime()
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic


def test_new_optimizer_has_only_readout_parameters_and_no_old_moments():
    s=dict(qpos_mean=[0.]*8,qpos_std=[1.]*8,action_mean=[0.]*8,action_std=[1.]*8)
    p=ArmPolicy(s,dict(mean=[0.]*7,std=[.002]*7),variant='B',pretrained=False)
    opt=p.optimizer()
    assert not opt.state
    assert {id(x) for g in opt.param_groups for x in g['params']}=={id(x) for x in p.arm_readout.parameters()}
