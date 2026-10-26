"""Offline ACT data processing must not require hardware SDK imports."""

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class OfflineActImportTest(unittest.TestCase):
    def test_factory_keeps_default_v120_and_explicit_firmware(self):
        from src.utils.arm import create_nero_arm

        sdk = SimpleNamespace(
            NeroFW=SimpleNamespace(V120="v120"),
            ArmModel=SimpleNamespace(NERO="nero"),
            create_agx_arm_config=lambda **values: values,
            AgxArmFactory=SimpleNamespace(create_arm=lambda config: config),
        )
        with patch.dict(sys.modules, {"pyAgxArm": sdk}):
            self.assertEqual(create_nero_arm(), {
                "robot": "nero", "firmeware_version": "v120",
                "interface": "socketcan", "channel": "can0",
            })
            self.assertEqual(create_nero_arm("custom", "can1")["firmeware_version"], "custom")
            self.assertEqual(create_nero_arm("custom", "can1")["channel"], "can1")
            self.assertIsNone(create_nero_arm(None)["firmeware_version"])

    def test_hdf5_roundtrip_and_training_import_without_hardware_sdks(self):
        code = r'''
import builtins
import tempfile
from pathlib import Path

original_import = builtins.__import__
def without_hardware_sdks(name, *args, **kwargs):
    if name.split('.')[0] in {'pyAgxArm', 'pyzed'}:
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_hardware_sdks

import numpy as np
from src.act.episode_io import save_episode
from src.act.nero_dataset import NeroEpisodicDataset, compute_norm_stats
from src.act.train import main

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'episode_0.hdf5'
    qpos = np.zeros((3, 7), dtype=np.float32)
    gripper = np.full((3, 1), 0.075, dtype=np.float32)
    actions = np.concatenate([qpos, gripper], axis=1)
    save_episode(path, np.full((3, 12, 16, 3), 128, dtype=np.uint8),
                 qpos, actions, np.arange(3) * 0.05, gripper_states=gripper)
    stats = compute_norm_stats([path])
    dataset = NeroEpisodicDataset([path], ['camera_0'], stats,
                                image_size=(90, 160), fixed_start_ts=0)
    images, state, targets, padding, start = dataset[0]
    assert tuple(images.shape) == (1, 3, 90, 160)
    assert tuple(state.shape) == (8,)
    assert targets.shape[-1] == 8
    assert bool(np.isfinite(state.numpy()).all())
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_hardware_creation_still_requires_the_sdk(self):
        code = r'''
import builtins
original_import = builtins.__import__
def without_sdk(name, *args, **kwargs):
    if name.split('.')[0] == 'pyAgxArm':
        raise ModuleNotFoundError("No module named 'pyAgxArm'", name='pyAgxArm')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_sdk
from src.utils.arm import create_nero_arm
try:
    create_nero_arm()
except ModuleNotFoundError as error:
    assert error.name == 'pyAgxArm'
else:
    raise AssertionError('Hardware creation unexpectedly succeeded without SDK')
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
