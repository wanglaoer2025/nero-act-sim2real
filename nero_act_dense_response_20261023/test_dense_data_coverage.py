import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

import train_dense


class DataCoverageTests(unittest.TestCase):
    def test_real_hdf5_sampling_keeps_all_original_and_dense_episodes(self):
        root = Path(__file__).resolve().parent
        payload = torch.load(root.parent/'nero_act_paired_response_20261023/training/pilot_A/policy_last.ckpt',
                             map_location='cpu', weights_only=False)
        # This test intentionally exercises the actual files and batches, not a toy sampler.
        _, batches, _, _, _ = train_dense.prepare_data(SimpleNamespace(config=payload['config']), payload['stats'])
        audit = json.loads((root/'sampling_audit.json').read_text())
        accepted = len(json.loads((root/'data/manifest.json').read_text())['episodes'])
        self.assertEqual(batches.shape, (1000, 32))
        self.assertEqual(audit['original_episodes_sampled'], 60)
        self.assertEqual(audit['dense_episodes_sampled'], accepted)
        self.assertEqual(audit['held_out_episodes_optimized'], 0)
        for stage in train_dense.STAGES:
            self.assertEqual(audit['role_draws']['new:'+stage], 1000)
            self.assertEqual(audit['role_draws']['old:'+stage], 2000)


if __name__ == '__main__':
    unittest.main()
