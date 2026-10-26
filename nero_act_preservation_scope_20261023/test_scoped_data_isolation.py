import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_scoped


class ScopedDataIsolationTests(unittest.TestCase):
    def test_real_data_preparation_never_rewrites_historical_sampling_audits(self):
        source = train_scoped.SOURCE
        historical = [source/'sampling_audit.json', source.parent/'nero_act_sampling_retention_20261023/sampling_audit.json']
        before = {p.resolve(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                  for p in historical}
        payload = torch.load(source.parent/'nero_act_paired_response_20261023/training/pilot_A/policy_last.ckpt',
                             map_location='cpu', weights_only=False)
        write_text = Path.write_text

        def protect_old_audits(path, *args, **kwargs):
            if path.resolve() in before:
                self.fail('Preparing a new trial attempted to rewrite a historical sampling audit')
            return write_text(path, *args, **kwargs)

        with patch.object(Path, 'write_text', protect_old_audits):
            _, batches, _, _, _ = train_scoped.prepare_data(SimpleNamespace(config=payload['config']), payload['stats'])
        np.testing.assert_array_equal(batches, np.load(source/'training/pilot_A/batch_order.npy', allow_pickle=False))
        audit = json.loads((train_scoped.ROOT/'sampling_audit.json').read_text())
        self.assertEqual(audit['original_episodes_sampled'], 60)
        self.assertEqual(audit['dense_episodes_sampled'], 158)
        self.assertEqual(audit['held_out_episodes_optimized'], 0)
        for path, expected in before.items():
            self.assertEqual((hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns), expected)


if __name__ == '__main__':
    unittest.main()
