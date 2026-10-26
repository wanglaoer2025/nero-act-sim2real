"""Actual HDF5-to-batch regression for the formerly omitted60 train episodes."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import train_retained as pipeline


class ActualCoverageTests(unittest.TestCase):
    def test_actual_retained_data_enters_batches_and_validation_stays_out(self):
        source=pipeline.ROOT.parent/'nero_act_paired_response_20261023/training/pilot_A/policy_last.ckpt'
        payload=torch.load(source,map_location='cpu',weights_only=False)
        original=json.loads((pipeline.trainer.prior.ROOT/'data/manifest.json').read_text())
        train={Path(r['path']).resolve() for r in original['episodes'] if r['split']=='train'}
        validation={Path(r['path']).resolve() for r in original['episodes'] if r['split']=='val'}
        with tempfile.TemporaryDirectory() as directory,patch.object(pipeline,'ROOT',Path(directory)):
            dataset,batches,_,_,_=pipeline.prepare_data(SimpleNamespace(config=payload['config']),payload['stats'])
            actual={dataset.paths[dataset.sample_starts[int(i)][0]].resolve() for i in batches.ravel()}
            self.assertEqual(len(train),60)
            self.assertTrue(train<=actual)
            self.assertFalse(validation&actual)
            audit=json.loads((Path(directory)/'sampling_audit.json').read_text())
            self.assertEqual(audit['actually_sampled_original_episodes'],60)
            self.assertEqual(audit['role_draws']['retained_clearance'],2000)
            self.assertEqual(audit['role_draws']['retained_align'],2000)


if __name__=='__main__':unittest.main()
