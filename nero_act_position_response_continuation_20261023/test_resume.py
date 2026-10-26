"""Catch accidentally restarting the previous pilot instead of continuing its weights."""
import unittest

import torch

from resume import INITIALIZER,load_initial


class ContinuationInitializerTests(unittest.TestCase):
    def test_starts_from_exact_latest_weights(self):
        policy,stats,groups=load_initial()
        payload=torch.load(INITIALIZER,map_location='cpu',weights_only=False)
        changed=[name for name,value in policy.state_dict().items()
                 if not torch.equal(value.detach().cpu(),payload['model'][name])]
        self.assertEqual(changed,[], 'Continuation must start from latest fixed-last weights')
        self.assertEqual(policy.config,payload['config'])
        self.assertTrue(groups)


if __name__=='__main__':unittest.main()
