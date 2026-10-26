import unittest
import numpy as np

try:
    from sampling import make_batches,STAGES
except ImportError:
    make_batches=None
    STAGES=('approach','descend','close','lift','carry','place','release','retreat')


class SamplingTests(unittest.TestCase):
    def pools(self):
        return {'correction':[1000,1001],**{f'{kind}:{stage}':[100+10*i+(0 if kind=='new' else 1)]
            for i,stage in enumerate(STAGES) for kind in ['new','old']}}

    def test_every_batch_retains_all_eight_task_stages_and_new_recovery(self):
        self.assertIsNotNone(make_batches,'Missing complete-flow balanced sampling')
        pools=self.pools();rows=make_batches(pools,steps=7,seed=42)
        self.assertEqual(rows.shape,(7,32))
        for row in rows:
            self.assertEqual(np.isin(row,pools['correction']).sum(),8)
            for stage in STAGES:
                self.assertEqual(np.isin(row,pools['new:'+stage]).sum(),1)
                self.assertEqual(np.isin(row,pools['old:'+stage]).sum(),2)
        np.testing.assert_array_equal(rows,make_batches(pools,steps=7,seed=42))

    def test_missing_recovery_or_task_stage_prevents_training(self):
        self.assertIsNotNone(make_batches,'Missing complete-flow balanced sampling')
        for key in ['correction','new:release','old:descend']:
            pools=self.pools();pools[key]=[]
            with self.subTest(key=key),self.assertRaises(ValueError):make_batches(pools,steps=2,seed=42)


if __name__=='__main__':unittest.main()
