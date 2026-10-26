import unittest
import numpy as np

try:
    from retention import make_batches
except ImportError:
    make_batches=None

STAGES=('approach','descend','close','lift','carry','place','release','retreat')


class RetentionTests(unittest.TestCase):
    def pools(self):
        return dict(new_correction=[100,101],retained_align=[110,111],retained_clearance=[120,121],
            **{f'{kind}:{stage}':[200+10*i+(0 if kind=='new' else 1)] for i,stage in enumerate(STAGES) for kind in ['new','old']})

    def test_original_position_and_clearance_corrections_are_actually_sampled(self):
        self.assertIsNotNone(make_batches,'Original correction data must enter optimization')
        pools=self.pools();batches=make_batches(pools,steps=20,seed=231019)
        self.assertEqual(batches.shape,(20,32))
        for row in batches:
            self.assertEqual(np.isin(row,pools['new_correction']).sum(),4)
            self.assertEqual(np.isin(row,pools['retained_align']).sum(),2)
            self.assertEqual(np.isin(row,pools['retained_clearance']).sum(),2)
            for stage in STAGES:
                self.assertEqual(np.isin(row,pools['new:'+stage]).sum(),1)
                self.assertEqual(np.isin(row,pools['old:'+stage]).sum(),2)

    def test_missing_original_correction_group_prevents_training(self):
        self.assertIsNotNone(make_batches,'Original correction data must enter optimization')
        for key in ['retained_align','retained_clearance','new_correction','old:release']:
            pools=self.pools();pools[key]=[]
            with self.subTest(key=key),self.assertRaises(ValueError):make_batches(pools,steps=3,seed=42)


if __name__=='__main__':unittest.main()
