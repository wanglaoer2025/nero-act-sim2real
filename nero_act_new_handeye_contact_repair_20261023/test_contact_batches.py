import unittest
import numpy as np

from contact_batches import contact_batches,STAGES


class ContactBatchTests(unittest.TestCase):
    def pools(self):
        keys=[f'{kind}:{stage}' for kind in ['center','other'] for stage in STAGES]
        keys+=['contact:close','contact:lift']
        return {key:[i*3+j for j in range(3)] for i,key in enumerate(keys)}

    def test_preserves_all_approach_stages_and_focuses_exact_contact_recoveries(self):
        pools=self.pools()
        batches=contact_batches(pools)
        self.assertEqual(batches.shape,(1000,32))
        self.assertTrue(np.array_equal(batches,contact_batches(pools)))
        for batch in batches:
            for key in pools:
                count=8 if key.startswith('contact:') else 1
                self.assertEqual(sum(x in pools[key] for x in batch),count)

    def test_missing_stage_or_duplicated_sample_rejected(self):
        pools=self.pools()
        pools['contact:lift']=[]
        with self.assertRaises(ValueError):contact_batches(pools)
        pools=self.pools()
        pools['contact:lift']=[pools['contact:close'][0]]
        with self.assertRaises(ValueError):contact_batches(pools)


if __name__=='__main__':unittest.main()
