import unittest

from protocol import offsets, classify_partition


class ProtocolTests(unittest.TestCase):
    def test_dense_training_fills_the_gap_without_reusing_validation_offsets(self):
        values = offsets()
        self.assertEqual(len(values), 24)
        self.assertIn((.01, .02), values)
        self.assertIn((-.02, -.01), values)
        for offset in values:
            self.assertEqual(classify_partition(offset), 'train')
        for offset in [(-.015, -.015), (-.015, .015), (.015, -.015), (.015, .015)]:
            self.assertEqual(classify_partition(offset), 'held_out')


if __name__ == '__main__':
    unittest.main()
