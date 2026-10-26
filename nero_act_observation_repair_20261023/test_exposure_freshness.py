import importlib.util
from pathlib import Path
import unittest


class ExposureFreshnessTests(unittest.TestCase):
    def test_duplicate_is_not_a_second_sample_and_clock_regression_is_an_error(self):
        path=Path(__file__).resolve().parent/'depth_quality.py'
        spec=importlib.util.spec_from_file_location('_quality_freshness',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertTrue(hasattr(m,'fresh_exposure'),'Missing duplicate-frame rejection without data reuse')
        self.assertTrue(m.fresh_exposure(100,None))
        self.assertFalse(m.fresh_exposure(100,100))
        self.assertTrue(m.fresh_exposure(101,100))
        with self.assertRaises(ValueError):m.fresh_exposure(99,100)


if __name__=='__main__':unittest.main()
