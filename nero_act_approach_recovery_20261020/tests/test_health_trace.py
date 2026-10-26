import gc
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class HealthTraceTests(unittest.TestCase):
    def new_trace(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.health_trace'),
                             'Health recording must avoid retaining nested GC graphs')
        from executor.health_trace import HealthTrace
        return HealthTrace()

    def test_health_export_preserves_every_snapshot_and_numpy_state(self):
        trace = self.new_trace()
        sample = dict(state=np.array([.1, .095]), flags={'enabled': True}, stamp=17.)
        trace.append(sample)
        sample['state'][0] = .2
        sample['flags']['enabled'] = False
        trace.append(sample)
        from executor.runtime import json_write
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'health.json'
            json_write(target, trace)
            self.assertEqual(json.loads(target.read_text()), [
                dict(state=[.1, .095], flags={'enabled': True}, stamp=17.),
                dict(state=[.2, .095], flags={'enabled': False}, stamp=17.)])
        self.assertEqual(len(trace), 2)

    def test_long_recording_does_not_grow_the_cyclic_gc_scan_graph(self):
        trace = self.new_trace()
        sample = dict(state=np.zeros(8), drivers=[dict(flags={'fault': False}) for _ in range(7)])
        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            before = len(gc.get_objects())
            for _ in range(1000):
                trace.append(sample)
            self.assertLess(len(gc.get_objects())-before, 100,
                            'Each sample must not add a nested graph to full-GC scans')
            self.assertEqual(len(trace), 1000)
        finally:
            if was_enabled:
                gc.enable()


if __name__ == '__main__':
    unittest.main()
