"""Keep complete health snapshots without retaining per-sample cyclic-GC graphs."""
import json

import numpy as np


class HealthTrace:
    def __init__(self):
        self.samples = []

    def append(self, value):
        self.samples.append(json.dumps(value, separators=(',', ':'), default=lambda x:
            x.tolist() if isinstance(x, np.ndarray) else str(x)))

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return (json.loads(row) for row in self.samples)
