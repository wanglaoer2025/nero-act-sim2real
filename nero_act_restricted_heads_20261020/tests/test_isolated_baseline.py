import importlib.util
import io
import json
import time
from types import SimpleNamespace

import numpy as np


def test_online_baseline_preserves_previous_submitted_gripper():
    from executor.runtime import LiveWorker
    state = [0.] * 7 + [.074]
    stamp = time.monotonic()
    worker = object.__new__(LiveWorker)
    worker.session = SimpleNamespace(
        spin=lambda _: None, actual=lambda: state,
        last_target=np.array([0.] * 7 + [.065]),
        latest={'/feedback/joint_states': {'receipt': stamp}})
    worker.process = SimpleNamespace(stdin=io.StringIO())
    worker.records = []
    worker.receive = lambda _: dict(type='prediction', chunk=[state]*20, details=dict(
        observed_monotonic_s=stamp, feedback_state=state,
        camera_received_monotonic_s=time.monotonic(), previous_gripper_target_m=.065))
    worker.predict()
    request = json.loads(worker.process.stdin.getvalue())
    assert request['previous_gripper_target_m'] == .065


def test_baseline_uses_compact_complete_health_trace(tmp_path):
    assert importlib.util.find_spec('executor.health_trace') is not None, 'Complete compact health storage missing'
    from executor.health_trace import HealthTrace
    from executor.runtime import json_write
    trace = HealthTrace()
    sample = dict(state=np.array([.1, .095]), driver=dict(enabled=True))
    trace.append(sample)
    sample['driver']['enabled'] = False
    sample['state'][0] = .2
    trace.append(sample)
    path = tmp_path/'trace.json'
    json_write(path, trace)
    assert json.loads(path.read_text()) == [
        dict(state=[.1, .095], driver=dict(enabled=True)),
        dict(state=[.2, .095], driver=dict(enabled=False))]
