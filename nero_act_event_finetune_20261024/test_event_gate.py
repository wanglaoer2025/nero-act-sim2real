import math
import pytest
from event_gate import StableEvent


def measurement(**changes):
    return dict(dict(bilateral=True,supported=True,cup_upright=1.,cup_speed=0.,
        unexpected_contact=False,contract_valid=True,gripper_width=.074),**changes)


def test_close_requires_ten_consecutive_loaded_stable_samples():
    gate=StableEvent('close',20)
    assert not any(gate.update(measurement(),.066) for _ in range(9))
    assert gate.update(measurement(),.066)


def test_interrupted_grip_restarts_stability():
    gate=StableEvent('close',20)
    for _ in range(9):assert not gate.update(measurement(),.066)
    assert not gate.update(measurement(bilateral=False),.066)
    assert not any(gate.update(measurement(),.066) for _ in range(9))
    assert gate.update(measurement(),.066)


def test_unloaded_or_moving_cup_never_closes_gate():
    for changes,target in [({},.075),({'cup_speed':.02},.066),({'cup_upright':.8},.066)]:
        gate=StableEvent('close',20)
        assert not any(gate.update(measurement(**changes),target) for _ in range(30))


def test_release_requires_open_supported_and_no_bilateral_contact():
    for changes in [dict(supported=False),dict(bilateral=True),dict(gripper_width=.08)]:
        gate=StableEvent('release',20)
        row=measurement(bilateral=False,gripper_width=.094);row.update(changes)
        assert not any(gate.update(row,.095) for _ in range(15))
    gate=StableEvent('release',20)
    assert not any(gate.update(measurement(bilateral=False,gripper_width=.094),.095) for _ in range(9))
    assert gate.update(measurement(bilateral=False,gripper_width=.094),.095)


def test_bad_measurements_and_timeout_fail_closed():
    for bad in [measurement(cup_speed=math.nan),measurement(unexpected_contact=True),measurement(contract_valid=False)]:
        with pytest.raises(ValueError):StableEvent('close',20).update(bad,.066)
    with pytest.raises(ValueError):StableEvent('close',20).update({},.066)
    gate=StableEvent('close',20)
    for _ in range(159):assert not gate.update(measurement(bilateral=False),.066)
    with pytest.raises(TimeoutError):gate.update(measurement(bilateral=False),.066)
    with pytest.raises(ValueError):StableEvent('carry',20)
