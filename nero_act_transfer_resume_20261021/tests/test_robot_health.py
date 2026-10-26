"""Current ACT observer accepts planning status2 without hiding actual faults."""
import sys
from types import SimpleNamespace as NS

import pytest

sys.path[:0] = ['/home/user/zyh/src', '/home/user/zyh/tests']
from test_arrival import HealthArm
from test_robot import feedback
from nero_pi05.robot import NeroRobot
from pyAgxArm import AgxArmFactory
from pyAgxArm.protocols.can_protocol.comms.can_comm import CanCommImpl
from pyAgxArm.protocols.can_protocol.drivers.core.arm_driver_abstract import ArmDriverAbstract

from robot_health import observer


@pytest.fixture
def sdk(monkeypatch):
    arm = HealthArm()
    arm.status, arm.error = 2, 0
    def status():
        values = dict(arm_status=arm.status, motion_status=0, ctrl_mode=1)
        if arm.error is not None:
            values['err_code'] = arm.error
        return feedback(NS(**values), arm.age)
    arm.get_arm_status = status
    comm = object.__new__(CanCommImpl)
    arm._ctx = NS(get_comm=lambda: comm)
    monkeypatch.setattr(AgxArmFactory, 'create_arm', lambda _config: arm)
    # Restore these class methods after the real observer installs its send guards.
    for cls, name in [(CanCommImpl, 'send'), (ArmDriverAbstract, '_send_msg'),
                      (ArmDriverAbstract, '_send_msgs')]:
        monkeypatch.setattr(cls, name, getattr(cls, name))
    return arm


@pytest.mark.parametrize('status', [0, 2])
def test_current_observer_preserves_status_and_never_sends(sdk, status):
    sdk.status = status
    robot = observer()
    try:
        snapshot = robot.read_health(require_enabled=True)
        assert snapshot['feedback']['arm_status'] == status
        assert snapshot['feedback']['err_code'] == 0
        with pytest.raises(PermissionError):
            sdk._ctx.get_comm().send(None)
    finally:
        robot.disconnect()
    assert sdk.events == [('init', 'gripper'), 'connect', 'disconnect']


@pytest.mark.parametrize('status,error', [(1, 0), (3, 0), (4, 0), (5, 0), (7, 0),
                                         (255, 0), (2, 16), (2, None)])
def test_other_states_and_independent_errors_still_reject(sdk, status, error):
    sdk.status, sdk.error = status, error
    with pytest.raises(RuntimeError):
        observer()
    assert sdk.events == [('init', 'gripper'), 'connect', 'disconnect']


@pytest.mark.parametrize('fault', ['collision_status', 'driver_overcurrent',
                                   'disabled', 'stale'])
def test_status_two_does_not_hide_driver_faults_or_age(sdk, fault):
    if fault == 'stale':
        sdk.driver_age = 1
    elif fault == 'disabled':
        sdk.flags['driver_enable_status'] = False
    else:
        sdk.flags[fault] = True
    with pytest.raises(TimeoutError if fault == 'stale' else RuntimeError):
        observer()
    assert sdk.events == [('init', 'gripper'), 'connect', 'disconnect']


def test_sdk_default_outside_current_observer_remains_strict(sdk):
    robot = NeroRobot(sdk=sdk, settle_s=0)
    robot.connect()
    try:
        with pytest.raises(RuntimeError):
            robot.read_health(require_enabled=True)
    finally:
        robot.disconnect()
    assert sdk.events == [('init', 'gripper'), 'connect', 'disconnect']
