"""Current ACT receive-only health entry; status2 alone is not a hardware fault.

The SDK's strict default and sealed historical entry remain unchanged. This
observer opts into its existing NO_SOLUTION allowance without sending a reset.
"""
import sys


def observer():
    sys.path[:0] = ['/home/user/zyh', '/home/user/zyh/src']
    from scripts.nero_methylene_teacher import reject_sdk_send
    from pyAgxArm.protocols.can_protocol.comms.can_comm import CanCommImpl
    from pyAgxArm.protocols.can_protocol.drivers.core.arm_driver_abstract import ArmDriverAbstract
    CanCommImpl.send = ArmDriverAbstract._send_msg = ArmDriverAbstract._send_msgs = reject_sdk_send
    from pyAgxArm import create_agx_arm_config, AgxArmFactory
    from nero_pi05.robot import NeroRobot
    sdk = AgxArmFactory.create_arm(create_agx_arm_config(robot='nero', comm='can',
                                    channel='can0', firmeware_version='v120'))
    robot = NeroRobot(sdk=sdk, settle_s=.5, max_age_s=.15, allow_no_solution=True)
    try:
        robot.connect()
        comm = sdk._ctx.get_comm()
        if getattr(comm.send, '__func__', comm.send) is not reject_sdk_send:
            raise RuntimeError('Observer transport send guard is missing')
        robot.read_health(require_enabled=True)
        return robot
    except BaseException:
        robot.disconnect()
        raise
