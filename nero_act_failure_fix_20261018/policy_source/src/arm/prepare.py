"""Move NERO to one fixed bent preparation pose."""

import time

from pyAgxArm import NeroFW

from src.utils.arm import (
    confirm_action,
    create_nero_arm,
    enable_arm,
    get_current_pose,
    move_joint_and_wait,
    print_joint_angles,
    print_pose,
    wait_joint_angles,
)


SPEED_PERCENT = 10

# 已有 ACT 数据仍引用这个固定起点；不要随操作员准备姿态一起改动。
START_JOINTS_DEG = [10.5, -7.25, 5.73, 90.0, 5.73, 5.73, 30.0]

# 操作员准备姿态：只调整腕部三关节。SDK FK 约为
# [-219, -91, 479] mm / [-90, 0, 180] deg，夹爪 TCP 朝 base -X，
# 两指沿 base Y 水平开合。该目标仅完成离线 FK 验证，首次必须空载运行。
TARGET_JOINTS_DEG = [10.5, -7.25, 5.73, 90.0, -1.37, 16.14, 7.51]


def main():
    print("NERO 工作区准备姿态")
    print(f"固定目标关节角（deg）：{TARGET_JOINTS_DEG}")
    print("目标夹爪 TCP 朝 base -X，两指沿 base Y 水平开合。")
    print("脚本只执行一次 move_j；该新腕部姿态尚未完成真机验收。")
    if not confirm_action(
        "移动到准备姿态",
        "关节会从当前位置直接扫向目标，请清空整段运动空间并准备急停。",
        "MOVE",
    ):
        return 0

    robot = create_nero_arm(NeroFW.V120)
    try:
        robot.connect()
        time.sleep(1.0)
        if robot.has_comm_error() or not enable_arm(robot):
            raise RuntimeError("机械臂通信、使能或制动器就绪检查失败")
        robot.set_follower_mode()
        time.sleep(0.5)
        print_joint_angles("当前关节角（度）：", wait_joint_angles(robot))
        move_joint_and_wait(
            robot,
            TARGET_JOINTS_DEG,
            "固定准备姿态",
            speed_percent=SPEED_PERCENT,
            timeout=45.0,
            tolerance_deg=1.5,
        )
        print_pose("准备姿态法兰位姿", get_current_pose(robot))
        print("\n已进入准备姿态；机械臂保持使能，可继续运行抓取程序。")
        return 0
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在发送电子急停。")
        robot.electronic_emergency_stop()
        return 130
    except Exception as exc:
        print(f"\n进入准备姿态失败：{exc}")
        return 1
    finally:
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
