"""演示 YOLO + ZED 三维定位及人工确认的抓取放置。

用法（仓库根目录）：
    python -m src.demo.yolo_pick_place --help
    python -m src.demo.yolo_pick_place
    python -m src.demo.yolo_pick_place --execute

默认只预览检测结果；``--execute`` 会控制机械臂和夹爪，且每个运动阶段仍需
输入对应的大写确认口令。首次实机验证应从悬停和低速空载开始。
"""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import pyzed.sl as sl

from src.utils.arm import adaptive_gripper_grasp
from src.utils.vision import (
    cylinder_center_from_surface,
    flange_pose_for_tcp,
    median_xyz_in_box,
    pose_to_transform,
    rotate_image_180,
    stable_point,
    transform_points,
    validate_calibration_quality,
    validate_rigid_transform,
)


WINDOW_NAME = "YOLOv8 RGB-D | m: lock/continue | r: reset | q: quit"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default="yolov8n.pt", help="YOLOv8 权重名称或本地路径"
    )
    parser.add_argument("--source-class", help="待抓物类别名或类别 ID")
    parser.add_argument(
        "--destination-class", help="承接物类别名或类别 ID"
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--max-spread-mm", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true", help="允许连接并控制机械臂")
    parser.add_argument(
        "--hover-only",
        action="store_true",
        help="只移动到杯子 +X 侧高位，不下降，保持夹爪张开",
    )
    parser.add_argument(
        "--side-approach-only",
        action="store_true",
        help="张开夹爪按 +X 预抓取路径接近，在杯面前停止",
    )
    parser.add_argument(
        "--pick-place",
        action="store_true",
        help="监督执行杯子侧向抓取并放到 book 中心",
    )
    parser.add_argument("--hover-clearance-mm", type=float, default=200.0)
    parser.add_argument("--source-radius-mm", type=float, default=40.0)
    parser.add_argument("--surface-center-correction-mm", type=float, default=1.0)
    parser.add_argument("--pregrasp-distance-mm", type=float, default=140.0)
    parser.add_argument("--final-stand-off-mm", type=float, default=50.0)
    parser.add_argument("--transit-speed-percent", type=int, default=30)
    parser.add_argument("--precision-speed-percent", type=int, default=5)
    parser.add_argument("--motion-timeout-s", type=float, default=60.0)
    parser.add_argument("--flange-rpy-deg", type=float, nargs=3)
    parser.add_argument("--tcp-offset-mm", type=float, nargs=3)
    parser.add_argument(
        "--workspace-mm",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument(
        "--source-offset-mm",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="在 base 坐标系修正待抓物坐标，单位 mm",
    )
    parser.add_argument(
        "--destination-offset-mm",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="在 base 坐标系修正放置目标坐标，单位 mm",
    )
    parser.add_argument("--grasp-z-offset-mm", type=float, default=0.0)
    parser.add_argument("--approach-height-mm", type=float, default=80.0)
    parser.add_argument("--destination-clearance-mm", type=float, default=80.0)
    parser.add_argument("--cup-height-mm", type=float, default=220.0)
    parser.add_argument("--placement-gap-mm", type=float, default=5.0)
    parser.add_argument("--release-retreat-mm", type=float, default=50.0)
    parser.add_argument("--gripper-open-mm", type=float, default=95.0)
    parser.add_argument(
        "--gripper-close-mm",
        type=float,
        default=0.0,
        help="自适应闭合的最小目标宽度，不是预估物体宽度",
    )
    parser.add_argument("--gripper-contact-force-n", type=float, default=0.3)
    parser.add_argument("--gripper-force-n", type=float, default=1.0)
    parser.add_argument("--gripper-force-step-n", type=float, default=0.1)
    parser.add_argument("--gripper-force-step-delay-s", type=float, default=0.2)
    parser.add_argument("--gripper-min-width-gap-mm", type=float, default=3.0)
    parser.add_argument("--gripper-min-force-delta-n", type=float, default=0.08)
    parser.add_argument("--gripper-grasp-timeout-s", type=float, default=5.0)
    parser.add_argument("--auto-lock", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--detection-timeout-s", type=float, default=30.0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--act-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--act-secondary-camera-device", help=argparse.SUPPRESS)
    parser.add_argument("--act-secondary-camera-rotation-deg", type=int,
                        choices=(0, 90, 180, 270), default=0, help=argparse.SUPPRESS)
    parser.add_argument("--act-frequency", type=float, default=20.0, help=argparse.SUPPRESS)
    parser.add_argument("--act-action-lead-ms", type=float, default=0.0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--act-position-group", choices=("1", "2", "3"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if sum((args.execute, args.hover_only, args.side_approach_only, args.pick_place)) > 1:
        parser.error("execute、hover-only、side-approach-only、pick-place 只能选择一个")
    if bool(args.source_class) != bool(args.destination_class):
        parser.error("source-class 与 destination-class 必须同时提供或同时省略")
    if args.source_class and args.source_class == args.destination_class:
        parser.error("第一版要求 source-class 与 destination-class 不同")
    if args.stable_frames < 2 or args.max_spread_mm <= 0:
        parser.error("stable-frames 至少为 2，max-spread-mm 必须大于 0")
    if not 0 < args.confidence <= 1 or args.imgsz <= 0:
        parser.error("confidence 必须在 (0, 1] 范围内，imgsz 必须大于 0")
    if args.detection_timeout_s <= 0 or not 0 < args.act_frequency <= 20:
        parser.error("detection-timeout-s 必须大于 0，act-frequency 必须在 (0,20] Hz")
    if args.act_action_lead_ms < 0:
        parser.error("act-action-lead-ms 不得为负")
    if args.act_output and args.act_position_group is None:
        parser.error("ACT 自动示教必须提供 --act-position-group")
    if not 0 <= args.gripper_close_mm < args.gripper_open_mm <= 100:
        parser.error("夹爪宽度必须满足 0 <= close < open <= 100 mm")
    if not 0 < args.gripper_contact_force_n <= args.gripper_force_n <= 3:
        parser.error("夹爪力必须满足 0 < contact-force <= hold-force <= 3 N")
    if (
        args.gripper_force_step_n <= 0
        or args.gripper_force_step_delay_s < 0
        or args.gripper_min_width_gap_mm <= 0
        or args.gripper_min_force_delta_n <= 0
        or args.gripper_grasp_timeout_s <= 0
    ):
        parser.error("夹爪步长、检测门限和超时参数无效")
    if (
        args.approach_height_mm <= 0
        or args.destination_clearance_mm <= 0
        or args.cup_height_mm <= 0
        or args.placement_gap_mm < 0
        or args.release_retreat_mm <= args.source_radius_mm
        or args.hover_clearance_mm <= 0
        or args.source_radius_mm <= 0
        or args.surface_center_correction_mm < 0
        or args.pregrasp_distance_mm <= args.source_radius_mm
        or args.final_stand_off_mm <= 0
        or not 1 <= args.transit_speed_percent <= 100
        or not 1 <= args.precision_speed_percent <= 100
        or args.motion_timeout_s <= 0
    ):
        parser.error("接近路径参数无效；预抓取距离必须大于杯子半径")
    if args.execute and (
        not args.source_class
        or any(
            value is None
            for value in (args.flange_rpy_deg, args.tcp_offset_mm, args.workspace_mm)
        )
    ):
        parser.error(
            "--execute 必须提供两个类别、--flange-rpy-deg、--tcp-offset-mm 和 --workspace-mm"
        )
    if (args.hover_only or args.side_approach_only or args.pick_place) and (
        not args.source_class or args.workspace_mm is None
    ):
        parser.error("悬停、侧向接近和搬运模式必须提供两个类别和 --workspace-mm")
    if args.execute or args.hover_only or args.side_approach_only or args.pick_place:
        values = [
            *args.workspace_mm,
            *args.source_offset_mm,
            *args.destination_offset_mm,
        ]
        if args.execute:
            values += [*args.flange_rpy_deg, *args.tcp_offset_mm]
        if not np.isfinite(values).all():
            parser.error("运动参数必须是有限数值")
        xmin, xmax, ymin, ymax, zmin, zmax = args.workspace_mm
        if not (xmin < xmax and ymin < ymax and zmin < zmax):
            parser.error("workspace-mm 的每组 min 必须小于 max")
    return args


def class_id(names, value):
    """把模型类别名或数字 ID 解析为类别 ID。"""
    mapping = names if isinstance(names, dict) else dict(enumerate(names))
    if value.lstrip("-").isdigit():
        result = int(value)
    else:
        result = next((index for index, name in mapping.items() if name == value), -1)
    if result not in mapping:
        raise ValueError(f"模型中没有类别 {value!r}；可用类别：{mapping}")
    return result


def inside_workspace(point, bounds):
    x, y, z = point
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax


def pick_place_tcp_targets(args, source_xyz, destination_xyz):
    """计算杯子抓取、抬升、放置和松爪退出的 TCP 目标。"""
    grasp = source_xyz + [0, 0, args.grasp_z_offset_mm]
    lifted = grasp + [0, 0, args.approach_height_mm]
    if getattr(args, "act_output", None):
        # ponytail: ACT 的杯子和蓝卡共用桌面；改录高台目标时再显式提供支撑面 Z。
        placement_z_mm = grasp[2] + args.placement_gap_mm
    else:
        placement_z_mm = (
            destination_xyz[2]
            + args.cup_height_mm / 2.0
            + args.placement_gap_mm
        )
    placement = np.array([destination_xyz[0], destination_xyz[1], placement_z_mm])
    above = placement + [0, 0, args.destination_clearance_mm]
    retreat = placement + [args.release_retreat_mm, 0, 0]
    return_clearance = retreat + [0, 0, args.destination_clearance_mm]
    return grasp, lifted, above, placement, retreat, return_clearance


def execute_pick_hover(args, source_xyz, destination_xyz):
    """抓起 source，并低速直线搬运到 destination 上方；不自动松爪。"""
    from pyAgxArm import NeroFW

    from src.utils.arm import (
        confirm_action,
        create_nero_arm,
        enable_arm,
        move_pose_and_wait,
        print_pose,
        wait_gripper_feedback,
    )

    source_grasp = source_xyz + [0, 0, args.grasp_z_offset_mm]
    source_above = source_grasp + [0, 0, args.approach_height_mm]
    destination_above = destination_xyz + [0, 0, args.destination_clearance_mm]
    tcp_targets = [source_grasp, source_above, destination_above]
    if not all(inside_workspace(point, args.workspace_mm) for point in tcp_targets):
        raise ValueError(f"TCP 目标超出工作空间：{[point.tolist() for point in tcp_targets]}")
    poses = [
        flange_pose_for_tcp(point, args.tcp_offset_mm, args.flange_rpy_deg)
        for point in tcp_targets
    ]
    if not all(
        inside_workspace(np.asarray(pose[:3]) * 1000, args.workspace_mm)
        for pose in poses
    ):
        raise ValueError("TCP 偏移换算后的法兰目标超出工作空间")
    grasp_pose, source_above_pose, destination_above_pose = poses

    print(f"\nsource base XYZ (mm): {source_xyz.round(2).tolist()}")
    print(f"destination base XYZ (mm): {destination_xyz.round(2).tolist()}")
    print_pose("抓取法兰位姿", grasp_pose)
    print_pose("抓取点上方法兰位姿", source_above_pose)
    print_pose("目标物上方法兰位姿", destination_above_pose)
    if not confirm_action(
        "抓取物体并移动到目标上方",
        "机械臂和夹爪将执行完整抓取与搬运路径，请确认 TCP、工作空间和急停。",
        "PICK",
    ):
        return

    robot = create_nero_arm(NeroFW.V120)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    wait_options = {"timeout": 30.0, "position_tolerance": 0.004}
    try:
        robot.connect()
        time.sleep(1.0)
        if robot.has_comm_error() or not enable_arm(robot):
            raise RuntimeError("机械臂通信或使能失败")
        wait_gripper_feedback(gripper)
        gripper.move_gripper_m(
            value=args.gripper_open_mm / 1000.0,
            force=args.gripper_contact_force_n,
        )
        time.sleep(1.5)
        baseline_force_n = float(wait_gripper_feedback(gripper).force)
        print(f"夹爪空载力基线：{baseline_force_n:.3f} N")
        move_pose_and_wait(
            robot,
            "move_p",
            "移动到待抓物上方",
            source_above_pose,
            speed_percent=args.transit_speed_percent,
            **wait_options,
        )
        move_pose_and_wait(
            robot,
            "move_l",
            "垂直下降",
            grasp_pose,
            speed_percent=args.precision_speed_percent,
            **wait_options,
        )
        adaptive_gripper_grasp(gripper, args, baseline_force_n)
        move_pose_and_wait(
            robot,
            "move_l",
            "垂直抬起",
            source_above_pose,
            speed_percent=args.precision_speed_percent,
            **wait_options,
        )
        move_pose_and_wait(
            robot,
            "move_l",
            "移动到目标物上方",
            destination_above_pose,
            speed_percent=args.transit_speed_percent,
            **wait_options,
        )
        print("\n已到目标物上方并保持夹持；本演示不会自动下降或松爪。")
    except KeyboardInterrupt:
        print("\n用户中止，正在发送电子急停。")
        robot.electronic_emergency_stop()
    finally:
        robot.disconnect()


def execute_side_sequence(args, source_xyz, destination_xyz, *, preconfirmed=False):
    """从固定初始位姿验证侧向路径，或监督抓杯并放到 book 中心。"""
    from pyAgxArm import NeroFW

    from src.utils.arm import (
        PoseMotionError,
        confirm_action,
        create_nero_arm,
        enable_arm,
        get_current_pose,
        hold_current_joint_position,
        move_joint_and_wait,
        move_pose_and_wait,
        print_pose,
        radians_to_degrees,
        wait_gripper_feedback,
        wait_joint_angles,
    )

    config_path = Path("config/tcp_offset.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("method") != "manual_measurement" or not config.get("approximate"):
        raise ValueError("当前侧向流程要求标记为 approximate 的手工 TCP 配置")
    tcp_offset_mm = np.asarray(config["tcp_offset_flange_xyz_m"], dtype=float) * 1000
    flange_rpy_deg = np.asarray(config["side_grasp_flange_rpy_deg"], dtype=float)
    rotation = pose_to_transform(
        [0, 0, 0, *np.radians(flange_rpy_deg)]
    )[:3, :3]
    tool_direction = rotation @ (tcp_offset_mm / np.linalg.norm(tcp_offset_mm))
    opening_direction = rotation @ np.array([0.0, 0.0, 1.0])
    if (
        not np.isfinite(
            [*tcp_offset_mm, *flange_rpy_deg, *tool_direction, *opening_direction]
        ).all()
        or tool_direction[0] > -math.cos(math.radians(15))
        or abs(opening_direction[1]) < math.cos(math.radians(15))
    ):
        raise ValueError("粗测 TCP 姿态必须朝向 base -X，且沿 base Y 水平张开")

    high_tcp_mm = source_xyz + [
        args.pregrasp_distance_mm,
        0,
        args.hover_clearance_mm,
    ]
    side_tcp_mm = source_xyz + [args.pregrasp_distance_mm, 0, 0]
    stand_off_tcp_mm = source_xyz + [
        args.source_radius_mm + args.final_stand_off_mm,
        0,
        0,
    ]
    tcp_targets = [high_tcp_mm]
    if args.side_approach_only or args.pick_place:
        tcp_targets += [side_tcp_mm, stand_off_tcp_mm]
    if args.pick_place:
        (
            grasp_tcp_mm,
            lifted_tcp_mm,
            destination_above_tcp_mm,
            placement_tcp_mm,
            release_retreat_tcp_mm,
            return_clearance_tcp_mm,
        ) = pick_place_tcp_targets(args, source_xyz, destination_xyz)
        tcp_targets += [
            grasp_tcp_mm,
            lifted_tcp_mm,
            destination_above_tcp_mm,
            placement_tcp_mm,
            release_retreat_tcp_mm,
            return_clearance_tcp_mm,
        ]
    flange_targets = [
        flange_pose_for_tcp(point, tcp_offset_mm, flange_rpy_deg)
        for point in tcp_targets
    ]
    if not all(inside_workspace(point, args.workspace_mm) for point in tcp_targets):
        raise ValueError(f"接近 TCP 超出工作空间：{[p.round(2).tolist() for p in tcp_targets]}")
    if not all(
        inside_workspace(np.asarray(pose[:3]) * 1000, args.workspace_mm)
        for pose in flange_targets
    ):
        raise ValueError("接近法兰目标超出工作空间")

    print(f"\n检测杯子 base XYZ (mm)：{source_xyz.round(2).tolist()}")
    print(f"杯子 +X 高位 TCP (mm)：{high_tcp_mm.round(2).tolist()}")
    print_pose("杯子 +X 高位法兰目标", flange_targets[0])
    if args.side_approach_only or args.pick_place:
        print(f"侧面预抓取 TCP (mm)：{side_tcp_mm.round(2).tolist()}")
        print(f"最终停止 TCP (mm)：{stand_off_tcp_mm.round(2).tolist()}")
    if args.pick_place:
        print(f"抓取中心 TCP (mm)：{grasp_tcp_mm.round(2).tolist()}")
        print(f"放置 TCP (mm)：{placement_tcp_mm.round(2).tolist()}")
        print(f"book 上方 TCP (mm)：{destination_above_tcp_mm.round(2).tolist()}")
        print("\n程序会夹紧、搬运并松开杯子；" + (
            "自动示教模式将连续执行。" if preconfirmed else "到杯面前会再次确认。"
        ))
    else:
        print("\n夹爪始终张开；程序不会夹紧或移动杯子。")
    confirmation = (
        "PICK" if args.pick_place else "APPROACH" if args.side_approach_only else "HOVER"
    )
    if not preconfirmed and not confirm_action(
        "执行视觉定位运动",
        "机械臂将按检测结果运动，请确认目标、扫掠路径、工作空间和急停。",
        confirmation,
    ):
        return

    robot = create_nero_arm(NeroFW.V120)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    recorder = None
    try:
        robot.set_tcp_offset([*(tcp_offset_mm / 1000.0), 0.0, 0.0, 0.0])
        robot.connect()
        time.sleep(1.0)
        if robot.has_comm_error() or not enable_arm(robot):
            raise RuntimeError("机械臂通信或使能失败")
        robot.set_follower_mode()
        if args.act_output:
            from src.arm.prepare import START_JOINTS_DEG

            move_joint_and_wait(
                robot, START_JOINTS_DEG, "ACT 统一预备位姿",
                speed_percent=10, timeout=args.motion_timeout_s, tolerance_deg=1.5,
            )
        initial_joints_deg = None
        if args.pick_place:
            initial_joints_deg = radians_to_degrees(
                wait_joint_angles(robot), digits=3
            )
            initial_pose = get_current_pose(robot)
            print(f"已记录启动关节位置（deg）：{initial_joints_deg}")
            print_pose("启动法兰位姿", initial_pose)
            if initial_pose[1] <= 0:
                print("警告：启动位置不在 base +Y 区域；仍会返回该启动位置。")
        wait_gripper_feedback(gripper)
        gripper.move_gripper_m(
            value=args.gripper_open_mm / 1000.0,
            force=args.gripper_contact_force_n,
        )
        time.sleep(1.5)
        baseline_force_n = float(wait_gripper_feedback(gripper).force)
        print(f"夹爪空载力基线：{baseline_force_n:.3f} N")
        if args.act_output:
            from src.act.record_yolo_episode import YoloEpisodeRecorder
            from src.arm.prepare import START_JOINTS_DEG

            recorder = YoloEpisodeRecorder(
                args.act_output,
                frequency=args.act_frequency,
                action_lead_ms=args.act_action_lead_ms,
                secondary_camera_device=args.act_secondary_camera_device,
                secondary_camera_rotation_degrees=args.act_secondary_camera_rotation_deg,
                metadata={
                    "task_name": "cup_to_blue_teacher",
                    "source_class": args.source_class,
                    "destination_class": args.destination_class,
                    "source_base_xyz_mm": source_xyz.tolist(),
                    "destination_base_xyz_mm": destination_xyz.tolist(),
                    "position_group": args.act_position_group,
                    "fixed_start_joints_deg": START_JOINTS_DEG,
                },
            )
            # Start immediately before the first path command: gripper setup and
            # its 1.5 s idle wait are intentionally excluded from the episode.
            recorder.start(robot, gripper, args.gripper_open_mm / 1000.0)

        compact_entry = False
        try:
            move_pose_and_wait(
                robot,
                "move_p",
                "移动到杯子 +X 侧高位",
                flange_targets[0],
                speed_percent=args.transit_speed_percent,
                planning_retries=1,
                timeout=args.motion_timeout_s,
                position_tolerance=0.004,
            )
        except PoseMotionError as exc:
            if not exc.recoverable or not (args.side_approach_only or args.pick_place):
                raise
            compact_high_tcp_mm = stand_off_tcp_mm + [0, 0, args.hover_clearance_mm]
            compact_high_pose = flange_pose_for_tcp(
                compact_high_tcp_mm, tcp_offset_mm, flange_rpy_deg
            )
            if not inside_workspace(
                compact_high_tcp_mm, args.workspace_mm
            ) or not inside_workspace(
                np.asarray(compact_high_pose[:3]) * 1000,
                args.workspace_mm,
            ):
                raise RuntimeError("自动恢复高位点超出工作空间，拒绝尝试") from exc
            print(
                "\n原 +X 高位不可达，尝试更紧凑的安全入口："
                f"TCP={compact_high_tcp_mm.round(2).tolist()} mm；"
                f"仍与杯面保持 {args.final_stand_off_mm:.1f} mm。"
            )
            move_pose_and_wait(
                robot,
                "move_p",
                "移动到紧凑高位入口",
                compact_high_pose,
                speed_percent=args.transit_speed_percent,
                planning_retries=1,
                timeout=args.motion_timeout_s,
                position_tolerance=0.004,
            )
            compact_entry = True
        if args.side_approach_only or args.pick_place:
            if not compact_entry:
                move_pose_and_wait(
                    robot,
                    "move_l",
                    "在杯子 +X 侧下降到抓取高度",
                    flange_targets[1],
                    speed_percent=args.precision_speed_percent,
                    planning_retries=1,
                    timeout=args.motion_timeout_s,
                    position_tolerance=0.004,
                )
            move_pose_and_wait(
                robot,
                "move_l",
                "从紧凑入口下降到杯面前" if compact_entry else "沿 base -X 接近并在杯面前停止",
                flange_targets[2],
                speed_percent=args.precision_speed_percent,
                planning_retries=1,
                timeout=args.motion_timeout_s,
                position_tolerance=0.004,
            )
        if args.pick_place:
            if not preconfirmed and not confirm_action(
                "进入抓取中心并闭合夹爪",
                "机械臂已到物体前方，请确认夹爪与抓取中心对齐。",
                "GRASP",
            ):
                return False
            (
                grasp_pose,
                lifted_pose,
                destination_above_pose,
                placement_pose,
                retreat_pose,
                return_clearance_pose,
            ) = flange_targets[3:]
            move_pose_and_wait(
                robot,
                "move_l",
                "沿 base -X 进入杯子抓取中心",
                grasp_pose,
                speed_percent=args.precision_speed_percent,
                planning_retries=1,
                timeout=args.motion_timeout_s,
                position_tolerance=0.004,
            )
            if recorder:
                recorder.set_gripper_target(args.gripper_close_mm / 1000.0)
            adaptive_gripper_grasp(gripper, args, baseline_force_n)
            move_pose_and_wait(
                robot,
                "move_l",
                "垂直抬起杯子",
                lifted_pose,
                speed_percent=args.precision_speed_percent,
                planning_retries=1,
                timeout=args.motion_timeout_s,
                position_tolerance=0.004,
            )
            move_pose_and_wait(
                robot,
                "move_p",
                "移动到 book 中心上方",
                destination_above_pose,
                speed_percent=args.transit_speed_percent,
                planning_retries=1,
                timeout=args.motion_timeout_s,
                position_tolerance=0.004,
            )
            move_pose_and_wait(
                robot,
                "move_l",
                "下降并把杯子放到 book 中心",
                placement_pose,
                speed_percent=args.precision_speed_percent,
                planning_retries=1,
                timeout=args.motion_timeout_s,
                position_tolerance=0.004,
            )
            if recorder:
                recorder.set_gripper_target(args.gripper_open_mm / 1000.0)
            gripper.move_gripper_m(
                value=args.gripper_open_mm / 1000.0,
                force=args.gripper_contact_force_n,
            )
            time.sleep(1.5)
            move_pose_and_wait(
                robot,
                "move_l",
                "松爪后沿 base +X 退出",
                retreat_pose,
                speed_percent=args.precision_speed_percent,
                planning_retries=1,
                timeout=args.motion_timeout_s,
                position_tolerance=0.004,
            )
            move_pose_and_wait(
                robot,
                "move_l",
                "退出后垂直抬高",
                return_clearance_pose,
                speed_percent=args.precision_speed_percent,
                planning_retries=1,
                timeout=args.motion_timeout_s,
                position_tolerance=0.004,
            )
            # robot.set_speed_percent(args.transit_speed_percent)
            # move_joint_and_wait(
            #     robot,
            #     initial_joints_deg,
            #     "返回启动位置",
            #     timeout=args.motion_timeout_s,
            #     tolerance_deg=1.5,
            # )
            # print("\n杯子已放到 book 中心，机械臂已返回启动位置。")
        actual_tcp = robot.get_flange2tcp_pose(get_current_pose(robot))
        print_pose("实际到达 TCP", actual_tcp)
        print("\n路径验证完成；请观察夹爪相对杯子中心和杯面的偏差。")
        if recorder:
            recorder.finish(save=True)
            recorder = None
    except KeyboardInterrupt:
        print("\n用户中止，正在发送电子急停。")
        robot.electronic_emergency_stop()
        return False
    except Exception as exc:
        print(f"\n抓取流程安全停止：{exc}")
        try:
            held = hold_current_joint_position(robot, speed_percent=5)
        except Exception as hold_error:
            held = False
            print(f"当前位置保持命令发送失败：{hold_error}")
        hold_result = "机械臂已保持当前位置；" if held else "机械臂未确认保持状态；"
        print(hold_result + "请根据上方运动诊断检查目标、姿态和工作空间。")
        return False
    finally:
        if recorder:
            recorder.finish(save=False)
        robot.disconnect()
    return True


def main(argv=None, *, preconfirmed=False, coordinates_only=False):
    args = parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "缺少 ultralytics；请在 nero 虚拟环境运行：python -m pip install ultralytics"
        ) from exc

    calibration_path = Path("config/camera_to_base.json")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    validate_calibration_quality(calibration)
    base_from_camera = validate_rigid_transform(
        calibration["transform_base_from_camera"]
    )
    model = YOLO(args.weights)
    source_id = destination_id = None
    if args.source_class:
        source_id = class_id(model.names, args.source_class)
        destination_id = class_id(model.names, args.destination_class)
        if source_id == destination_id:
            raise ValueError("source-class 与 destination-class 解析后不能是同一类别")

    camera = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.AUTO
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    status = camera.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"打开 ZED 相机失败：{status!r}")

    left, point_cloud = sl.Mat(), sl.Mat()
    runtime = sl.RuntimeParameters()
    tracked_ids = (source_id, destination_id) if source_id is not None else ()
    histories = {
        wanted_id: deque(maxlen=args.stable_frames) for wanted_id in tracked_ids
    }
    locked = None
    last_report = 0.0
    detection_started = time.monotonic()
    grab_failures = 0
    try:
        print("ZED 已打开，正在取帧并等待 YOLO 目标稳定……", flush=True)
        if not coordinates_only:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            grab_status = camera.grab(runtime)
            if grab_status > sl.ERROR_CODE.SUCCESS:
                grab_failures += 1
                if grab_failures >= 10:
                    raise RuntimeError(
                        f"ZED 连续 {grab_failures} 次取帧失败，最后状态：{grab_status!r}"
                    )
                time.sleep(0.05)
                continue
            grab_failures = 0
            camera.retrieve_image(left, sl.VIEW.LEFT)
            camera.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            image = rotate_image_180(left.get_data())[:, :, :3].copy()
            xyz_image = rotate_image_180(point_cloud.get_data())[:, :, :3]
            predict_options = {
                "conf": args.confidence,
                "imgsz": args.imgsz,
                "verbose": False,
            }
            if source_id is not None:
                predict_options["classes"] = [source_id, destination_id]
            result = model.predict(image, **predict_options)[0]
            detections = []
            best = {}
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                for box, confidence, detected_id in zip(
                    boxes.xyxy, boxes.conf, boxes.cls.astype(int)
                ):
                    confidence = float(confidence)
                    detections.append((box, confidence, detected_id))
                    if detected_id in histories and (
                        detected_id not in best or confidence > best[detected_id][1]
                    ):
                        best[detected_id] = (box, confidence)

            now = time.monotonic()
            if now - last_report >= 2.0:
                summary = ", ".join(
                    f"{model.names[detected_id]}={confidence:.2f}"
                    for _box, confidence, detected_id in detections
                ) or "none"
                print(
                    f"检测：{summary}；推理 {result.speed.get('inference', 0):.0f} ms",
                    flush=True,
                )
                last_report = now

            stable = {}
            coordinates = {}
            for box, confidence, detected_id in detections:
                camera_xyz = median_xyz_in_box(xyz_image, box)
                x1, y1, x2, y2 = map(int, box)
                color = (0, 180, 255)
                label = f"{model.names[detected_id]} {confidence:.2f}"
                if camera_xyz is None:
                    label += " | depth invalid"
                    color = (0, 0, 255)
                else:
                    base_xyz = transform_points([camera_xyz], base_from_camera)[0]
                    if detected_id == source_id:
                        if args.surface_center_correction_mm:
                            base_xyz = cylinder_center_from_surface(
                                base_xyz,
                                camera_xyz,
                                base_from_camera,
                                args.surface_center_correction_mm,
                            )
                        base_xyz = base_xyz + np.asarray(args.source_offset_mm)
                    elif detected_id == destination_id:
                        base_xyz = base_xyz + np.asarray(args.destination_offset_mm)
                    coordinates[id(box)] = base_xyz
                    label += f" | base {base_xyz.round(1).tolist()} mm"
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            for wanted_id, history in histories.items():
                if wanted_id not in best:
                    history.clear()
                    continue
                box, _confidence = best[wanted_id]
                base_xyz = coordinates.get(id(box))
                if base_xyz is None:
                    history.clear()
                    continue
                history.append(base_xyz)
                if len(history) == args.stable_frames:
                    point = stable_point(history, args.max_spread_mm)
                    if point is not None:
                        stable[wanted_id] = point

            ready = (
                source_id is not None
                and source_id in stable
                and destination_id in stable
            )
            if args.auto_lock and ready:
                locked = stable[source_id].copy(), stable[destination_id].copy()
                print("目标已稳定，自动锁定并开始连续示教。")
                break
            if args.auto_lock and time.monotonic() - detection_started >= args.detection_timeout_s:
                raise TimeoutError(
                    f"{args.detection_timeout_s:.1f} s 内未同时获得稳定的 "
                    f"{args.source_class}/{args.destination_class} 坐标"
                )
            if coordinates_only:
                continue
            if source_id is None:
                speed = result.speed.get("inference", 0)
                hint = f"detections: {len(detections)} | inference: {speed:.0f} ms | q: quit"
            else:
                hint = "STABLE - press m" if ready else "waiting for two stable targets"
            cv2.putText(
                image,
                hint,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if ready else (0, 180, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("r"):
                for history in histories.values():
                    history.clear()
            if key == ord("m") and ready:
                locked = stable[source_id].copy(), stable[destination_id].copy()
                break
    except KeyboardInterrupt:
        print("\n用户中止目标检测。")
        return 130
    finally:
        camera.close()
        if not coordinates_only:
            cv2.destroyAllWindows()

    source_xyz, destination_xyz = locked
    print(f"source base XYZ (mm): {source_xyz.round(2).tolist()}")
    print(f"destination base XYZ (mm): {destination_xyz.round(2).tolist()}")
    if coordinates_only:
        return source_xyz, destination_xyz
    if args.hover_only or args.side_approach_only or args.pick_place:
        if execute_side_sequence(
            args, source_xyz, destination_xyz, preconfirmed=preconfirmed
        ) is False:
            return 1
    elif args.execute:
        execute_pick_hover(args, source_xyz, destination_xyz)
    else:
        print("影子模式结束；未使用 --execute，因此未连接机械臂。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
