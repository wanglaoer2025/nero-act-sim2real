"""NERO 机械臂共用的连接、转换、状态检查和运动辅助函数。"""

import math
import time

_DEFAULT_FIRMWARE = object()


def confirm_action(action, warning, token):
    """Require an exact action token; empty or mismatched input cancels."""
    print(f"\n操作：{action}")
    print(f"风险：{warning}")
    try:
        answer = input(f"输入 {token} 继续，直接回车取消：").strip()
    except EOFError:
        answer = ""

    if answer != token:
        print("确认未通过，本阶段操作已取消。")
        return False
    return True


_RECOVERABLE_PLANNING_STATUSES = {0x02, 0x03, 0x04}
BENT_PREPARE_JOINTS_DEG = [5.73, 30, 5.73, 70, 5.73, 5.73, 30]


class PoseMotionError(RuntimeError):
    """Cartesian motion failure with enough state for a safe caller fallback."""

    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def recoverable(self):
        return self.status_code in _RECOVERABLE_PLANNING_STATUSES


def create_nero_arm(firmware=_DEFAULT_FIRMWARE, channel="can0", interface="socketcan"):
    """Create a NERO arm instance without connecting to the hardware."""
    # Offline ACT data tools must remain usable without the hardware SDK.
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

    if firmware is _DEFAULT_FIRMWARE:
        firmware = NeroFW.V120

    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=firmware,  # The SDK intentionally uses this spelling.
        interface=interface,
        channel=channel,
    )
    return AgxArmFactory.create_arm(config)


def degrees_to_radians(values):
    """Convert an iterable of angles from degrees to radians."""
    return [math.radians(value) for value in values]


def radians_to_degrees(values, digits=2):
    """Convert radians to rounded degrees for display."""
    return [round(math.degrees(value), digits) for value in values]


def pose_mm_deg_to_si(pose):
    """Convert ``[mm, mm, mm, deg, deg, deg]`` to SDK SI units."""
    x, y, z, roll, pitch, yaw = pose
    return [
        x / 1000.0,
        y / 1000.0,
        z / 1000.0,
        math.radians(roll),
        math.radians(pitch),
        math.radians(yaw),
    ]


def angle_error(current, target):
    """Return the smallest absolute difference between two angles."""
    return abs((current - target + math.pi) % (2.0 * math.pi) - math.pi)


def offset_pose(pose, dx=0.0, dy=0.0, dz=0.0):
    """Copy a flange pose and apply a Cartesian offset in metres."""
    target = list(pose)
    target[0] += dx
    target[1] += dy
    target[2] += dz
    return target


def enable_arm(robot, timeout=5.0, settle_duration=0.3):
    """Enable all joints and wait until controller feedback is motion-ready."""
    deadline = time.monotonic() + timeout
    enabled = False
    ready_since = None
    while time.monotonic() < deadline:
        if not enabled:
            enabled = bool(robot.enable())
        if enabled:
            status = robot.get_arm_status()
            joints = robot.get_joint_angles()
            ready = (
                status is not None
                and joints is not None
                and status.msg.arm_status
                in {0x00, *_RECOVERABLE_PLANNING_STATUSES}
            )
            if ready:
                if ready_since is None:
                    ready_since = time.monotonic()
                if time.monotonic() - ready_since >= settle_duration:
                    return True
            else:
                ready_since = None
        time.sleep(0.05)
    return False


def get_current_pose(robot):
    """Read the flange pose and fail clearly when feedback is absent."""
    feedback = robot.get_flange_pose()
    if feedback is None:
        raise RuntimeError("没有收到机械臂末端位姿反馈")
    return list(feedback.msg)


def wait_stable_leader_pose(
    robot,
    stable_duration=0.4,
    timeout=3.0,
    joint_tolerance=math.radians(0.3),
):
    """等待 Leader 关节停止变化，并用 FK 返回当前法兰位姿和反馈。"""
    deadline = time.monotonic() + timeout
    stable_since = None
    reference = None
    while time.monotonic() < deadline:
        feedback = robot.get_leader_joint_angles()
        if feedback is None:
            time.sleep(0.02)
            continue
        joints = list(feedback.msg)
        if reference is None or max(
            angle_error(current, start) for current, start in zip(joints, reference)
        ) > joint_tolerance:
            reference = joints
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= stable_duration:
            return robot.fk(joints), feedback
        time.sleep(0.02)
    raise RuntimeError("机械臂仍在移动或没有收到 Leader 关节反馈，本次未采样")


def wait_fresh_flange_pose(robot, previous_timestamp, timeout=0.5):
    """等待时间戳更新的控制器法兰反馈，避免使用缓存。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        feedback = robot.get_flange_pose()
        if feedback is not None and feedback.timestamp > previous_timestamp:
            return list(feedback.msg), feedback
        time.sleep(0.02)
    raise RuntimeError("没有收到新的控制器法兰反馈")


def flange_pose_difference(first, second):
    """返回两个法兰位姿之间的位置距离和最大姿态角差，单位为米和弧度。"""
    position = math.sqrt(sum((first[i] - second[i]) ** 2 for i in range(3)))
    angle = max(angle_error(first[i], second[i]) for i in range(3, 6))
    return position, angle


def wait_stable_flange_pose(
    robot,
    previous_timestamp=0,
    stable_duration=0.3,
    timeout=3.0,
    position_tolerance=0.0005,
    angle_tolerance=math.radians(0.1),
):
    """等待 Follower 中的新法兰反馈稳定并返回位姿及反馈。"""
    deadline = time.monotonic() + timeout
    stable_since = None
    reference = None
    latest_timestamp = previous_timestamp
    while time.monotonic() < deadline:
        feedback = robot.get_flange_pose()
        if feedback is None or feedback.timestamp <= latest_timestamp:
            time.sleep(0.02)
            continue
        latest_timestamp = feedback.timestamp
        pose = list(feedback.msg)
        if reference is None:
            reference = pose
            stable_since = time.monotonic()
        else:
            position_error, angle_error_rad = flange_pose_difference(reference, pose)
            if (
                position_error > position_tolerance
                or angle_error_rad > angle_tolerance
            ):
                reference = pose
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= stable_duration:
                return pose, feedback
        time.sleep(0.02)
    raise RuntimeError("Follower 法兰位姿仍在变化或没有新反馈，本次未采样")


def check_arm_normal(robot):
    """Return the current status, raising if feedback or arm state is abnormal."""
    status = robot.get_arm_status()
    if status is None:
        raise RuntimeError("没有收到机械臂状态反馈")
    if status.msg.arm_status != 0:
        raise RuntimeError(f"机械臂状态异常：\n{status.msg}")
    return status.msg


def wait_joint_angles(robot, timeout=3.0):
    """Wait for and return one joint-angle feedback message."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        feedback = robot.get_joint_angles()
        if feedback is not None:
            return list(feedback.msg)
        time.sleep(0.02)
    raise RuntimeError("没有收到关节角反馈")


def wait_joint_reached(robot, target, timeout=30.0, tolerance_deg=1.5):
    """Return whether every joint reaches the target within the tolerance."""
    deadline = time.monotonic() + timeout
    tolerance = math.radians(tolerance_deg)
    while time.monotonic() < deadline:
        feedback = robot.get_joint_angles()
        if feedback is not None:
            errors = [abs(now - wanted) for now, wanted in zip(feedback.msg, target)]
            if errors and max(errors) <= tolerance:
                return True
        time.sleep(0.05)
    return False


def wait_pose_reached(
    robot,
    target,
    timeout=30.0,
    position_tolerance=0.004,
    angle_tolerance=math.radians(3.0),
    failed_motion_grace=1.0,
):
    """Return whether the flange reaches a Cartesian target before timeout."""
    started = time.monotonic()
    deadline = started + timeout
    last_progress = started
    best_position_error = math.inf
    best_angle_error = math.inf
    while time.monotonic() < deadline:
        status = check_arm_normal(robot)
        current = get_current_pose(robot)
        position_errors = [abs(current[i] - target[i]) for i in range(3)]
        angle_errors = [angle_error(current[i], target[i]) for i in range(3, 6)]
        position_error = max(position_errors)
        orientation_error = max(angle_errors)
        if position_error <= position_tolerance and orientation_error <= angle_tolerance:
            return True
        now = time.monotonic()
        if (
            position_error < best_position_error - 0.0005
            or orientation_error < best_angle_error - math.radians(0.25)
        ):
            best_position_error = min(best_position_error, position_error)
            best_angle_error = min(best_angle_error, orientation_error)
            last_progress = now
        if (
            status.motion_status == 1
            and now - started >= failed_motion_grace
            and now - last_progress >= failed_motion_grace
        ):
            raise RuntimeError("控制器报告未能到达目标位置")
        time.sleep(0.05)
    return False


def controller_pose_target(target, current):
    """Use the equivalent ±pi representation nearest the current wrist pose."""
    result = list(target)
    for index in (3, 5):
        if math.isclose(abs(result[index]), math.pi, abs_tol=1e-7):
            result[index] = math.copysign(math.pi, current[index])
    return result


def print_pose_motion_debug(robot, name, method, target, error=None):
    """Print actionable feedback after one failed Cartesian motion."""
    print(f"\n运动诊断：{name}；方法={method}")
    print_pose("控制器目标位姿", target)
    try:
        current = get_current_pose(robot)
        print_pose("失败后实际位姿", current)
        position_errors = [abs(current[i] - target[i]) * 1000 for i in range(3)]
        angle_errors = [
            math.degrees(angle_error(current[i], target[i])) for i in range(3, 6)
        ]
        print(
            "位置误差 XYZ (mm)："
            f"{[round(value, 2) for value in position_errors]}；"
            "姿态误差 RPY (deg)："
            f"{[round(value, 2) for value in angle_errors]}"
        )
    except Exception as feedback_error:
        print(f"实际位姿：读取失败（{feedback_error}）")

    joints = robot.get_joint_angles()
    if joints is not None:
        print(f"当前关节角（deg）：{radians_to_degrees(joints.msg, digits=3)}")
    get_ik = getattr(robot, "get_ik_joint_angles", None)
    ik = None if get_ik is None else get_ik()
    if ik is not None:
        print(f"控制器 IK 候选（deg）：{radians_to_degrees(ik.msg, digits=3)}")
    status = robot.get_arm_status()
    print(f"控制器状态：\n{status.msg}" if status is not None else "控制器状态：无反馈")
    if error is not None:
        print(f"底层原因：{error}")
    return None if status is None else status.msg.arm_status


def hold_current_joint_position(robot, *, speed_percent=5, timeout=2.0):
    """Clear a recoverable planning state by commanding the measured joints."""
    status = robot.get_arm_status()
    status_code = None if status is None else status.msg.arm_status
    if status_code not in {0x00, *_RECOVERABLE_PLANNING_STATUSES}:
        print("当前为不可恢复控制器错误，不自动发送保持命令。")
        return False
    joints = robot.get_joint_angles()
    if joints is None:
        print("没有关节反馈，无法安全保持当前位置。")
        return False
    set_motion_speed(robot, speed_percent)
    robot.move_j(list(joints.msg))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None and status.msg.arm_status == 0x00:
            print("已保持当前关节位置并清除可恢复的规划状态。")
            return True
        time.sleep(0.05)
    print("已发送当前位置保持命令，但控制器状态未在等待时间内恢复 NORMAL。")
    return False


def wait_motion_done(robot, timeout=30.0, start_delay=0.5):
    """Wait for controller motion status; raise immediately on arm errors."""
    time.sleep(start_delay)  # Avoid accepting the previous motion's completed state.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None:
            if status.msg.arm_status != 0:
                raise RuntimeError(f"机械臂状态异常：{status.msg.arm_status}")
            if status.msg.motion_status == 0:
                return True
            if status.msg.motion_status == 1:
                raise RuntimeError("机械臂未能到达目标位置")
        time.sleep(0.1)
    return False


def set_motion_speed(robot, speed_percent):
    """Set and validate the speed used by the immediately following motion."""
    if isinstance(speed_percent, bool) or not isinstance(speed_percent, int):
        raise ValueError("speed_percent 必须是整数")
    if not 1 <= speed_percent <= 100:
        raise ValueError("speed_percent 必须在 1..100 范围内")
    robot.set_speed_percent(speed_percent)


def move_pose_and_wait(
    robot, method, name, target, *, speed_percent, planning_retries=0, **wait_options
):
    """Send a Cartesian target, retry recoverable planning failures, and wait."""
    if planning_retries < 0:
        raise ValueError("planning_retries 不能小于 0")
    current = get_current_pose(robot)
    targets = target if method == "move_c" else [target]
    if method == "move_c" and len(targets) != 3:
        raise ValueError("move_c 必须提供起点、中点、终点三个笛卡尔位姿")
    controller_targets = []
    reference = current
    for pose in targets:
        adjusted = controller_pose_target(pose, reference)
        controller_targets.append(adjusted)
        reference = adjusted
    controller_target = controller_targets[-1]
    if any(adjusted != list(pose) for adjusted, pose in zip(controller_targets, targets)):
        print("姿态边界修正：使用与当前位置连续的等价 ±180° 表示。")

    for attempt in range(planning_retries + 1):
        suffix = "" if attempt == 0 else f"，规划重试 {attempt}/{planning_retries}"
        print(f"\n开始执行：{name}（速度 {speed_percent}%{suffix}）")
        set_motion_speed(robot, speed_percent)
        if method == "move_c":
            # NERO V120 needs a real non-circular target to clear its MOVE_C buffer.
            robot.move_l(list(current))
            mode_deadline = time.monotonic() + 1.0
            while time.monotonic() < mode_deadline:
                if check_arm_normal(robot).mode_feedback == 0x02:
                    break
                time.sleep(0.02)
            else:
                raise RuntimeError("控制器未能在圆弧前接受零位移 MOVE_L 指令")
            robot.move_c(*controller_targets)
        else:
            getattr(robot, method)(controller_target)
        time.sleep(0.3)  # Give the controller time to accept the new target.
        error = None
        try:
            reached = wait_pose_reached(robot, controller_target, **wait_options)
        except RuntimeError as exc:
            reached = False
            error = exc
        if reached:
            print(f"完成：{name}")
            return

        status_code = print_pose_motion_debug(
            robot, name, method, controller_target, error
        )
        recoverable_failure = (
            status_code in _RECOVERABLE_PLANNING_STATUSES
            or (status_code == 0x00 and error is not None)
        )
        if (
            attempt < planning_retries
            and recoverable_failure
            and hold_current_joint_position(robot, speed_percent=min(speed_percent, 10))
        ):
            continue
        reason = (
            "控制器报告可恢复的路径规划失败"
            if recoverable_failure
            else "运动超时或控制器状态异常"
        )
        raise PoseMotionError(
            f"{name}失败：{reason}；已输出目标、反馈、误差、关节角和控制器状态",
            status_code=status_code,
        )


def move_joint_and_wait(
    robot, target_deg, name, *, speed_percent, **wait_options
):
    """Set speed, send a joint target in degrees, and wait for arrival."""
    status = robot.get_arm_status()
    if status is None:
        raise RuntimeError("没有收到机械臂状态反馈")
    # 规划失败会保留到下一条命令；新的关节命令可以覆盖这些历史状态。
    if status.msg.arm_status not in {0x00, *_RECOVERABLE_PLANNING_STATUSES}:
        raise RuntimeError(f"机械臂状态异常，拒绝发送关节运动：\n{status.msg}")
    if status.msg.arm_status in _RECOVERABLE_PLANNING_STATUSES:
        print(f"检测到上一次运动遗留状态，将用本次 move_j 覆盖：{status.msg.arm_status}")

    target = degrees_to_radians(target_deg)
    print(f"{name}目标角度：{target_deg}（速度 {speed_percent}%）")
    set_motion_speed(robot, speed_percent)
    robot.move_j(target)
    if not wait_joint_reached(robot, target, **wait_options):
        current = robot.get_joint_angles()
        if current is not None:
            print_joint_angles("当前关节角（度）：", current.msg)
        status = robot.get_arm_status()
        if status is not None:
            print(f"控制器状态：\n{status.msg}")
        raise RuntimeError(f"{name}未在规定时间内到达目标位置")
    feedback = robot.get_joint_angles()
    if feedback is not None:
        print(f"{name}完成，当前角度：", radians_to_degrees(feedback.msg))


def move_to_prepare_pose(
    robot, target_deg, *, speed_percent, timeout=30.0, tolerance_deg=1.5
):
    """Move to a bent joint pose before starting Cartesian motion."""
    print("\n准备进入非奇异关节姿态")
    move_joint_and_wait(
        robot,
        target_deg,
        "准备姿态",
        speed_percent=speed_percent,
        timeout=timeout,
        tolerance_deg=tolerance_deg,
    )
    print("已进入非奇异准备姿态")
    time.sleep(1.0)


def execute_linear_path(
    robot,
    offsets,
    *,
    speed_percent,
    pause=0.5,
    timeout=30.0,
    position_tolerance=0.003,
    angle_tolerance=math.radians(2.0),
):
    """Preview, confirm, and execute Cartesian offsets from the current pose."""
    start = get_current_pose(robot)
    targets = [offset_pose(start, *offset) for offset in offsets]
    print_pose("直线路径起始位姿", start)
    print("\n即将执行以下直线路径：")
    for index, target in enumerate(targets, 1):
        print_pose(f"路径点{index}", target)
    if not confirm_action(
        "执行直线路径",
        "机械臂将依次经过以上路径点，请确认整段扫掠空间安全。",
        "PATH",
    ):
        return

    for index, target in enumerate(targets, 1):
        move_pose_and_wait(
            robot,
            "move_l",
            f"第{index}段直线",
            target,
            speed_percent=speed_percent,
            timeout=timeout,
            position_tolerance=position_tolerance,
            angle_tolerance=angle_tolerance,
        )
        print_pose(f"第{index}段实际到达位姿", get_current_pose(robot))
        print(f"第{index}段直线运动完成")
        time.sleep(pause)
    print("\n全部直线运动执行完成")


def wait_gripper_feedback(gripper, timeout=3.0):
    """Wait for and return a gripper status message."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = gripper.get_gripper_status()
        if status is not None:
            print("夹爪状态：")
            print(status.msg)
            return status.msg
        time.sleep(0.05)
    raise RuntimeError("没有收到夹爪状态反馈")


def wait_gripper_grasp(
    gripper,
    target_width_m,
    baseline_force_n,
    timeout=3.0,
    stable_duration=0.3,
    min_width_gap_m=0.003,
    min_force_delta_n=0.08,
    width_tolerance_m=0.001,
    force_tolerance_n=0.05,
):
    """等待夹爪稳定，并用宽度余量和力变化共同判断是否抓住物体。

    AGX 夹爪没有单独的“接触”状态位，因此不能只看力反馈：夹爪空抓到
    机械限位时也可能出现力。实际宽度没有到达闭合目标，且相对空载基线的
    力变化足够大时，才认为物体阻挡了夹爪。
    """
    deadline = time.monotonic() + timeout
    stable_since = None
    reference = None
    latest = None
    peak_force_delta = 0.0

    while time.monotonic() < deadline:
        feedback = gripper.get_gripper_status()
        if feedback is None:
            time.sleep(0.05)
            continue

        latest = feedback.msg
        foc = latest.foc_status
        faults = [
            name
            for name in (
                "voltage_too_low",
                "motor_overheating",
                "driver_overcurrent",
                "driver_overheating",
                "driver_error_status",
            )
            if getattr(foc, name, False)
        ]
        if faults:
            raise RuntimeError(f"夹爪驱动异常：{', '.join(faults)}")

        width = float(latest.value)
        force = float(latest.force)
        peak_force_delta = max(peak_force_delta, abs(force - baseline_force_n))
        if reference is None or (
            abs(width - reference[0]) > width_tolerance_m
            or abs(force - reference[1]) > force_tolerance_n
        ):
            reference = (width, force)
            stable_since = time.monotonic()

        if time.monotonic() - stable_since >= stable_duration:
            width_gap = width - target_width_m
            if width_gap < min_width_gap_m:
                raise RuntimeError(
                    "夹爪已接近闭合目标，未检测到物体；请检查物体位置或减小目标宽度"
                )
            if peak_force_delta < min_force_delta_n:
                raise RuntimeError(
                    "夹爪宽度受到阻挡，但力变化不足；请检查反馈并重新设定阈值"
                )
            print(
                "抓握确认："
                f"实际宽度={width * 1000:.1f} mm，"
                f"当前力={force:.3f} N，最大力变化={peak_force_delta:.3f} N"
            )
            return latest
        time.sleep(0.05)

    if latest is None:
        raise RuntimeError("没有收到夹爪状态反馈")
    raise RuntimeError("夹爪未在规定时间内稳定，无法确认抓握状态")


def ramp_gripper_force(
    gripper,
    target_width_m,
    start_force_n,
    target_force_n,
    force_step_n=0.1,
    step_delay_s=0.2,
):
    """保持闭合宽度目标，并把夹爪设定力从接触力逐步升到保持力。"""
    if not 0 < start_force_n <= target_force_n <= 3:
        raise ValueError("夹爪力必须满足 0 < start_force_n <= target_force_n <= 3")
    if force_step_n <= 0 or step_delay_s < 0:
        raise ValueError("力步长必须大于 0，步间隔不能小于 0")

    requested_force = start_force_n
    latest = None
    while requested_force < target_force_n:
        requested_force = min(requested_force + force_step_n, target_force_n)
        gripper.move_gripper_m(value=target_width_m, force=requested_force)
        time.sleep(step_delay_s)
        feedback = gripper.get_gripper_status()
        if feedback is not None:
            latest = feedback.msg
            foc = latest.foc_status
            if any(
                getattr(foc, name, False)
                for name in (
                    "voltage_too_low",
                    "motor_overheating",
                    "driver_overcurrent",
                    "driver_overheating",
                    "driver_error_status",
                )
            ):
                raise RuntimeError(f"夹爪驱动异常：\n{latest}")
            print(
                f"设定力={requested_force:.2f} N，"
                f"反馈力={float(latest.force):.3f} N，"
                f"宽度={float(latest.value) * 1000:.1f} mm"
            )

    return latest if latest is not None else wait_gripper_feedback(gripper)


def adaptive_gripper_grasp(gripper, args, baseline_force_n):
    """低力闭合至物体，确认接触后逐步增加到目标保持力。"""
    target_width_m = args.gripper_close_mm / 1000.0
    gripper.move_gripper_m(
        value=target_width_m,
        force=args.gripper_contact_force_n,
    )
    contact = wait_gripper_grasp(
        gripper,
        target_width_m=target_width_m,
        baseline_force_n=baseline_force_n,
        timeout=args.gripper_grasp_timeout_s,
        min_width_gap_m=args.gripper_min_width_gap_mm / 1000.0,
        min_force_delta_n=args.gripper_min_force_delta_n,
    )
    print(
        f"低力接触确认：宽度={float(contact.value) * 1000:.1f} mm，"
        f"反馈力={float(contact.force):.3f} N"
    )
    held = ramp_gripper_force(
        gripper,
        target_width_m=target_width_m,
        start_force_n=args.gripper_contact_force_n,
        target_force_n=args.gripper_force_n,
        force_step_n=args.gripper_force_step_n,
        step_delay_s=args.gripper_force_step_delay_s,
    )
    print(
        f"保持力已建立：宽度={float(held.value) * 1000:.1f} mm，"
        f"反馈力={float(held.force):.3f} N"
    )
    return held


def print_joint_angles(name, joints):
    """Print joint angles in degrees."""
    print(name, radians_to_degrees(joints))


def print_pose(name, pose):
    """Print an SDK flange pose using millimetres and degrees."""
    print(f"\n{name}：")
    print(
        f"  X={pose[0] * 1000:.2f} mm, Y={pose[1] * 1000:.2f} mm, "
        f"Z={pose[2] * 1000:.2f} mm"
    )
    print(
        f"  Roll={math.degrees(pose[3]):.2f}°, "
        f"Pitch={math.degrees(pose[4]):.2f}°, Yaw={math.degrees(pose[5]):.2f}°"
    )


def print_arm_status(robot, title):
    """Print and return the arm status message, or ``None`` without feedback."""
    status = robot.get_arm_status()
    if status is None:
        print(f"{title}：没有收到状态反馈")
        return None
    print(f"\n{title}：")
    print(status.msg)
    return status.msg
