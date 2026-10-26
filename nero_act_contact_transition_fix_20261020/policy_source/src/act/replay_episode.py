"""Safely dry-run or replay a recorded NERO ACT joint trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.act.episode_io import ACTION_DIM, load_episode
from src.act.robot_wrapper import ARM_DOF, NeroRobotWrapper
from src.act.validate_episode import DEFAULT_MAX_ARM_STEP_RAD, NERO_JOINT_LIMITS_RAD, validate_episode_file
from src.utils.arm import confirm_action


DEFAULT_START_TOLERANCE_RAD = 0.12
DEFAULT_MAX_TRACKING_ERROR_RAD = 0.25
DEFAULT_MOVE_TO_START_SPEED_PERCENT = 10
DEFAULT_MOVE_TO_START_TIMEOUT_S = 45.0
DEFAULT_MOVE_TO_START_TOLERANCE_RAD = 0.02
DEFAULT_REPLAY_SPEED_PERCENT = 10
MAX_FIRST_REPLAY_SPEED = 0.5


def select_range(length: int, start_step: int, end_step: int | None) -> slice:
    end = length if end_step is None else end_step
    if start_step < 0 or end <= start_step or end > length:
        raise ValueError(f"step 范围必须满足 0 <= start < end <= {length}（end 不包含）")
    return slice(start_step, end)


def preflight_actions(actions: np.ndarray, *, max_arm_step_rad: float = DEFAULT_MAX_ARM_STEP_RAD,
                      joint_limit_margin_rad: float = 0.0) -> dict:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM or not len(values):
        raise ValueError(f"action shape 必须为 (T,{ACTION_DIM}) 且非空")
    if not np.isfinite(values).all():
        raise ValueError("action 包含 NaN 或 Inf")
    lower = NERO_JOINT_LIMITS_RAD[:, 0] - joint_limit_margin_rad
    upper = NERO_JOINT_LIMITS_RAD[:, 1] + joint_limit_margin_rad
    if np.any((values[:, :ARM_DOF] < lower) | (values[:, :ARM_DOF] > upper)):
        raise ValueError("action 超出 NERO 标称关节限位")
    max_step = float(np.max(np.abs(np.diff(values[:, :ARM_DOF], axis=0)))) if len(values) > 1 else 0.0
    if max_step > max_arm_step_rad:
        raise ValueError(f"action 单步变化 {max_step:.4f} rad 超过 {max_arm_step_rad:.4f} rad")
    if np.any((values[:, -1] < 0.0) | (values[:, -1] > 0.1)):
        raise ValueError("夹爪目标必须在 0..0.1 m")
    return {"steps": len(values), "max_arm_step_rad": max_step}


def move_to_replay_start(wrapper, target: np.ndarray, *, speed_percent: int,
                         timeout_s: float, tolerance_rad: float,
                         sleep=time.sleep, clock=time.monotonic) -> np.ndarray:
    """Send one low-speed joint target and wait without restarting the planner."""
    wrapper.set_speed_percent(speed_percent)
    wrapper.send_arm_qpos(target)
    deadline = clock() + timeout_s
    current = None
    while clock() < deadline:
        current = wrapper.get_arm_qpos()
        if np.max(np.abs(current - target)) <= tolerance_rad:
            return current
        sleep(0.05)
    raise RuntimeError(
        f"未在 {timeout_s:.1f} s 内到达 replay 起点；"
        f"最大误差={np.max(np.abs(current - target)) if current is not None else float('nan'):.4f} rad"
    )


def replay_actions(wrapper, actions: np.ndarray, timestamps: np.ndarray, *, speed_scale: float,
                   episode_start_step: int = 0, gripper_force_n: float = 1.0,
                   max_tracking_error_rad: float = DEFAULT_MAX_TRACKING_ERROR_RAD,
                   sleep=time.sleep, clock=time.monotonic) -> list[dict]:
    """Execute already validated actions and return per-step tracking rows."""
    rows = []
    start = clock()
    previous_gripper = None
    for local_step, action in enumerate(actions):
        if local_step:
            target_elapsed = (timestamps[local_step] - timestamps[0]) / speed_scale
            delay = start + target_elapsed - clock()
            if delay > 0:
                sleep(delay)
        send_start = clock()
        wrapper.send_arm_qpos(action[:ARM_DOF])
        send_latency = clock() - send_start
        gripper_target = float(action[-1])
        if previous_gripper is None or abs(gripper_target - previous_gripper) > 1e-5:
            wrapper.open_gripper(gripper_target, gripper_force_n)
            previous_gripper = gripper_target
        feedback_start = clock()
        feedback = wrapper.get_arm_qpos()
        feedback_latency = clock() - feedback_start
        error = feedback - action[:ARM_DOF]
        if np.max(np.abs(error)) > max_tracking_error_rad:
            raise RuntimeError(
                f"step {episode_start_step + local_step} 跟踪误差 "
                f"{np.max(np.abs(error)):.4f} rad 超过 {max_tracking_error_rad:.4f} rad"
            )
        rows.append({
            "episode_step": episode_start_step + local_step,
            "elapsed_s": clock() - start,
            "command_action": json.dumps(action.tolist()),
            "feedback_qpos": json.dumps(feedback.tolist()),
            "tracking_error": json.dumps(error.tolist()),
            "max_abs_error_rad": float(np.max(np.abs(error))),
            "send_latency_s": send_latency,
            "feedback_latency_s": feedback_latency,
        })
    return rows


def write_log(rows: list[dict], log_dir: Path, episode: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    output = log_dir / f"act_replay_{episode.stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只做离线 preflight（默认）")
    mode.add_argument("--execute", action="store_true", help="连接并回放真实机械臂")
    parser.add_argument("episode", type=Path)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, help="结束索引，不包含该帧")
    parser.add_argument("--speed-scale", type=float, default=0.25)
    parser.add_argument("--max-arm-step-rad", type=float, default=DEFAULT_MAX_ARM_STEP_RAD)
    parser.add_argument("--joint-limit-margin-rad", type=float, default=0.0)
    parser.add_argument("--start-tolerance-rad", type=float, default=DEFAULT_START_TOLERANCE_RAD)
    parser.add_argument("--move-to-start", action="store_true",
                        help="经二次确认后低速移动到 episode 起点")
    parser.add_argument("--move-only", action="store_true",
                        help="到达 episode 起点后退出，不回放任何 action")
    parser.add_argument("--move-to-start-speed-percent", type=int,
                        default=DEFAULT_MOVE_TO_START_SPEED_PERCENT)
    parser.add_argument("--move-to-start-timeout-s", type=float,
                        default=DEFAULT_MOVE_TO_START_TIMEOUT_S)
    parser.add_argument("--move-to-start-tolerance-rad", type=float,
                        default=DEFAULT_MOVE_TO_START_TOLERANCE_RAD)
    parser.add_argument("--replay-speed-percent", type=int, default=DEFAULT_REPLAY_SPEED_PERCENT,
                        help="控制器速度上限；speed-scale 仍只缩放时间轴")
    parser.add_argument("--max-tracking-error-rad", type=float, default=DEFAULT_MAX_TRACKING_ERROR_RAD)
    parser.add_argument("--gripper-force-n", type=float, default=1.0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    if not args.execute:
        args.dry_run = True
    if (not 0 < args.speed_scale <= 1 or args.max_arm_step_rad <= 0
            or args.start_tolerance_rad <= 0 or args.max_tracking_error_rad <= 0):
        parser.error("speed-scale 必须在 (0,1]，安全阈值必须大于 0")
    if args.joint_limit_margin_rad < 0 or not 0 < args.gripper_force_n <= 3:
        parser.error("joint margin 不得为负，gripper force 必须在 (0,3] N")
    if (not 1 <= args.move_to_start_speed_percent <= 30
            or not 1 <= args.replay_speed_percent <= 30
            or args.move_to_start_timeout_s <= 0
            or args.move_to_start_tolerance_rad <= 0):
        parser.error("起点/replay 速度必须在 1..30%，timeout 和 tolerance 必须大于 0")
    if args.move_only and (not args.execute or not args.move_to_start):
        parser.error("--move-only 必须和 --execute --move-to-start 一起使用")
    return args


def main() -> int:
    args = parse_args()
    validation = validate_episode_file(args.episode, max_arm_step_rad=args.max_arm_step_rad,
                                       joint_limit_margin_rad=args.joint_limit_margin_rad)
    if validation["status"] != "valid":
        print(f"拒绝 replay：{'；'.join(validation['reasons'])}")
        return 1
    episode = load_episode(args.episode)
    try:
        selected = select_range(len(episode["actions"]), args.start_step, args.end_step)
        actions = episode["actions"][selected]
        timestamps = episode["timestamps"][selected]
        metrics = preflight_actions(actions, max_arm_step_rad=args.max_arm_step_rad,
                                    joint_limit_margin_rad=args.joint_limit_margin_rad)
    except ValueError as exc:
        print(f"拒绝 replay：{exc}")
        return 1
    print("\nACT Replay 计划")
    print(f"  episode：{args.episode}")
    print(f"  范围：[{args.start_step}, {args.start_step + len(actions)})，共 {metrics['steps']} 帧")
    print(f"  速度：{args.speed_scale:.2f}x；最大关节步长：{metrics['max_arm_step_rad']:.4f} rad")
    print(f"  模式：{'实机执行' if args.execute else 'dry-run（不连接、不发送）'}")
    if not args.execute:
        print("preflight 通过；未连接机械臂。")
        return 0
    if args.speed_scale > MAX_FIRST_REPLAY_SPEED:
        print("警告：首次真机 replay 建议 speed-scale <= 0.5。")
    action_name = "移动到 ACT 示教起点" if args.move_only else "执行 ACT 轨迹回放"
    warning = ("机械臂将低速移动，但不会回放轨迹或控制夹爪；请确认整段扫掠空间安全。"
               if args.move_only else "机械臂和夹爪将运动；请准备实体急停并清空扫掠区域。")
    if not confirm_action(action_name, warning, "REPLAY"):
        return 0

    wrapper = NeroRobotWrapper()
    try:
        current = wrapper.connect()
        start_error = np.abs(current - actions[0, :ARM_DOF])
        if np.max(start_error) > args.start_tolerance_rad:
            print("\n当前位置（deg）：", [round(math.degrees(float(v)), 2) for v in current])
            print("replay 起点（deg）：", [round(math.degrees(float(v)), 2) for v in actions[0, :ARM_DOF]])
            print("逐关节差异（deg）：", [round(math.degrees(float(v)), 2) for v in start_error])
            if not args.move_to_start:
                raise RuntimeError(
                    f"当前位置与 replay 起点最大差异 {np.max(start_error):.4f} rad，超过 "
                    f"{args.start_tolerance_rad:.4f} rad；确认路径安全后可显式加 --move-to-start"
                )
            if not confirm_action(
                "低速移动到 replay 起点",
                "这是直接关节运动，不含碰撞检测；请确认当前位置到上述目标的整段扫掠空间安全。",
                "MOVE_START",
            ):
                return 0
            move_to_replay_start(
                wrapper,
                actions[0, :ARM_DOF],
                speed_percent=args.move_to_start_speed_percent,
                timeout_s=args.move_to_start_timeout_s,
                tolerance_rad=args.move_to_start_tolerance_rad,
            )
            print("已到达 replay 起点。")
        if args.move_only:
            print("已位于 ACT 示教起点；按要求不回放 action、不控制夹爪。")
            return 0
        wrapper.set_speed_percent(args.replay_speed_percent)
        rows = replay_actions(wrapper, actions, timestamps, speed_scale=args.speed_scale,
                              episode_start_step=args.start_step, gripper_force_n=args.gripper_force_n,
                              max_tracking_error_rad=args.max_tracking_error_rad)
        log = write_log(rows, args.log_dir, args.episode)
        errors = [row["max_abs_error_rad"] for row in rows]
        print(f"replay 完成：最大跟踪误差 mean/max={np.mean(errors):.4f}/{np.max(errors):.4f} rad")
        print(f"日志：{log}")
        return 0
    except KeyboardInterrupt:
        print("\nreplay 被中断，正在电子急停。")
        if wrapper.connected:
            wrapper.emergency_stop()
        return 130
    except Exception as exc:
        print(f"replay 失败：{exc}")
        if wrapper.connected:
            wrapper.emergency_stop()
        return 1
    finally:
        wrapper.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
