"""Safely characterize NERO ``move_j`` for continuous ACT joint targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.act.robot_wrapper import ARM_DOF, NeroRobotWrapper
from src.utils.arm import confirm_action


DEFAULT_FREQUENCY_HZ = 5.0
DEFAULT_DURATION_S = 5.0
DEFAULT_AMPLITUDE_RAD = 0.02
DEFAULT_RETURN_TIMEOUT_S = 10.0
RETURN_TOLERANCE_RAD = 0.01
MAX_FREQUENCY_HZ = 20.0
MAX_DURATION_S = 30.0
MAX_AMPLITUDE_RAD = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印测试轨迹（默认行为）",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="连接并向机械臂发送小幅单关节目标",
    )
    parser.add_argument(
        "--read-hardware",
        action="store_true",
        help="dry-run 时经确认后连接、使能并读取真实 qpos，但不发送目标",
    )
    parser.add_argument("--joint-index", type=int, default=0)
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--amplitude-rad", type=float, default=DEFAULT_AMPLITUDE_RAD)
    parser.add_argument("--speed-percent", type=int,
                        help="可选控制器速度上限；省略时保持 SDK 当前/默认设置")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    if not 0 <= args.joint_index < ARM_DOF:
        parser.error(f"joint-index 必须在 0..{ARM_DOF - 1} 范围内")
    if not 0 < args.frequency <= MAX_FREQUENCY_HZ:
        parser.error(f"frequency 必须在 (0, {MAX_FREQUENCY_HZ}] Hz 范围内")
    if not 0 < args.duration <= MAX_DURATION_S:
        parser.error(f"duration 必须在 (0, {MAX_DURATION_S}] s 范围内")
    if not 0 < args.amplitude_rad <= MAX_AMPLITUDE_RAD:
        parser.error(f"amplitude-rad 必须在 (0, {MAX_AMPLITUDE_RAD}] rad 范围内")
    if args.read_hardware and args.execute:
        parser.error("--read-hardware 仅用于 dry-run，不能与 --execute 同时使用")
    if args.speed_percent is not None and not 1 <= args.speed_percent <= 100:
        parser.error("speed-percent 必须在 1..100 范围内")
    return args


def print_plan(args: argparse.Namespace, q0: np.ndarray | None) -> None:
    print("\nACT Joint Stream 测试计划")
    print(f"  模式：{'实机执行' if args.execute else 'dry-run（不发送目标）'}")
    print(f"  关节索引：{args.joint_index}（J{args.joint_index + 1}）")
    print(f"  请求频率：{args.frequency:.1f} Hz")
    print(f"  持续时间：{args.duration:.1f} s")
    print(f"  最大偏移：±{args.amplitude_rad:.4f} rad")
    print(f"  控制器速度：{args.speed_percent if args.speed_percent is not None else '保持默认'}")
    if q0 is None:
        print("  qpos：未连接硬件；预期 shape=(7,), dtype=float32, unit=rad")
        center = np.zeros(ARM_DOF, dtype=np.float32)
    else:
        print(f"  qpos：{q0.tolist()}")
        print(f"  qpos 元数据：shape={q0.shape}, dtype={q0.dtype}, unit=rad")
        center = q0
    samples = []
    for elapsed in np.linspace(0.0, args.duration, 5):
        target = center.copy()
        target[args.joint_index] += args.amplitude_rad * math.sin(
            2.0 * math.pi * elapsed / args.duration
        )
        samples.append(round(float(target[args.joint_index]), 6))
    print(f"  J{args.joint_index + 1} 轨迹预览：{samples}")


def percentile95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def write_log(rows: list[dict], args: argparse.Namespace) -> Path:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    path = args.log_dir / f"act_joint_stream_{datetime.now():%Y%m%d_%H%M%S}.csv"
    fields = [
        "requested_frequency_hz",
        "duration_s",
        "joint_index",
        "amplitude_rad",
        "step",
        "timestamp_s",
        "command_q_target_rad",
        "feedback_qpos_rad",
        "move_j_latency_s",
        "feedback_latency_s",
        "tested_joint_error_rad",
        "max_abs_error_rad",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_summary(rows: list[dict], loop_starts: list[float], args) -> None:
    loop_dt = [b - a for a, b in zip(loop_starts, loop_starts[1:])]
    move_latency = [row["move_j_latency_s"] for row in rows]
    feedback_latency = [row["feedback_latency_s"] for row in rows]
    tested_joint_errors = [
        abs(row["tested_joint_error_rad"])
        for row in rows
        if row["tested_joint_error_rad"] != ""
    ]
    none_count = sum(not row["feedback_qpos_rad"] for row in rows)
    errors = [row["error"] for row in rows if row["error"]]
    actual_hz = 1.0 / statistics.mean(loop_dt) if loop_dt else 0.0
    print("\n测试统计")
    print(f"  requested frequency：{args.frequency:.2f} Hz")
    print(f"  actual loop frequency：{actual_hz:.2f} Hz")
    print(
        "  loop dt mean/P95/max："
        f"{statistics.mean(loop_dt) if loop_dt else 0.0:.6f} / "
        f"{percentile95(loop_dt):.6f} / {max(loop_dt, default=0.0):.6f} s"
    )
    print(
        "  move_j latency mean/max："
        f"{statistics.mean(move_latency):.6f} / {max(move_latency):.6f} s"
    )
    print(
        "  feedback latency mean/max："
        f"{statistics.mean(feedback_latency):.6f} / {max(feedback_latency):.6f} s"
    )
    print(
        "  tested joint |error| mean/P95/max："
        f"{statistics.mean(tested_joint_errors) if tested_joint_errors else 0.0:.6f} / "
        f"{percentile95(tested_joint_errors):.6f} / "
        f"{max(tested_joint_errors, default=0.0):.6f} rad"
    )
    print(f"  None feedback：{none_count}")
    print(f"  errors：{errors or 'none'}")


def return_to_start(
    wrapper: NeroRobotWrapper,
    q0: np.ndarray,
    timeout_s: float = DEFAULT_RETURN_TIMEOUT_S,
) -> None:
    """Return the tested joint to q0 and verify it before disconnecting."""
    wrapper.send_arm_qpos(q0)
    deadline = time.monotonic() + timeout_s
    current = None
    while time.monotonic() < deadline:
        current = wrapper.get_arm_qpos()
        if np.max(np.abs(current - q0)) <= RETURN_TOLERANCE_RAD:
            return
        time.sleep(0.05)
    errors = current - q0 if current is not None else np.full_like(q0, np.nan)
    raise RuntimeError(
        f"测试结束后未能在 {timeout_s:.1f} s 内回到起点；"
        f"当前 qpos={current.tolist() if current is not None else None}；"
        f"逐关节误差 rad={errors.tolist()}"
    )


def execute_stream(wrapper: NeroRobotWrapper, q0: np.ndarray, args) -> Path:
    period = 1.0 / args.frequency
    rows: list[dict] = []
    loop_starts: list[float] = []
    start = time.monotonic()
    step = 0
    while True:
        loop_start = time.monotonic()
        elapsed = loop_start - start
        if elapsed >= args.duration:
            break
        loop_starts.append(loop_start)
        target = q0.copy()
        target[args.joint_index] += args.amplitude_rad * math.sin(
            2.0 * math.pi * elapsed / args.duration
        )
        row = {
            "requested_frequency_hz": args.frequency,
            "duration_s": args.duration,
            "joint_index": args.joint_index,
            "amplitude_rad": args.amplitude_rad,
            "step": step,
            "timestamp_s": elapsed,
            "command_q_target_rad": json.dumps(target.tolist()),
            "feedback_qpos_rad": "",
            "move_j_latency_s": 0.0,
            "feedback_latency_s": 0.0,
            "tested_joint_error_rad": "",
            "max_abs_error_rad": "",
            "error": "",
        }
        command_start = time.monotonic()
        wrapper.send_arm_qpos(target)
        row["move_j_latency_s"] = time.monotonic() - command_start
        feedback_start = time.monotonic()
        try:
            feedback = wrapper.get_arm_qpos()
            row["feedback_qpos_rad"] = json.dumps(feedback.tolist())
            errors = feedback - target
            row["tested_joint_error_rad"] = float(errors[args.joint_index])
            row["max_abs_error_rad"] = float(np.max(np.abs(errors)))
        except TimeoutError as exc:
            row["error"] = str(exc)
        row["feedback_latency_s"] = time.monotonic() - feedback_start
        rows.append(row)
        if row["error"]:
            log_path = write_log(rows, args)
            print_summary(rows, loop_starts, args)
            raise RuntimeError(f"关节反馈中断，已停止测试；日志：{log_path}")
        step += 1
        remaining = start + step * period - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    log_path = write_log(rows, args)
    print_summary(rows, loop_starts, args)
    print(f"  CSV 日志：{log_path}")
    print("\n正在单次发送起始 qpos，并等待机械臂回位……")
    return_to_start(wrapper, q0)
    return log_path


def main() -> int:
    args = parse_args()
    if not args.execute and not args.read_hardware:
        print_plan(args, None)
        return 0
    wrapper = NeroRobotWrapper()
    q0 = None
    try:
        if args.read_hardware:
            if not confirm_action(
                "读取 ACT 初始关节状态",
                "机械臂将连接、使能并进入 Follower，但不会发送运动目标。请清空工作区。",
                "CONNECT",
            ):
                print_plan(args, None)
                return 0
            q0 = wrapper.connect()

        print_plan(args, q0)
        if not args.execute:
            return 0
        if args.frequency > DEFAULT_FREQUENCY_HZ:
            print("警告：应先用相同姿态和幅度成功完成 5 Hz 测试。")
        if not confirm_action(
            "执行 ACT 单关节流式测试",
            "机械臂将使能并小幅运动；请准备急停并确保扫掠范围无人和障碍物。",
            "STREAM",
        ):
            return 0
        q0 = wrapper.connect()
        if args.speed_percent is not None:
            wrapper.set_speed_percent(args.speed_percent)
        execute_stream(wrapper, q0, args)
        return 0
    except KeyboardInterrupt:
        print("\n测试被中断，正在触发电子急停。")
        if wrapper.connected:
            wrapper.emergency_stop()
        return 130
    except Exception as exc:
        print(f"\n测试失败：{exc}")
        if wrapper.connected:
            wrapper.emergency_stop()
        return 1
    finally:
        try:
            wrapper.disconnect()
        except Exception as exc:
            print(f"断开 CAN 时发生异常：{exc}")


if __name__ == "__main__":
    raise SystemExit(main())
