"""Run the trained NERO ACT policy with real camera/feedback and a safety gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.act.adaptive_mpc import AdaptiveMPC
from src.act.nero_dataset import denormalize_actions, preprocess_images
from src.act.nero_policy import NeroActPolicy
from src.act.realtime_logger import RealtimeLogger
from src.act.robot_wrapper import NeroRobotWrapper
from src.act.safety_filter import SafetyFilter
from src.act.temporal_ensemble import TemporalEnsembler
from src.utils.arm import confirm_action
from src.utils.vision import pose_to_transform


COMPATIBILITY_KEYS = (
    "state_dim", "action_dim", "chunk_size", "camera_names", "image_size",
    "hidden_dim", "dim_feedforward", "nheads", "enc_layers", "dec_layers", "dropout", "backbone",
)


def load_runtime(config_path: Path, device_override: str | None = None):
    """Load and strictly validate Day 6 config, checkpoint, and normalization stats."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    requested = device_override or config.get("device", "auto")
    device_name = "cuda" if requested == "auto" and torch.cuda.is_available() else (
        "cpu" if requested == "auto" else requested
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但 torch.cuda.is_available() 为 false")
    device = torch.device(device_name)
    checkpoint = Path(config["checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get('hardware_deployment_allowed') is False:
        raise ValueError('该权重包仅通过离线/仿真核验，请使用 run_act_shadow；尚未验收实机执行方式及场景对齐')
    model_config = payload.get("config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint 缺少训练 config")
    for key in COMPATIBILITY_KEYS:
        expected = tuple(config[key]) if isinstance(config[key], list) else config[key]
        actual = tuple(model_config.get(key, ())) if isinstance(expected, tuple) else model_config.get(key)
        if actual != expected:
            raise ValueError(f"checkpoint {key}={actual!r} 与 Day6 config {expected!r} 不匹配")

    with Path(config["dataset_stats"]).open("rb") as stream:
        stats = pickle.load(stream)
    dimensions = {
        "qpos_mean": int(config["state_dim"]),
        "qpos_std": int(config["state_dim"]),
        "action_mean": int(config["action_dim"]),
        "action_std": int(config["action_dim"]),
    }
    for key, dimension in dimensions.items():
        values = np.asarray(stats.get(key), dtype=np.float32)
        if values.shape != (dimension,) or not np.isfinite(values).all():
            raise ValueError(f"normalization {key} 必须是有限 float32[{dimension}]")
        if key.endswith("std") and np.any(values <= 0):
            raise ValueError(f"normalization {key} 必须大于 0")
        stats[key] = values

    runtime_model_config = dict(model_config, device=device_name, pretrained_backbone=False)
    policy = NeroActPolicy(runtime_model_config).to(device)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()
    if device.type == "cuda":
        with torch.inference_mode():
            policy(
                torch.zeros((1, int(config["state_dim"])), device=device),
                torch.zeros((1, len(config["camera_names"]), 3, *config["image_size"]), device=device),
            )
        torch.cuda.synchronize(device)
    return config, stats, policy, device


def _full_state(robot) -> np.ndarray:
    return np.concatenate([robot.get_arm_qpos(), robot.get_gripper_state()]).astype(np.float32)


def resolve_task_coordinates(
    config: dict,
    source_override_mm: list[float] | None = None,
    destination_override_mm: list[float] | None = None,
) -> np.ndarray | None:
    """Return source/destination task coordinates in metres for a 14-D policy."""
    if not config.get("coordinate_conditioning", False):
        return None
    if (source_override_mm is None) != (destination_override_mm is None):
        raise ValueError("source/destination 坐标覆盖必须同时提供")
    if source_override_mm is not None:
        points = np.asarray([source_override_mm, destination_override_mm], dtype=np.float32)
    else:
        task = config["task_coordinates"]
        from src.demo.yolo_pick_place import main as run_yolo

        located = run_yolo([
            "--weights", str(task["weights"]),
            "--source-class", str(task["source_class"]),
            "--destination-class", str(task["destination_class"]),
            "--auto-lock",
            "--detection-timeout-s", str(task.get("detection_timeout_s", 30.0)),
            "--stable-frames", str(task.get("stable_frames", 8)),
            "--max-spread-mm", str(task.get("max_spread_mm", 5.0)),
            "--confidence", str(task.get("confidence", 0.25)),
            "--imgsz", str(task.get("imgsz", 640)),
            "--source-offset-mm", *map(str, task.get("source_offset_mm", [0, 0, 0])),
            "--destination-offset-mm", *map(str, task.get("destination_offset_mm", [0, 0, 0])),
            "--surface-center-correction-mm",
            str(task.get("surface_center_correction_mm", 1.0)),
        ], coordinates_only=True)
        if not isinstance(located, tuple) or len(located) != 2:
            raise RuntimeError("YOLO 定位未返回 source/destination 坐标")
        points = np.asarray(located, dtype=np.float32)
    if points.shape != (2, 3) or not np.isfinite(points).all():
        raise ValueError("任务坐标必须是有限的 source/destination base XYZ")
    return points.reshape(6) / 1000.0


def rollout_position_diagnostics(
    rows: list[dict], task_coordinates, fk, tcp_offset_m, task_config: dict,
) -> dict:
    """Compare detected targets with estimated TCP positions at gripper transitions."""
    detected = np.asarray(task_coordinates, dtype=float).reshape(2, 3) * 1000.0
    grasp_target = detected[0] + [0.0, 0.0, float(task_config.get("grasp_z_offset_mm", 0.0))]
    place_target = np.array([
        detected[1, 0], detected[1, 1],
        grasp_target[2] + float(task_config.get("placement_gap_mm", 5.0)),
    ])
    offset = np.asarray(tcp_offset_m, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("TCP offset 必须是有限 3 维米坐标")

    samples = []
    for row in rows:
        tcp_xyz = estimate_tcp_xyz_mm(row["actual_qpos"], fk, offset)
        samples.append((row, tcp_xyz))

    def closest(target):
        if not samples:
            return None
        row, tcp_xyz = min(samples, key=lambda item: np.linalg.norm(item[1] - target))
        delta = tcp_xyz - target
        return {
            "step": int(row["step"]),
            "estimated_tcp_xyz_mm": tcp_xyz.tolist(),
            "offset_xyz_mm": delta.tolist(),
            "distance_mm": float(np.linalg.norm(delta)),
            "xy_error_mm": float(np.linalg.norm(delta[:2])),
        }

    events = []
    was_closed = False
    for row, tcp_xyz in samples:
        if not row.get("command_sent"):
            continue
        is_closed = float(row["gripper_command_m"]) < 0.0475
        event = "grasp" if is_closed and not was_closed else (
            "place" if not is_closed and was_closed else None
        )
        was_closed = is_closed
        if event is None:
            continue
        target = grasp_target if event == "grasp" else place_target
        delta = tcp_xyz - target
        events.append({
            "event": event,
            "step": int(row["step"]),
            "timestamp": float(row["timestamp"]),
            "estimated_tcp_xyz_mm": tcp_xyz.tolist(),
            "expected_tcp_xyz_mm": target.tolist(),
            "offset_xyz_mm": delta.tolist(),
            "xy_error_mm": float(np.linalg.norm(delta[:2])),
        })
    return {
        "detected_source_base_xyz_mm": detected[0].tolist(),
        "detected_destination_base_xyz_mm": detected[1].tolist(),
        "expected_grasp_tcp_xyz_mm": grasp_target.tolist(),
        "expected_place_tcp_xyz_mm": place_target.tolist(),
        "tcp_offset_flange_xyz_m": offset.tolist(),
        "tcp_offset_approximate": True,
        "final_estimated_tcp_xyz_mm": samples[-1][1].tolist() if samples else None,
        "closest_to_expected_grasp": closest(grasp_target),
        "closest_to_expected_place": closest(place_target),
        "gripper_events": events,
    }


def estimate_tcp_xyz_mm(qpos, fk, tcp_offset_m) -> np.ndarray:
    """Estimate TCP XYZ in base millimetres from joint feedback and flange FK."""
    offset = np.asarray(tcp_offset_m, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("TCP offset 必须是有限 3 维米坐标")
    flange_pose = np.asarray(fk(np.asarray(qpos, dtype=float).tolist()), dtype=float)
    if flange_pose.shape != (6,) or not np.isfinite(flange_pose).all():
        raise ValueError("FK 法兰位姿必须是有限 6 维坐标")
    transform = pose_to_transform(flange_pose)
    return (transform[:3, 3] + transform[:3, :3] @ offset) * 1000.0


def gate_gripper_close(target, current_qpos, previous_gripper_command,
                       grasp_target_mm, tolerance_mm, fk, tcp_offset_m,
                       gripper_open_m):
    """Block only a new close transition while TCP is outside the grasp sphere."""
    safe = np.asarray(target, dtype=np.float32).copy()
    grasp_target = np.asarray(grasp_target_mm, dtype=float)
    tolerance = float(tolerance_mm)
    if safe.shape != (8,) or grasp_target.shape != (3,) or tolerance <= 0:
        raise ValueError("抓取门禁要求 target[8]、grasp target[3] 和正容差")
    if safe[7] >= 0.0475 or previous_gripper_command < 0.0475:
        return safe, None, False
    distance = float(np.linalg.norm(
        estimate_tcp_xyz_mm(current_qpos, fk, tcp_offset_m) - grasp_target
    ))
    blocked = distance > tolerance
    if blocked:
        safe[7] = float(gripper_open_m)
    return safe, distance, blocked


def print_position_diagnostics(report: dict) -> None:
    print("\nACT 目标/夹爪位置诊断（TCP 为手工偏移估算）")
    print(f"  检测杯子 base XYZ (mm)：{np.round(report['detected_source_base_xyz_mm'], 2).tolist()}")
    print(f"  检测蓝卡 base XYZ (mm)：{np.round(report['detected_destination_base_xyz_mm'], 2).tolist()}")
    print(f"  期望抓取 TCP (mm)：{np.round(report['expected_grasp_tcp_xyz_mm'], 2).tolist()}")
    print(f"  期望放置 TCP (mm)：{np.round(report['expected_place_tcp_xyz_mm'], 2).tolist()}")
    if report["final_estimated_tcp_xyz_mm"] is not None:
        print(f"  rollout 最终估算 TCP (mm)：{np.round(report['final_estimated_tcp_xyz_mm'], 2).tolist()}")
    for key, label in (
        ("closest_to_expected_grasp", "最接近期望抓取点"),
        ("closest_to_expected_place", "最接近期望放置点"),
    ):
        closest = report[key]
        if closest is not None:
            print(
                f"  {label} step={closest['step']}：估算 TCP "
                f"{np.round(closest['estimated_tcp_xyz_mm'], 2).tolist()} mm；"
                f"偏差 {np.round(closest['offset_xyz_mm'], 2).tolist()} mm；"
                f"距离 {closest['distance_mm']:.2f} mm"
            )
    if not report["gripper_events"]:
        print("  本次 rollout 未触发夹爪抓取/放置命令。")
    for event in report["gripper_events"]:
        label = "抓取" if event["event"] == "grasp" else "放置"
        print(
            f"  {label} step={event['step']}：估算 TCP "
            f"{np.round(event['estimated_tcp_xyz_mm'], 2).tolist()} mm；"
            f"偏差 {np.round(event['offset_xyz_mm'], 2).tolist()} mm；"
            f"XY 误差 {event['xy_error_mm']:.2f} mm"
        )


def _jitter(actions: list[np.ndarray]) -> dict[str, float]:
    if len(actions) < 2:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    values = np.linalg.norm(np.diff(np.asarray(actions)[:, :7], axis=0), axis=1)
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def next_control_deadline(previous_deadline: float, now: float, period: float) -> float:
    """Advance one period without issuing burst commands after an overrun."""
    scheduled = previous_deadline + period
    return scheduled if scheduled > now else now + period


def command_speed_scale(speed_scale: float, command_stride: int) -> float:
    """Preserve requested velocity when commands are decimated."""
    return min(1.0, speed_scale * command_stride)


def save_model_input_snapshot(
    image: torch.Tensor, camera_names: list[str], output_dir: Path, step: int,
) -> list[dict]:
    """Save the exact resized RGB tensors passed into ACT plus viewable PNGs."""
    values = image.detach().cpu().numpy()
    if values.shape[:3] != (1, len(camera_names), 3):
        raise ValueError(f"ACT 输入图像 shape 异常：{values.shape}")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for camera_index, camera_name in enumerate(camera_names):
        chw = np.ascontiguousarray(values[0, camera_index], dtype=np.float32)
        stem = output_dir / f"step_{step:04d}_{camera_name}"
        np.save(stem.with_suffix(".npy"), chw)
        rgb = np.round(chw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        if not cv2.imwrite(str(stem.with_suffix(".png")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"无法保存 ACT 输入图像：{stem.with_suffix('.png')}")
        record = {
            "step": step, "camera": camera_name, "shape": list(chw.shape),
            "mean": float(chw.mean()), "std": float(chw.std()),
            "sha256": hashlib.sha256(chw.tobytes()).hexdigest(),
            "tensor_path": str(stem.with_suffix(".npy")),
            "preview_path": str(stem.with_suffix(".png")),
        }
        snapshots.append(record)
        print(
            f"ACT input step={step} {camera_name}: mean={record['mean']:.6f} "
            f"std={record['std']:.6f} sha256={record['sha256'][:12]}"
        )
    return snapshots


def run_control_loop(
    policy,
    camera,
    robot,
    stats: dict,
    config: dict,
    logger: RealtimeLogger,
    *,
    execute: bool,
    max_steps: int,
    speed_scale: float,
    temporal_enabled: bool,
    adaptive_mpc_enabled: bool = False,
    task_coordinates: np.ndarray | None = None,
    tcp_offset_m: np.ndarray | None = None,
    max_commands: int | None = None,
    control_gripper: bool = True,
    model_input_steps: tuple[int, ...] = (),
    clock=time.monotonic,
    sleep=time.sleep,
) -> dict:
    """Run a bounded loop. Dry-run executes the identical path but sends nothing."""
    safety_summary = json.loads(Path(config["safety_summary"]).read_text(encoding="utf-8"))
    safety = SafetyFilter(
        config["safety"], safety_summary["train_action_min"], safety_summary["train_action_max"],
        safety_summary["train_qpos_min"], safety_summary["train_qpos_max"],
    )
    current = _full_state(robot)
    safety.validate_initial_state(current, float(config["initial_state_margin_rad"]))
    task_coordinates = None if task_coordinates is None else np.asarray(task_coordinates, dtype=np.float32)
    expected_task_dim = int(config.get("state_dim", len(current))) - len(current)
    if expected_task_dim:
        if task_coordinates is None or task_coordinates.shape != (expected_task_dim,):
            raise ValueError(f"ACT state 需要 {expected_task_dim} 维任务坐标")
        if not np.isfinite(task_coordinates).all():
            raise ValueError("ACT 任务坐标必须是有限数值")
    task_config = config.get("task_coordinates", {})
    destination_mm = (
        None if task_coordinates is None
        else task_coordinates[3:6].astype(float) * 1000.0
    )
    table_surface_z_mm = None if destination_mm is None else float(destination_mm[2])
    grasp_tolerance_mm = task_config.get("grasp_gate_tolerance_mm")
    abort_on_grasp_gate_block = task_config.get("abort_on_grasp_gate_block", False)
    if not isinstance(abort_on_grasp_gate_block, bool):
        raise ValueError("abort_on_grasp_gate_block 必须是布尔值")
    grasp_target_mm = None
    if grasp_tolerance_mm is not None:
        grasp_tolerance_mm = float(grasp_tolerance_mm)
        if grasp_tolerance_mm <= 0:
            raise ValueError("grasp_gate_tolerance_mm 必须大于 0")
        if task_coordinates is None or tcp_offset_m is None:
            raise ValueError("启用抓取门禁时必须提供任务坐标和 TCP offset")
        grasp_target_mm = task_coordinates[:3].astype(float) * 1000.0
        grasp_target_mm[2] += float(task_config.get("grasp_z_offset_mm", 0.0))
    if abort_on_grasp_gate_block and grasp_target_mm is None:
        raise ValueError("启用提前闭爪终止时必须同时启用抓取门禁")
    mpc = None
    place_target_mm = None
    if adaptive_mpc_enabled:
        if grasp_target_mm is None or tcp_offset_m is None or task_coordinates is None:
            raise ValueError("启用 adaptive MPC 时必须提供任务坐标、抓取目标和 TCP offset")
        mpc = AdaptiveMPC(
            config.get("adaptive_mpc", {}), config["safety"]["joint_limits_rad"]
        )
        place_target_mm = np.array([
            destination_mm[0], destination_mm[1],
            grasp_target_mm[2] + float(task_config.get("placement_gap_mm", 5.0)),
        ])
    task_progress_window_s = float(task_config.get("progress_watchdog_window_s", 0.0))
    task_progress_start_distance_mm = float(
        task_config.get("progress_watchdog_start_distance_mm", 0.0)
    )
    task_progress_min_improvement_mm = float(
        task_config.get("progress_watchdog_min_improvement_mm", 0.0)
    )
    if task_progress_window_s and (
        grasp_target_mm is None
        or min(task_progress_window_s, task_progress_start_distance_mm,
               task_progress_min_improvement_mm) <= 0
    ):
        raise ValueError("抓取进展 watchdog 参数必须全部大于 0")
    min_grasp_z_offset_mm = task_config.get("grasp_height_gate_min_z_offset_mm")
    min_grasp_table_clearance_mm = task_config.get(
        "grasp_height_gate_min_table_clearance_mm"
    )
    if min_grasp_z_offset_mm is not None and min_grasp_table_clearance_mm is not None:
        raise ValueError("抓取高度门禁只能配置 Z offset 或桌面净空其中一种")
    if min_grasp_z_offset_mm is not None:
        min_grasp_z_offset_mm = float(min_grasp_z_offset_mm)
        if (
            grasp_target_mm is None
            or task_progress_start_distance_mm <= 0
            or not np.isfinite(min_grasp_z_offset_mm)
            or min_grasp_z_offset_mm >= 0
        ):
            raise ValueError(
                "抓取高度门禁需要抓取点、正的 progress 起始距离和负的 Z offset 阈值"
            )
    if min_grasp_table_clearance_mm is not None:
        min_grasp_table_clearance_mm = float(min_grasp_table_clearance_mm)
        if (
            grasp_target_mm is None
            or table_surface_z_mm is None
            or task_progress_start_distance_mm <= 0
            or not np.isfinite(min_grasp_table_clearance_mm)
            or min_grasp_table_clearance_mm <= 0
        ):
            raise ValueError(
                "桌面净空门禁需要抓取点、目标平面、正的 progress 起始距离和正净空"
            )
    ensemble = TemporalEnsembler(
        int(config["chunk_size"]), int(config["action_dim"]),
        float(config["temporal_ensemble"]["decay"]),
    )
    device = next(policy.parameters()).device
    control_frequency = float(config["control_frequency_hz"])
    command_frequency = float(config.get("command_frequency_hz", control_frequency))
    if control_frequency <= 0 or command_frequency <= 0 or command_frequency > control_frequency:
        raise ValueError("command_frequency_hz 必须在 (0, control_frequency_hz] 范围内")
    command_stride = round(control_frequency / command_frequency)
    if command_stride < 1 or not np.isclose(
        command_stride * command_frequency, control_frequency
    ):
        raise ValueError("command_frequency_hz 必须能整除 control_frequency_hz")
    period = 1.0 / control_frequency
    command_tolerance = float(config.get("command_reached_tolerance_rad", 0.005))
    command_timeout = float(config.get("command_timeout_s", 5.0))
    progress_window = float(config.get("progress_window_s", 1.0))
    min_progress_request = float(config.get("no_progress_min_request_rad", 0.015))
    min_feedback_progress = float(config.get("no_progress_min_feedback_rad", 0.001))
    if min(command_tolerance, command_timeout, progress_window,
           min_progress_request, min_feedback_progress) <= 0:
        raise ValueError("command/progress 参数必须大于 0")
    first_actions: list[np.ndarray] = []
    aggregate_actions: list[np.ndarray] = []
    final_actions: list[np.ndarray] = []
    actual_qposes: list[np.ndarray] = []
    mpc_fallbacks = 0
    input_snapshots: list[dict] = []
    input_steps = set(model_input_steps)
    grasp_gate_blocks = 0
    grasp_gate_distances: list[float] = []
    grasp_gate_was_blocked = False
    grasp_height_gate_blocks = 0
    grasp_z_offsets: list[float] = []
    task_progress_started = None
    task_progress_baseline_mm = None
    task_progress_best_mm = None
    grasp_completed = False
    commands_sent = 0
    previous_safe = None
    last_sent_target = None
    progress_started = clock()
    progress_origin = current[:7].copy()
    max_requested_progress = 0.0
    last_gripper_command = config["safety"]["gripper_open_m"] if current[7] >= 0.0475 else 0.0
    if execute:
        robot.set_speed_percent(int(config["execution_speed_percent"]))
    next_deadline = previous_loop = clock()
    send_speed_scale = command_speed_scale(speed_scale, command_stride)

    for step in range(max_steps):
        remaining = next_deadline - clock()
        if remaining > 0:
            sleep(remaining)
        loop_started = clock()

        stamp = clock()
        frames = camera.capture_rgb()
        camera_latency = clock() - stamp
        stamp = clock()
        current = _full_state(robot)
        qpos_latency = clock() - stamp
        if mpc is not None:
            mpc.observe(current[:7])
        if execute and clock() - progress_started >= progress_window:
            feedback_progress = float(np.max(np.abs(current[:7] - progress_origin)))
            if (max_requested_progress >= min_progress_request
                    and feedback_progress < min_feedback_progress):
                raise RuntimeError(
                    f"运动无进展：{progress_window:.2f} s 窗口内已请求 "
                    f"{max_requested_progress:.4f} rad，关节反馈仅变化 "
                    f"{feedback_progress:.4f} rad"
                )
            progress_started = clock()
            progress_origin = current[:7].copy()
            max_requested_progress = 0.0

        if previous_safe is not None:
            tracking_error = float(np.max(np.abs(current[:7] - previous_safe[:7])))
            if tracking_error > float(config["max_tracking_error_rad"]):
                raise RuntimeError(
                    f"关节跟踪误差 {tracking_error:.4f} rad 超过 "
                    f"{config['max_tracking_error_rad']:.4f} rad"
                )

        stamp = clock()
        camera_names = config.get("camera_names", ["camera_0"])
        if isinstance(frames, dict):
            missing = [name for name in camera_names if name not in frames]
            if missing:
                raise RuntimeError(f"实时相机缺少配置视角：{missing}")
            image = torch.cat([
                preprocess_images(frames[name][None], config["image_size"])
                for name in camera_names
            ])[None].to(device)
        else:
            if len(camera_names) != 1:
                raise RuntimeError("多相机模型需要 camera.capture_rgb() 返回按名称索引的图像")
            image = preprocess_images(frames[None], config["image_size"])[None].to(device)
        if step in input_steps:
            input_snapshots.extend(save_model_input_snapshot(
                image, camera_names,
                logger.csv_path.parent / f"{logger.csv_path.stem}_model_inputs", step,
            ))
        model_state = current if task_coordinates is None else np.concatenate([current, task_coordinates])
        normalized_state = torch.from_numpy(
            (model_state - stats["qpos_mean"]) / stats["qpos_std"]
        )[None].to(device)
        preprocess_latency = clock() - stamp
        stamp = clock()
        with torch.inference_mode():
            prediction = policy(normalized_state, image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_latency = clock() - stamp
        if tuple(prediction.shape) != (1, int(config["chunk_size"]), int(config["action_dim"])):
            raise ValueError(f"ACT 输出 shape 异常：{tuple(prediction.shape)}")
        pred_action_raw = prediction[0, 0].detach().cpu().numpy()
        stamp = clock()
        raw_chunk = denormalize_actions(prediction[0], stats).cpu().numpy()
        denorm_latency = clock() - stamp
        first_action = raw_chunk[0].copy()
        ensemble.add(step, raw_chunk)
        aggregated = ensemble.current(step)
        selected = aggregated if temporal_enabled else first_action
        first_actions.append(first_action)
        aggregate_actions.append(aggregated)

        stamp = clock()
        mpc_info = {
            "active": False, "blend": 0.0, "correction_norm_rad": 0.0,
            "fallback": "", "solve_latency_s": 0.0,
            "tcp_distance_mm": None, "predicted_terminal_error_mm": None,
            "response_alpha": [], "target_xyz_mm": [],
        }
        if mpc is not None:
            references = raw_chunk[:, :7].copy()
            references[0] = selected[:7]
            selected = selected.copy()
            selected[:7], mpc_info = mpc.plan(
                current[:7], references,
                place_target_mm if grasp_completed else grasp_target_mm,
                lambda qpos: estimate_tcp_xyz_mm(
                    qpos, robot.robot.fk, tcp_offset_m
                ),
                previous_command=None if previous_safe is None else previous_safe[:7],
            )
            mpc_fallbacks += int(bool(mpc_info["fallback"]))
        safe_target, info = safety.filter(
            selected, current, speed_scale=send_speed_scale,
            command_reference=previous_safe if execute else None,
        )
        grasp_tcp_distance_mm = None
        grasp_gate_blocked = False
        premature_grasp_error = None
        if grasp_target_mm is not None:
            safe_target, grasp_tcp_distance_mm, grasp_gate_blocked = gate_gripper_close(
                safe_target, current[:7], last_gripper_command,
                grasp_target_mm, grasp_tolerance_mm, robot.robot.fk,
                tcp_offset_m, config["safety"]["gripper_open_m"],
            )
            if grasp_tcp_distance_mm is not None:
                grasp_gate_distances.append(grasp_tcp_distance_mm)
            if grasp_gate_blocked:
                grasp_gate_blocks += 1
                info["gripper_command_m"] = float(safe_target[7])
                if execute and abort_on_grasp_gate_block:
                    premature_grasp_error = (
                        f"抓取时序提前：TCP 距抓取点 {grasp_tcp_distance_mm:.1f} mm，"
                        f"超过 {grasp_tolerance_mm:.1f} mm；已停止发送，禁止进入抬升阶段"
                    )
                if not grasp_gate_was_blocked:
                    print(
                        f"抓取门禁：TCP 距抓取点 {grasp_tcp_distance_mm:.1f} mm > "
                        f"{grasp_tolerance_mm:.1f} mm，禁止闭爪。"
                    )
            elif grasp_tcp_distance_mm is not None and grasp_gate_was_blocked:
                print(
                    f"抓取门禁：TCP 距抓取点 {grasp_tcp_distance_mm:.1f} mm，"
                    "允许闭爪。"
                )
            grasp_gate_was_blocked = grasp_gate_blocked
        estimated_tcp_xyz_mm = None
        grasp_distance_mm = None
        grasp_tcp_z_offset_mm = None
        grasp_table_clearance_mm = None
        grasp_height_gate_blocked = False
        grasp_height_error = None
        if grasp_target_mm is not None:
            estimated_tcp_xyz_mm = estimate_tcp_xyz_mm(
                current[:7], robot.robot.fk, tcp_offset_m
            )
            grasp_distance_mm = float(np.linalg.norm(
                estimated_tcp_xyz_mm - grasp_target_mm
            ))
            grasp_tcp_z_offset_mm = float(
                estimated_tcp_xyz_mm[2] - grasp_target_mm[2]
            )
            if table_surface_z_mm is not None:
                grasp_table_clearance_mm = float(
                    estimated_tcp_xyz_mm[2] - table_surface_z_mm
                )
            grasp_z_offsets.append(grasp_tcp_z_offset_mm)
            height_below_limit = (
                min_grasp_table_clearance_mm is not None
                and grasp_table_clearance_mm < min_grasp_table_clearance_mm
            ) or (
                min_grasp_z_offset_mm is not None
                and grasp_tcp_z_offset_mm < min_grasp_z_offset_mm
            )
            grasp_height_gate_blocked = bool(
                height_below_limit
                and not grasp_completed
                and last_gripper_command >= 0.0475
                and grasp_distance_mm <= task_progress_start_distance_mm
            )
            if grasp_height_gate_blocked:
                grasp_height_gate_blocks += 1
                if execute:
                    if min_grasp_table_clearance_mm is not None:
                        grasp_height_error = (
                            f"TCP 抓取高度过低：距抓取点 {grasp_distance_mm:.1f} mm 时，"
                            f"距桌面 {grasp_table_clearance_mm:.1f} mm，低于 "
                            f"{min_grasp_table_clearance_mm:.1f} mm；已停止发送"
                        )
                    else:
                        grasp_height_error = (
                            f"TCP 抓取高度过低：距抓取点 {grasp_distance_mm:.1f} mm 时 "
                            f"Z 偏差 {grasp_tcp_z_offset_mm:.1f} mm，低于 "
                            f"{min_grasp_z_offset_mm:.1f} mm；已停止发送"
                        )
        if (
            execute and task_progress_window_s and not grasp_completed
            and safe_target[7] >= 0.0475 and grasp_height_error is None
        ):
            now = clock()
            if task_progress_started is None:
                if grasp_distance_mm <= task_progress_start_distance_mm:
                    task_progress_started = now
                    task_progress_baseline_mm = grasp_distance_mm
                    task_progress_best_mm = grasp_distance_mm
            else:
                task_progress_best_mm = min(task_progress_best_mm, grasp_distance_mm)
                if now - task_progress_started >= task_progress_window_s:
                    improvement = task_progress_baseline_mm - task_progress_best_mm
                    if improvement < task_progress_min_improvement_mm:
                        raise RuntimeError(
                            f"TCP 抓取进展停滞：{task_progress_window_s:.1f} s 内"
                            f"最佳距离仅改善 {improvement:.1f} mm，小于 "
                            f"{task_progress_min_improvement_mm:.1f} mm"
                        )
                    task_progress_started = now
                    task_progress_baseline_mm = task_progress_best_mm
        safety_latency = clock() - stamp
        stamp = clock()
        command_sent = (
            execute and step % command_stride == 0 and
            (max_commands is None or commands_sent < max_commands) and
            premature_grasp_error is None and grasp_height_error is None
        )
        if command_sent:
            robot.send_arm_qpos(safe_target[:7])
            if mpc is not None:
                mpc.record_command(current[:7], safe_target[:7])
            commands_sent += 1
            last_sent_target = safe_target[:7].copy()
            max_requested_progress = max(
                max_requested_progress,
                float(np.max(np.abs(last_sent_target - progress_origin))),
            )
            if control_gripper and safe_target[7] != last_gripper_command:
                if safety.gripper_mode == "continuous_width":
                    robot.send_gripper_width(float(safe_target[7]),float(config["gripper_force_n"]))
                    if safe_target[7] <= safety.gripper_close_threshold:
                        grasp_completed = True
                elif safe_target[7] > 0:
                    robot.open_gripper(safe_target[7], float(config["gripper_force_n"]))
                else:
                    robot.close_gripper(0.0, float(config["gripper_force_n"]))
                    grasp_completed = True
                last_gripper_command = float(safe_target[7])
            previous_safe = safe_target.copy()
        elif not execute:
            previous_safe = safe_target.copy()
        send_latency = clock() - stamp
        actual = _full_state(robot) if command_sent else current.copy()
        final_actions.append(safe_target.copy())
        actual_qposes.append(actual[:7].copy())
        processing_latency = clock() - loop_started
        logger.log({
            "timestamp": time.time(), "step": step,
            "current_qpos": current[:7],
            "pred_action_raw": pred_action_raw,
            "pred_action_denorm": first_action,
            "final_action_command": safe_target,
            "actual_qpos": actual[:7],
            "gripper_prediction": float(first_action[7]),
            "gripper_state": float(actual[7]),
            "qpos": current,
            "raw_pred": raw_chunk, "first_action": first_action,
            "aggregated_pred": aggregated, "safe_target": safe_target,
            "mpc_target": mpc_info["target_xyz_mm"],
            "mpc_active": int(mpc_info["active"]),
            "mpc_blend": mpc_info["blend"],
            "mpc_correction_norm_rad": mpc_info["correction_norm_rad"],
            "mpc_fallback": mpc_info["fallback"],
            "mpc_solve_latency_s": mpc_info["solve_latency_s"],
            "mpc_tcp_distance_mm": mpc_info["tcp_distance_mm"],
            "mpc_predicted_terminal_error_mm": mpc_info["predicted_terminal_error_mm"],
            "mpc_response_alpha": mpc_info["response_alpha"],
            "joint_limit_clamps": info["joint_limit_clamps"],
            "delta_clamps": info["delta_clamps"],
            "command_lead_clamps": info["command_lead_clamps"],
            "range_anomalies": info["range_anomalies"],
            "gripper_command_m": info["gripper_command_m"],
            "grasp_gate_blocked": int(grasp_gate_blocked),
            "grasp_tcp_distance_mm": grasp_tcp_distance_mm,
            "estimated_tcp_xyz_mm": estimated_tcp_xyz_mm,
            "grasp_distance_mm": grasp_distance_mm,
            "grasp_tcp_z_offset_mm": grasp_tcp_z_offset_mm,
            "grasp_table_clearance_mm": grasp_table_clearance_mm,
            "grasp_height_gate_blocked": int(grasp_height_gate_blocked),
            "command_sent": int(command_sent),
            "camera_latency_s": camera_latency, "qpos_latency_s": qpos_latency,
            "preprocess_latency_s": preprocess_latency, "model_latency_s": model_latency,
            "denorm_latency_s": denorm_latency, "safety_latency_s": safety_latency,
            "send_latency_s": send_latency,
            "total_loop_dt_s": loop_started - previous_loop if step else 0.0,
            "processing_latency_s": processing_latency,
            "late_cycle": int(processing_latency > period),
        })
        stop_error = grasp_height_error or premature_grasp_error
        if stop_error is not None:
            raise RuntimeError(stop_error)
        previous_loop = loop_started
        print(
            f"step {step + 1:02d}/{max_steps} model={model_latency * 1000:.1f} ms "
            f"clamp={info['delta_clamps']} anomaly={info['range_anomalies']} "
            f"target={np.round(safe_target, 4).tolist()}"
        )
        next_deadline = next_control_deadline(next_deadline, clock(), period)

    settle_started = clock()
    settle_origin = _full_state(robot)[:7] if execute and last_sent_target is not None else None
    while execute and last_sent_target is not None:
        current = _full_state(robot)
        target_error = float(np.max(np.abs(current[:7] - last_sent_target)))
        if target_error <= command_tolerance:
            break
        if clock() - settle_started >= command_timeout:
            feedback_progress = float(np.max(np.abs(current[:7] - settle_origin)))
            raise RuntimeError(
                f"运动无进展：目标误差仍为 {target_error:.4f} rad，"
                f"关节反馈仅变化 {feedback_progress:.4f} rad"
            )
        sleep(0.05)

    return {
        "mode": "execute" if execute else "dry-run",
        "command_frequency_hz": command_frequency,
        "command_speed_scale": send_speed_scale,
        "progress_window_s": progress_window,
        "commands_sent": commands_sent,
        "temporal_ensemble": temporal_enabled,
        "adaptive_mpc": adaptive_mpc_enabled,
        "mpc_fallbacks": mpc_fallbacks,
        "mpc_response_alpha_final": (
            mpc.alpha.astype(float).tolist() if mpc is not None else None
        ),
        "first_action_jitter_rad": _jitter(first_actions),
        "temporal_action_jitter_rad": _jitter(aggregate_actions),
        "final_action_jitter_rad": _jitter(final_actions),
        "actual_qpos_jitter_rad": _jitter(actual_qposes),
        "model_input_snapshots": input_snapshots,
        "grasp_gate_blocks": grasp_gate_blocks,
        "grasp_gate_min_attempt_distance_mm": (
            min(grasp_gate_distances) if grasp_gate_distances else None
        ),
        "grasp_height_gate_blocks": grasp_height_gate_blocks,
        "grasp_min_z_offset_mm": min(grasp_z_offsets) if grasp_z_offsets else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="完整推理但不发送任何运动命令（默认）")
    mode.add_argument("--execute", action="store_true", help="向机械臂发送经过 safety 的低速目标")
    parser.add_argument("--config", type=Path, default=Path("configs/act_nero_day6.yaml"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--speed-scale", type=float, default=0.25)
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument(
        "--adaptive-mpc", action="store_true",
        help="在 ACT chunk 后启用目标感知、自适应短时域 MPC（默认关闭）",
    )
    parser.add_argument(
        "--single-action-test", action="store_true",
        help="只发送第一条实时预测关节目标，随后继续观察反馈",
    )
    parser.add_argument("--extended", action="store_true", help="明确允许超过 10 step 的实机阶段")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--source-base-xyz-mm", nargs=3, type=float)
    parser.add_argument("--destination-base-xyz-mm", nargs=3, type=float)
    parser.add_argument(
        "--save-model-inputs", action="store_true",
        help="保存 step 0/50/100 真正送入 ACT 的缩放后 RGB tensor、PNG、统计量和哈希",
    )
    parser.add_argument(
        "--save-model-input-every", type=int, metavar="N",
        help="每 N 步保存一次真正送入 ACT 的双相机图像",
    )
    args = parser.parse_args()
    if not args.execute:
        args.dry_run = True
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps 必须大于 0")
    if not 0 < args.speed_scale <= 1:
        parser.error("--speed-scale 必须在 (0,1] 内")
    if args.save_model_input_every is not None and args.save_model_input_every <= 0:
        parser.error("--save-model-input-every 必须大于 0")
    if args.execute and args.max_steps is not None and args.max_steps > 10 and not args.extended:
        if not args.single_action_test:
            parser.error("首次实机最多 10 step；验证后显式加 --extended 才能继续")
    if args.single_action_test and not args.execute:
        parser.error("--single-action-test 必须和 --execute 一起使用")
    if (args.source_base_xyz_mm is None) != (args.destination_base_xyz_mm is None):
        parser.error("source/destination base XYZ 必须同时提供")
    return args


def main() -> int:
    args = parse_args()
    config, stats, policy, device = load_runtime(args.config, args.device)
    try:
        task_coordinates = resolve_task_coordinates(
            config, args.source_base_xyz_mm, args.destination_base_xyz_mm
        )
    except (Exception, KeyboardInterrupt) as exc:
        message = "用户中止" if isinstance(exc, KeyboardInterrupt) else str(exc)
        print(f"任务坐标定位失败：{message}；未连接机械臂。")
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    max_steps = args.max_steps or int(config["max_steps"])
    temporal_enabled = args.temporal_ensemble or bool(config["temporal_ensemble"].get("enabled", False))
    adaptive_mpc_enabled = args.adaptive_mpc or bool(
        config.get("adaptive_mpc", {}).get("enabled", False)
    )
    if args.execute and max_steps > 10 and not args.extended and not args.single_action_test:
        raise SystemExit("首次实机最多 10 step；验证后显式加 --extended 才能继续")
    mode = "execute_temporal" if args.execute and temporal_enabled else (
        "execute" if args.execute else "dry_run"
    )
    if adaptive_mpc_enabled:
        mode += "_mpc"
    logger = RealtimeLogger(Path(config["log_dir"]), mode)
    robot = NeroRobotWrapper()
    camera = None
    summary = {}
    tcp_config = json.loads(Path("config/tcp_offset.json").read_text(encoding="utf-8"))
    tcp_offset_m = np.asarray(tcp_config["tcp_offset_flange_xyz_m"], dtype=float)
    print("\nACT Day6 推理计划")
    print(f"  device：{device}；steps：{max_steps}；frequency：{config['control_frequency_hz']} Hz")
    print(f"  模式：{'实机低速执行' if args.execute else 'dry-run（绝不发送命令）'}")
    print(f"  temporal ensemble：{temporal_enabled}；speed scale：{args.speed_scale}")
    print(f"  adaptive MPC：{adaptive_mpc_enabled}")
    if config.get("fixed_start_joints_deg") is not None:
        print(f"  统一预备关节位姿（deg）：{config['fixed_start_joints_deg']}")
    if task_coordinates is not None:
        points_mm = task_coordinates.reshape(2, 3) * 1000
        task_config = config.get("task_coordinates", {})
        grasp_target = points_mm[0] + [0, 0, float(task_config.get("grasp_z_offset_mm", 0.0))]
        place_target = [
            points_mm[1, 0], points_mm[1, 1],
            grasp_target[2] + float(task_config.get("placement_gap_mm", 5.0)),
        ]
        print(f"  source base XYZ (mm)：{np.round(points_mm[0], 2).tolist()}")
        print(f"  destination base XYZ (mm)：{np.round(points_mm[1], 2).tolist()}")
        print(f"  expected grasp TCP XYZ (mm)：{np.round(grasp_target, 2).tolist()}")
        print(f"  expected place TCP XYZ (mm)：{np.round(place_target, 2).tolist()}")
        if task_config.get("grasp_gate_tolerance_mm") is not None:
            print(
                "  抓取安全门禁：TCP 需进入 "
                f"{float(task_config['grasp_gate_tolerance_mm']):.1f} mm 球形容差才允许闭爪"
            )
            if task_config.get("abort_on_grasp_gate_block", False):
                print("  提前闭爪保护：容差外请求闭爪将立即保持当前位置并结束")
        if task_config.get("progress_watchdog_window_s"):
            print(
                "  TCP 进展 watchdog：进入 "
                f"{float(task_config['progress_watchdog_start_distance_mm']):.1f} mm 后，"
                f"{float(task_config['progress_watchdog_window_s']):.1f} s 内至少改善 "
                f"{float(task_config['progress_watchdog_min_improvement_mm']):.1f} mm"
            )
        if task_config.get("grasp_height_gate_min_table_clearance_mm") is not None:
            print(
                "  抓取高度门禁：进入上述进展区域后，TCP 距目标平面低于 "
                f"{float(task_config['grasp_height_gate_min_table_clearance_mm']):.1f} mm 将停止"
            )
        elif task_config.get("grasp_height_gate_min_z_offset_mm") is not None:
            print(
                "  抓取高度门禁：进入上述进展区域后，TCP 的 Z 偏差低于 "
                f"{float(task_config['grasp_height_gate_min_z_offset_mm']):.1f} mm 将停止"
            )
    if args.single_action_test:
        print("  单次目标诊断：只发送第 1 条目标，随后只读取反馈")
    token = "MOVE_ONCE" if args.single_action_test else ("ACT_RUN" if args.execute else "CONNECT")
    warning = (
        "机械臂将低速运动；普通失败会保持当前位置，Ctrl-C 会电子急停。"
        "请准备实体急停并清空扫掠区域。"
        if args.execute else
        "将打开相机并连接、使能机械臂读取反馈，但不会发送运动目标。"
    )
    if not confirm_action("启动 ACT Day6 实机链路", warning, token):
        logger.close({"status": "cancelled"})
        return 0
    try:
        from src.act.record_episode import ActRgbCameras

        camera = ActRgbCameras(
            int(config["camera_width"]),
            config.get("secondary_camera_device"),
            int(config.get("secondary_camera_rotation_degrees", 0)),
        )
        robot.connect()
        if args.execute and config.get("fixed_start_joints_deg") is not None:
            from src.act.replay_episode import move_to_replay_start

            fixed_start = np.radians(
                np.asarray(config["fixed_start_joints_deg"], dtype=np.float32)
            )
            if fixed_start.shape != (7,):
                raise ValueError("fixed_start_joints_deg 必须为 7 维")
            print("正在低速移动到 ACT 统一预备位姿……")
            move_to_replay_start(
                robot, fixed_start,
                speed_percent=int(config.get("move_to_start_speed_percent", 10)),
                timeout_s=float(config.get("move_to_start_timeout_s", 45.0)),
                tolerance_rad=float(config.get("move_to_start_tolerance_rad", 0.02)),
            )
            print("已到达 ACT 统一预备位姿。")
        camera.open()
        summary = run_control_loop(
            policy, camera, robot, stats, config, logger,
            execute=args.execute, max_steps=max_steps, speed_scale=args.speed_scale,
            temporal_enabled=temporal_enabled,
            adaptive_mpc_enabled=adaptive_mpc_enabled,
            task_coordinates=task_coordinates,
            tcp_offset_m=tcp_offset_m,
            max_commands=1 if args.single_action_test else None,
            control_gripper=not args.single_action_test,
            model_input_steps=tuple(sorted(
                ({0, 50, 100} if args.save_model_inputs else set())
                | (set(range(0, max_steps, args.save_model_input_every))
                   if args.save_model_input_every else set())
            )),
        )
        if task_coordinates is not None:
            position_report = rollout_position_diagnostics(
                logger.rows, task_coordinates, robot.robot.fk, tcp_offset_m,
                config.get("task_coordinates", {}),
            )
            summary["position_diagnostics"] = position_report
            print_position_diagnostics(position_report)
        summary["status"] = "passed"
        print("\nDay6 loop 完成。")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        message = "用户中断" if interrupted else str(exc)
        no_progress = "无进展" in message or "进展停滞" in message
        summary = {
            "status": "failed", "error": message,
            "nan_rejections": int("NaN" in message or "Inf" in message or "有限" in message),
            "feedback_failures": int(not no_progress and ("反馈" in message or "取帧" in message)),
            "no_progress_rejections": int(no_progress),
        }
        if args.execute and robot.connected:
            try:
                status = robot.robot.get_arm_status()
                if status is not None:
                    print(f"控制器状态：\n{status.msg}")
                if interrupted:
                    robot.emergency_stop()
                    print("用户中断：已执行电子急停。")
                else:
                    held = robot.hold_current_position()
                    print(f"普通失败：已保持当前关节位置 {np.round(held, 4).tolist()}")
            except Exception as hold_error:
                print(f"保持当前位置失败：{hold_error}；回退到电子急停。")
                try:
                    robot.emergency_stop()
                except Exception as stop_error:
                    print(f"电子急停命令也失败：{stop_error}")
        print(f"Day6 失败：{message}")
        return 130 if interrupted else 1
    finally:
        if task_coordinates is not None:
            points_mm = task_coordinates.reshape(2, 3) * 1000.0
            summary["detected_source_base_xyz_mm"] = points_mm[0].astype(float).tolist()
            summary["detected_destination_base_xyz_mm"] = points_mm[1].astype(float).tolist()
        saved = logger.close(summary)
        if camera is not None:
            camera.close()
        if robot.connected:
            robot.disconnect()
        print(f"CSV：{logger.csv_path}")
        print(f"summary：{logger.summary_path}")
        if saved:
            print(json.dumps(saved, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
