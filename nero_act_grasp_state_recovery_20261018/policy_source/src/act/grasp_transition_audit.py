"""Audit ACT grasp transitions offline; never connects to cameras or the robot.

Run from the repository root::

    python -m src.act.grasp_transition_audit \
      data/act_dual_fixed/episode_*.hdf5 \
      --config configs/act_nero_dual_xyz_grasp_focus_day6.yaml \
      --output-dir artifacts/act_dual_xyz_grasp_focus_day5/grasp_transition_audit \
      --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import h5py
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nero-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.act.episode_inspect import inspect_episode
from src.act.nero_dataset import denormalize_actions, preprocess_images
from src.act.run_act_real import estimate_tcp_xyz_mm, load_runtime
from src.utils.arm import create_nero_arm


OPEN_THRESHOLD_M = 0.0475
RUNTIME_CLOSE_THRESHOLD_M = 0.035
DEFAULT_OFFSETS = (-30, -20, -10, -5, 0, 10)


def gripper_transitions(actions: np.ndarray) -> tuple[list[int], list[int]]:
    """Return open-to-close and close-to-open transition indices."""
    open_state = np.asarray(actions)[:, 7] >= OPEN_THRESHOLD_M
    close = (np.flatnonzero(open_state[:-1] & ~open_state[1:]) + 1).tolist()
    reopen = (np.flatnonzero(~open_state[:-1] & open_state[1:]) + 1).tolist()
    return close, reopen


def first_close_offset(values: np.ndarray, threshold_m: float) -> int | None:
    indices = np.flatnonzero(np.asarray(values) <= threshold_m)
    return int(indices[0]) if len(indices) else None


def episode_id(path: Path) -> int:
    return int(path.stem.removeprefix("episode_"))


def read_point(root: h5py.File, name: str) -> np.ndarray:
    if name not in root.attrs:
        raise ValueError(f"缺少 {name}")
    raw = root.attrs[name]
    if isinstance(raw, (str, bytes)):
        raw = json.loads(raw)
    point = np.asarray(raw, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} 不是有限三维坐标")
    return point


def load_tcp_offset(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    offset = np.asarray(payload["tcp_offset_flange_xyz_m"], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("TCP offset 必须是有限三维米坐标")
    return offset


def protocol_reasons(
    *, camera_names: list[str], frequency_hz: float, action_lead_steps: int,
    start_error_rad: float, first_move_step: int | None,
    close_count: int, reopen_count: int,
) -> list[str]:
    reasons = []
    if camera_names != ["camera_0", "camera_1"]:
        reasons.append("camera_set")
    if abs(frequency_hz - 20.0) > 0.5:
        reasons.append("frequency")
    if action_lead_steps != 0:
        reasons.append("legacy_action_lead")
    if start_error_rad > 0.03:
        reasons.append("home_pose")
    if first_move_step is None or first_move_step > 10:
        reasons.append("long_or_missing_start_pause")
    if close_count != 1 or reopen_count != 1:
        reasons.append("gripper_events")
    return reasons


def save_preview(
    root: h5py.File, close_index: int, offsets: tuple[int, ...], output: Path,
) -> None:
    cells = []
    for camera in ("camera_0", "camera_1"):
        row = []
        frames = root[f"observations/images/{camera}"]
        for offset in offsets:
            index = min(max(close_index + offset, 0), len(frames) - 1)
            bgr = cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR)
            cell = cv2.resize(bgr, (200, 112), interpolation=cv2.INTER_AREA)
            cv2.putText(
                cell, f"{camera} {offset:+d}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 255), 1, cv2.LINE_AA,
            )
            row.append(cell)
        cells.append(np.hstack(row))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.vstack(cells), [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f"无法保存预览：{output}")


def audit_episodes(
    paths: list[Path], output_dir: Path, tcp_offset_m: np.ndarray,
    home_qpos: np.ndarray, offsets: tuple[int, ...],
) -> tuple[list[dict], list[dict]]:
    robot = create_nero_arm()
    rows, windows = [], []
    preview_dir = output_dir / "previews"
    for path in paths:
        schema_error = None
        try:
            inspect_episode(path)
        except (OSError, KeyError, ValueError) as exc:
            schema_error = str(exc)
        row = {"episode_id": episode_id(path), "path": str(path), "schema_error": schema_error}
        if schema_error:
            row.update({"protocol_compatible": False, "reasons": "schema"})
            rows.append(row)
            continue
        with h5py.File(path, "r") as root:
            qpos = root["observations/qpos"][:].astype(float)
            actions = root["action"][:].astype(float)
            timestamps = root["timestamp"][:].astype(float)
            cameras = list(root["observations/images"].keys())
            close, reopen = gripper_transitions(actions)
            displacement = np.linalg.norm(qpos - qpos[0], axis=1)
            moved = np.flatnonzero(displacement > 0.01)
            first_move = int(moved[0]) if len(moved) else None
            frequency = float(root.attrs.get("actual_frequency_hz", root.attrs.get("control_frequency_hz", 0)))
            action_lead = int(root.attrs.get("action_lead_steps", -1))
            start_error = float(np.max(np.abs(qpos[0] - home_qpos)))
            reasons = protocol_reasons(
                camera_names=cameras, frequency_hz=frequency,
                action_lead_steps=action_lead, start_error_rad=start_error,
                first_move_step=first_move, close_count=len(close), reopen_count=len(reopen),
            )
            source = read_point(root, "source_base_xyz_mm")
            destination = read_point(root, "destination_base_xyz_mm")
            row.update({
                "created_at": str(root.attrs.get("created_at", "")),
                "num_steps": len(qpos), "frequency_hz": frequency,
                "action_lead_steps": action_lead, "first_move_step": first_move,
                "start_error_rad": start_error, "close_indices": json.dumps(close),
                "reopen_indices": json.dumps(reopen),
                "source_xyz_mm": json.dumps(source.tolist()),
                "destination_xyz_mm": json.dumps(destination.tolist()),
                "protocol_compatible": not reasons, "reasons": ";".join(reasons),
            })
            if len(close) != 1:
                rows.append(row)
                continue
            close_index = close[0]
            lo, hi = max(0, close_index - 30), min(len(qpos), close_index + 11)
            tcp = np.stack([
                estimate_tcp_xyz_mm(joints, robot.fk, tcp_offset_m) for joints in qpos[lo:hi]
            ])
            relative = tcp - source
            dt = np.diff(timestamps[lo:hi])
            speed = np.r_[0.0, np.linalg.norm(np.diff(tcp, axis=0), axis=1) / dt]
            close_local = close_index - lo
            approach_start = max(0, close_local - 20)
            approach_delta = relative[close_local] - relative[approach_start]
            before_speed = speed[max(0, close_local - 30):close_local]
            approach_path = relative[approach_start:close_local + 1]
            row.update({
                "close_relative_tcp_xyz_mm": json.dumps(relative[close_local].tolist()),
                "approach_20_delta_xyz_mm": json.dumps(approach_delta.tolist()),
                "preclose_speed_median_mm_s": float(np.median(before_speed)),
                "preclose_speed_max_mm_s": float(np.max(before_speed)),
                "preclose_pause_frames_below_2_mm_s": int(np.count_nonzero(before_speed < 2.0)),
                "approach_20_y_span_mm": float(np.ptp(approach_path[:, 1])),
                "approach_20_z_span_mm": float(np.ptp(approach_path[:, 2])),
                "approach_20_z_sag_mm": float(
                    approach_path[-1, 2] - np.min(approach_path[:, 2])
                ),
                "approach_20_x_reverse_mm": float(
                    np.maximum(np.diff(approach_path[:, 0]), 0.0).sum()
                ),
                "preview": str(preview_dir / f"episode_{episode_id(path)}.jpg"),
            })
            for local, index in enumerate(range(lo, hi)):
                windows.append({
                    "episode_id": episode_id(path), "step": index,
                    "offset_from_close": index - close_index,
                    "timestamp_s": float(timestamps[index] - timestamps[close_index]),
                    "tcp_x_mm": float(tcp[local, 0]), "tcp_y_mm": float(tcp[local, 1]),
                    "tcp_z_mm": float(tcp[local, 2]),
                    "relative_tcp_x_mm": float(relative[local, 0]),
                    "relative_tcp_y_mm": float(relative[local, 1]),
                    "relative_tcp_z_mm": float(relative[local, 2]),
                    "tcp_speed_mm_s": float(speed[local]),
                    "qpos": json.dumps(qpos[index].tolist()),
                    "gripper_action_m": float(actions[index, 7]),
                    "source_hdf5": str(path),
                })
            save_preview(root, close_index, offsets, preview_dir / f"episode_{episode_id(path)}.jpg")
            rows.append(row)
    return rows, windows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_tcp_windows(windows: list[dict], compatible_ids: set[int], output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    grouped = defaultdict(list)
    for row in windows:
        if row["episode_id"] in compatible_ids:
            grouped[row["episode_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["offset_from_close"])
        x = [row["offset_from_close"] for row in values]
        for axis, key in zip(axes, ("relative_tcp_x_mm", "relative_tcp_y_mm", "relative_tcp_z_mm")):
            axis.plot(x, [row[key] for row in values], alpha=0.35, linewidth=1)
    for axis, label in zip(axes, ("X", "Y", "Z")):
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_ylabel(f"TCP-source {label} (mm)")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("steps relative to close")
    figure.suptitle("Grasp-compatible transitions (TCP offset is approximate)")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def summarize_audit(rows: list[dict], windows: list[dict]) -> dict:
    for row in rows:
        trajectory_reasons = []
        warnings = []
        if row.get("close_relative_tcp_xyz_mm"):
            close = np.asarray(json.loads(row["close_relative_tcp_xyz_mm"]))
            approach = np.asarray(json.loads(row["approach_20_delta_xyz_mm"]))
            if np.linalg.norm(close) > 10.0:
                trajectory_reasons.append("close_point")
            if approach[0] > -60.0 or abs(approach[1]) > 10.0 or abs(approach[2]) > 10.0:
                trajectory_reasons.append("side_approach")
            if int(row["preclose_pause_frames_below_2_mm_s"]) > 5:
                trajectory_reasons.append("preclose_pause")
            if float(row["approach_20_z_sag_mm"]) > 15.0:
                warnings.append("vertical_sag_over_15mm")
        else:
            trajectory_reasons.append("missing_close_window")
        row["trajectory_reasons"] = ";".join(trajectory_reasons)
        row["warnings"] = ";".join(warnings)
        row["grasp_compatible"] = bool(
            row.get("protocol_compatible") and not trajectory_reasons
        )
    good = [row for row in rows if row.get("grasp_compatible")]
    close_positions = np.array([
        json.loads(row["close_relative_tcp_xyz_mm"]) for row in good
        if row.get("close_relative_tcp_xyz_mm")
    ])
    destinations = np.array([
        json.loads(row["destination_xyz_mm"]) for row in rows if row.get("destination_xyz_mm")
    ])
    reason_counts = Counter(
        reason for row in rows for reason in str(row.get("reasons", "")).split(";") if reason
    )
    summary = {
        "episode_count": len(rows),
        "schema_valid_count": sum(not row.get("schema_error") for row in rows),
        "protocol_compatible_count": sum(
            bool(row.get("protocol_compatible")) for row in rows
        ),
        "protocol_compatible_ids": [
            row["episode_id"] for row in rows if row.get("protocol_compatible")
        ],
        "grasp_compatible_count": len(good),
        "grasp_compatible_ids": [row["episode_id"] for row in good],
        "rejected_ids": [row["episode_id"] for row in rows if not row.get("grasp_compatible")],
        "reason_counts": dict(reason_counts),
        "trajectory_reason_counts": dict(Counter(
            reason for row in rows
            for reason in str(row.get("trajectory_reasons", "")).split(";") if reason
        )),
        "warning_counts": dict(Counter(
            warning for row in rows
            for warning in str(row.get("warnings", "")).split(";") if warning
        )),
        "warning_ids": [row["episode_id"] for row in rows if row.get("warnings")],
        "window_row_count": len(windows),
        "tcp_offset_approximate": True,
    }
    if len(close_positions):
        summary["close_relative_tcp_xyz_mm"] = {
            "mean": close_positions.mean(axis=0).tolist(),
            "std": close_positions.std(axis=0).tolist(),
            "min": close_positions.min(axis=0).tolist(),
            "max": close_positions.max(axis=0).tolist(),
        }
        summary["approach_20_steps"] = {
            key: {
                "p50": float(np.median([float(row[key]) for row in good])),
                "p95": float(np.percentile([float(row[key]) for row in good], 95)),
                "max": float(np.max([float(row[key]) for row in good])),
            }
            for key in (
                "approach_20_y_span_mm", "approach_20_z_span_mm",
                "approach_20_z_sag_mm", "approach_20_x_reverse_mm",
                "preclose_speed_median_mm_s",
            )
        }
    if len(destinations):
        summary["destination_xyz_mm"] = {
            "min": destinations.min(axis=0).tolist(),
            "max": destinations.max(axis=0).tolist(),
            "span": np.ptp(destinations, axis=0).tolist(),
            "fixed_destination": bool(np.linalg.norm(np.ptp(destinations, axis=0)) <= 10.0),
        }
    return summary


def transition_group_summary(records: list[dict]) -> dict:
    result = {}
    for (split, offset), values in sorted(
        ((key, group) for key, group in _group_records(records).items()),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        expected = [row for row in values if row["target_close_offset"] is not None]
        detected = [row for row in expected if row["predicted_close_offset"] is not None]
        no_close_expected = [row for row in values if row["target_close_offset"] is None]
        timing = [
            row["predicted_close_offset"] - row["target_close_offset"] for row in detected
        ]
        result[f"{split}:{offset:+d}"] = {
            "count": len(values), "expected_close_count": len(expected),
            "close_detected_count": len(detected),
            "close_recall": len(detected) / len(expected) if expected else None,
            "premature_close_count": sum(
                row["predicted_close_offset"] is not None for row in no_close_expected
            ),
            "timing_error_steps_mean": float(np.mean(timing)) if timing else None,
            "predicted_min_gripper_m_mean": float(np.mean([
                row["predicted_min_gripper_m"] for row in values
            ])),
        }
    return result


def _group_records(records: list[dict]) -> dict[tuple[str, int], list[dict]]:
    groups = defaultdict(list)
    for row in records:
        groups[(row["split"], row["offset_from_close"])].append(row)
    return groups


def run_transition_test(
    config_path: Path, offsets: tuple[int, ...], device_name: str,
) -> tuple[list[dict], dict]:
    config, stats, policy, device = load_runtime(config_path, device_name)
    payload = torch.load(Path(config["checkpoint"]), map_location="cpu", weights_only=False)
    split = payload["split"]
    id_to_split = {int(eid): name for name in ("train", "val") for eid in split[name]}
    manifest = json.loads(Path(payload["config"]["manifest"]).read_text(encoding="utf-8"))
    paths = [Path(row["path"]) for row in manifest["episodes"]
             if row["status"] == "valid" and episode_id(Path(row["path"])) in id_to_split]
    samples = []
    chunk_size = int(config["chunk_size"])
    for path in paths:
        with h5py.File(path, "r") as root:
            actions = root["action"][:].astype(np.float32)
            close, _ = gripper_transitions(actions)
            if len(close) != 1:
                continue
            task = np.concatenate([
                read_point(root, "source_base_xyz_mm"),
                read_point(root, "destination_base_xyz_mm"),
            ]).astype(np.float32) / 1000.0
            for offset in offsets:
                start = close[0] + offset
                if not 0 <= start < len(actions):
                    continue
                state = np.concatenate([
                    root["observations/qpos"][start],
                    root["observations/gripper_state"][start], task,
                ]).astype(np.float32)
                image = torch.stack([
                    preprocess_images(
                        root[f"observations/images/{camera}"][start][None], config["image_size"]
                    )[0] for camera in config["camera_names"]
                ])
                target_gripper = actions[start:min(start + chunk_size, len(actions)), 7]
                samples.append((path, offset, state, image, target_gripper))
    records = []
    batch_size = 4
    with torch.inference_mode():
        for begin in range(0, len(samples), batch_size):
            batch = samples[begin:begin + batch_size]
            state = np.stack([(sample[2] - stats["qpos_mean"]) / stats["qpos_std"] for sample in batch])
            images = torch.stack([sample[3] for sample in batch]).to(device)
            prediction = policy(torch.from_numpy(state).to(device), images)
            prediction = denormalize_actions(prediction, stats).cpu().numpy()
            for sample, chunk in zip(batch, prediction):
                path, offset, _, _, target = sample
                pred_gripper = chunk[:, 7]
                records.append({
                    "episode_id": episode_id(path), "split": id_to_split[episode_id(path)],
                    "offset_from_close": offset,
                    "target_close_offset": first_close_offset(target, OPEN_THRESHOLD_M),
                    "predicted_close_offset": first_close_offset(
                        pred_gripper, RUNTIME_CLOSE_THRESHOLD_M
                    ),
                    "predicted_binary_close_offset": first_close_offset(
                        pred_gripper, OPEN_THRESHOLD_M
                    ),
                    "predicted_min_gripper_m": float(pred_gripper.min()),
                    "predicted_first_gripper_m": float(pred_gripper[0]),
                    "predicted_gripper_chunk": json.dumps(pred_gripper.astype(float).tolist()),
                    "target_gripper_chunk": json.dumps(target.astype(float).tolist()),
                })
    return records, {
        "checkpoint": str(config["checkpoint"]),
        "checkpoint_epoch": int(payload["epoch"]),
        "chunk_size": chunk_size,
        "runtime_close_threshold_m": RUNTIME_CLOSE_THRESHOLD_M,
        "binary_close_threshold_m": OPEN_THRESHOLD_M,
        "episode_ids": split,
        "by_split_and_offset": transition_group_summary(records),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tcp-config", type=Path, default=Path("config/tcp_offset.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--snapshot-offsets", nargs="+", type=int, default=DEFAULT_OFFSETS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = sorted(set(args.episodes), key=episode_id)
    if not paths:
        raise ValueError("没有 episode")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    home = np.radians(np.asarray(config["fixed_start_joints_deg"], dtype=float))
    offsets = tuple(sorted(set(args.snapshot_offsets)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, windows = audit_episodes(
        paths, args.output_dir, load_tcp_offset(args.tcp_config), home, offsets,
    )
    audit_summary = summarize_audit(rows, windows)
    write_csv(args.output_dir / "episode_audit.csv", rows)
    write_csv(args.output_dir / "grasp_windows.csv", windows)
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_tcp_windows(
        windows, set(audit_summary["grasp_compatible_ids"]),
        args.output_dir / "task_relative_tcp.png",
    )
    predictions, transition_summary = run_transition_test(args.config, offsets, args.device)
    write_csv(args.output_dir / "grasp_transition_predictions.csv", predictions)
    (args.output_dir / "grasp_transition_summary.json").write_text(
        json.dumps(transition_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"audit": audit_summary, "transition": transition_summary},
                     ensure_ascii=False, indent=2))
    print(f"离线审计完成：{args.output_dir}；未连接相机、CAN 或机械臂")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
