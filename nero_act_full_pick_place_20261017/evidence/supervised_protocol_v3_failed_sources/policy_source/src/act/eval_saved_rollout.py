"""Replay saved ACT model inputs offline and summarize prediction stability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.act.nero_dataset import denormalize_actions
from src.act.run_act_real import load_runtime


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "p95": float(np.percentile(values, 95)) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
    }


def prediction_stability(first_actions: np.ndarray, states: np.ndarray) -> dict:
    """Summarize requested joint offsets and their changes between snapshots."""
    requested = first_actions[:, :7] - states[:, :7]
    changes = np.linalg.norm(np.diff(requested, axis=0), axis=1)
    signs = np.sign(np.where(np.abs(requested) >= 0.002, requested, 0.0))
    flips = np.sum((signs[1:] * signs[:-1]) < 0, axis=0)
    return {
        "requested_joint_offset_norm_rad": _percentiles(
            np.linalg.norm(requested, axis=1)
        ),
        "consecutive_offset_change_norm_rad": _percentiles(changes),
        "direction_flips_by_joint": flips.astype(int).tolist(),
        "direction_flip_total": int(flips.sum()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--model-input-dir", type=Path, required=True)
    parser.add_argument("--source-base-xyz-mm", nargs=3, type=float, required=True)
    parser.add_argument("--destination-base-xyz-mm", nargs=3, type=float, required=True)
    parser.add_argument("--from-step", type=int, default=0)
    parser.add_argument("--to-step", type=int)
    parser.add_argument("--fixed-qpos-step", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config, stats, policy, device = load_runtime(args.config, args.device)
    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = {int(row["step"]): row for row in csv.DictReader(stream)}

    camera_names = list(config["camera_names"])
    first_camera = camera_names[0]
    snapshots = []
    for path in sorted(args.model_input_dir.glob(f"step_*_{first_camera}.npy")):
        step = int(path.name.split("_")[1])
        if step < args.from_step or (args.to_step is not None and step > args.to_step):
            continue
        if step not in rows:
            raise ValueError(f"CSV 缺少模型输入 step {step}")
        images = [np.load(args.model_input_dir / f"step_{step:04d}_{name}.npy")
                  for name in camera_names]
        image = np.stack(images).astype(np.float32, copy=False)
        if image.shape != (len(camera_names), 3, *config["image_size"]):
            raise ValueError(f"step {step} 图像 shape 异常：{image.shape}")
        snapshots.append((step, image, np.asarray(json.loads(rows[step]["qpos"]), np.float32)))
    if len(snapshots) < 2:
        raise ValueError("至少需要两个处于指定 step 范围内的模型输入快照")

    task = np.asarray(
        [*args.source_base_xyz_mm, *args.destination_base_xyz_mm], np.float32
    ) / 1000.0

    def infer(image: np.ndarray, state: np.ndarray) -> np.ndarray:
        model_state = np.concatenate([state, task])
        normalized = (model_state - stats["qpos_mean"]) / stats["qpos_std"]
        with torch.inference_mode():
            prediction = policy(
                torch.from_numpy(normalized)[None].to(device),
                torch.from_numpy(image)[None].to(device),
            )[0]
        return denormalize_actions(prediction, stats).cpu().numpy()

    chunks = np.stack([infer(image, state) for _, image, state in snapshots])
    states = np.stack([state for _, _, state in snapshots])
    first_actions = chunks[:, 0]
    fixed_step = args.fixed_qpos_step
    if fixed_step is None:
        fixed_index = len(snapshots) // 2
    else:
        candidates = [index for index, (step, _, _) in enumerate(snapshots) if step == fixed_step]
        if not candidates:
            raise ValueError(f"模型输入中没有 --fixed-qpos-step {fixed_step}")
        fixed_index = candidates[0]
    fixed_state = snapshots[fixed_index][2]
    fixed_chunks = np.stack([infer(image, fixed_state) for _, image, _ in snapshots])
    fixed_first = fixed_chunks[:, 0, :7]
    pairwise_first_mae = [
        float(np.mean(np.abs(fixed_first[left] - fixed_first[right])))
        for left in range(len(fixed_first)) for right in range(left + 1, len(fixed_first))
    ]
    close_threshold = float(config["safety"]["gripper_close_threshold_m"])
    payload = torch.load(Path(config["checkpoint"]), map_location="cpu", weights_only=False)
    report = {
        "config": str(args.config),
        "checkpoint": str(config["checkpoint"]),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "csv": str(args.csv),
        "model_input_dir": str(args.model_input_dir),
        "steps": [step for step, _, _ in snapshots],
        "task_coordinates_mm": (task.reshape(2, 3) * 1000.0).astype(float).tolist(),
        "input_sha256": [
            hashlib.sha256(image.tobytes()).hexdigest() for _, image, _ in snapshots
        ],
        "live_state_and_image": prediction_stability(first_actions, states),
        "fixed_qpos_image_test": {
            "qpos_step": int(snapshots[fixed_index][0]),
            "pairwise_first_action_joint_mae_rad": _percentiles(
                np.asarray(pairwise_first_mae)
            ),
            "per_joint_first_action_range_rad": np.ptp(fixed_first, axis=0).tolist(),
        },
        "gripper": {
            "first_action_min_m": float(first_actions[:, 7].min()),
            "first_action_max_m": float(first_actions[:, 7].max()),
            "first_action_close_count": int(np.count_nonzero(first_actions[:, 7] <= close_threshold)),
            "chunk_close_snapshot_count": int(np.count_nonzero(
                np.any(chunks[:, :, 7] <= close_threshold, axis=1)
            )),
            "close_threshold_m": close_threshold,
        },
        "first_actions": [
            {"step": int(step), "action": action.astype(float).tolist()}
            for (step, _, _), action in zip(snapshots, first_actions)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
