"""Evaluate a NERO ACT checkpoint on every timestep of the validation episodes."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.act.action_safety import sanitize_action_chunk
from src.act.dataset_stats import build_summary
from src.act.nero_dataset import (
    NeroEpisodicDataset,
    denormalize_actions,
    episode_effective_end,
    episode_trim_start,
    valid_episode_paths,
)
from src.act.nero_policy import NeroActPolicy
from src.act.temporal_ensemble import compare_first_vs_temporal
from src.act.validate_episode import NERO_JOINT_LIMITS_RAD


LABELS = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "gripper"]


def accumulate_errors(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray,
                      totals: dict) -> None:
    error = np.abs(prediction - target)
    mask = valid[..., None]
    totals["sum"] += float((error * mask).sum())
    totals["count"] += int(mask.sum()) * error.shape[-1]
    totals["joint_sum"] += (error * mask).sum(axis=(0, 1))
    totals["joint_count"] += valid.sum()
    totals["step_sum"] += (error * mask).sum(axis=(0, 2))
    totals["step_count"] += valid.sum(axis=0) * error.shape[-1]


def _all_starts(paths: list[Path], *, first_step: int = 0,
                chunk_size: int = 1,
                first_steps_by_parent: dict[str, int] | None = None,
                trim_after_gripper_open_steps: int | None = None,
                ) -> list[tuple[int, int]]:
    samples = []
    for path_index, path in enumerate(paths):
        path_first_step = episode_trim_start(path, first_step, first_steps_by_parent)
        with h5py.File(path, "r") as root:
            last_start = episode_effective_end(
                root, trim_after_gripper_open_steps
            ) - chunk_size
            samples.extend(
                (path_index, start) for start in range(path_first_step, last_start + 1)
            )
    return samples


def _plot_bar(values, labels, title, ylabel, output):
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, values)
    axis.set(title=title, ylabel=ylabel)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def _plot_sample(gt: np.ndarray, prediction: np.ndarray, output: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(gt[:, index], label="GT")
        axis.plot(prediction[:, index], label="Pred")
        axis.set_title(LABELS[index])
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def evaluate(checkpoint: Path, config_path: Path, split_name: str,
             device_override: str | None = None) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    requested_device = device_override or config.get("device", "auto")
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else (
        "cpu" if requested_device == "auto" else requested_device
    ))
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    split = payload.get("split") or json.loads(Path(config["split_file"]).read_text(encoding="utf-8"))
    paths = valid_episode_paths(Path(config["manifest"]), split[split_name])
    with (checkpoint.parent / "dataset_stats.pkl").open("rb") as stream:
        stats = pickle.load(stream)
    samples = _all_starts(
        paths, first_step=int(config.get("trim_start_steps", 0)),
        chunk_size=int(config["chunk_size"]),
        first_steps_by_parent=config.get("trim_start_steps_by_parent"),
        trim_after_gripper_open_steps=config.get("trim_after_gripper_open_steps"),
    )
    dataset = NeroEpisodicDataset(
        paths, config["camera_names"], stats,
        alignment_offset=config["alignment_offset"], image_size=config["image_size"],
        sample_starts=samples,
        coordinate_conditioning=bool(config.get("coordinate_conditioning", False)),
        joint_lookahead_steps=int(config.get("joint_lookahead_steps", 0)),
        joint_target_lead_steps=config.get("joint_target_lead_steps"),
        trim_after_gripper_open_steps=config.get("trim_after_gripper_open_steps"),
    )
    loader = DataLoader(dataset, batch_size=int(config["eval_batch_size"]), shuffle=False, num_workers=0)
    model_config = dict(payload["config"])
    model_config["device"] = str(device)
    model_config["pretrained_backbone"] = False
    policy = NeroActPolicy(model_config).to(device)
    policy.load_state_dict(payload["model"])
    policy.eval()
    chunk = int(config["chunk_size"])
    empty = lambda: {
        "sum": 0.0, "count": 0, "joint_sum": np.zeros(8), "joint_count": 0,
        "step_sum": np.zeros(chunk), "step_count": np.zeros(chunk),
    }
    normalized, real = empty(), empty()
    pred_min, pred_max = np.full(8, np.inf), np.full(8, -np.inf)
    gt_min, gt_max = np.full(8, np.inf), np.full(8, -np.inf)
    joint_out_of_range = np.zeros(7, dtype=np.int64)
    gripper_below_zero = gripper_above_max = 0
    gripper_abs_sum = gripper_clipped_abs_sum = gripper_binary_abs_sum = 0.0
    gripper_correct = gripper_count = 0
    gripper_step_correct = np.zeros(chunk)
    gripper_step_count = np.zeros(chunk)
    safe_prediction_min, safe_prediction_max = np.full(8, np.inf), np.full(8, -np.inf)
    max_delta, non_finite_count, postprocess_rejections = 0.0, 0, 0
    prediction_chunks, current_targets = [], []
    lower, upper = NERO_JOINT_LIMITS_RAD[:, 0], NERO_JOINT_LIMITS_RAD[:, 1]
    with torch.no_grad():
        for image, qpos, action, is_pad, starts in loader:
            image, qpos = image.to(device), qpos.to(device)
            prediction = policy(qpos, image).cpu()
            target = action[:, :chunk]
            valid = (~is_pad[:, :chunk]).numpy()
            pred_norm, gt_norm = prediction.numpy(), target.numpy()
            pred_real = denormalize_actions(prediction, stats).numpy()
            gt_real = denormalize_actions(target, stats).numpy()
            prediction_chunks.append(pred_real)
            current_targets.append(gt_real[:, 0])
            accumulate_errors(pred_norm, gt_norm, valid, normalized)
            accumulate_errors(pred_real, gt_real, valid, real)
            pred_min, pred_max = np.minimum(pred_min, pred_real.min((0, 1))), np.maximum(pred_max, pred_real.max((0, 1)))
            valid_gt = gt_real[valid]
            gt_min, gt_max = np.minimum(gt_min, valid_gt.min(0)), np.maximum(gt_max, valid_gt.max(0))
            non_finite_count += int(pred_real.size - np.isfinite(pred_real).sum())
            joint_out_of_range += np.count_nonzero(
                (pred_real[..., :7] < lower) | (pred_real[..., :7] > upper), axis=(0, 1)
            )
            gripper_below_zero += int(np.count_nonzero(pred_real[..., 7] < 0.0))
            gripper_above_max += int(np.count_nonzero(pred_real[..., 7] > 0.1))
            valid_gripper_prediction = pred_real[..., 7][valid]
            valid_gripper_target = gt_real[..., 7][valid]
            clipped_gripper = np.clip(valid_gripper_prediction, 0.0, 0.095)
            binary_gripper = np.where(clipped_gripper >= 0.0475, 0.095, 0.0)
            target_open = valid_gripper_target >= 0.0475
            gripper_abs_sum += float(np.abs(valid_gripper_prediction - valid_gripper_target).sum())
            gripper_clipped_abs_sum += float(np.abs(clipped_gripper - valid_gripper_target).sum())
            gripper_binary_abs_sum += float(np.abs(binary_gripper - valid_gripper_target).sum())
            gripper_correct += int(np.count_nonzero((binary_gripper > 0) == target_open))
            gripper_count += len(valid_gripper_target)
            step_correct = ((pred_real[..., 7] >= 0.0475) == (gt_real[..., 7] >= 0.0475)) & valid
            gripper_step_correct += step_correct.sum(axis=0)
            gripper_step_count += valid.sum(axis=0)
            try:
                safe_prediction, _ = sanitize_action_chunk(pred_real)
            except ValueError:
                postprocess_rejections += len(pred_real)
            else:
                safe_prediction_min = np.minimum(safe_prediction_min, safe_prediction.min((0, 1)))
                safe_prediction_max = np.maximum(safe_prediction_max, safe_prediction.max((0, 1)))
            if chunk > 1:
                max_delta = max(max_delta, float(np.abs(np.diff(pred_real[..., :7], axis=1)).max()))

    normalized_mae = normalized["sum"] / normalized["count"]
    global_mae = real["sum"] / real["count"]
    per_joint = real["joint_sum"] / real["joint_count"]
    chunk_mae = np.divide(real["step_sum"], real["step_count"],
                          out=np.zeros_like(real["step_sum"]), where=real["step_count"] > 0)
    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _plot_bar(per_joint, LABELS, "Validation per-channel MAE", "rad / m",
              artifact_dir / "per_joint_mae.png")
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(np.arange(chunk), chunk_mae, marker="o")
    axis.set(title="Validation chunk-step MAE", xlabel="future step", ylabel="mean rad / m")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(artifact_dir / "chunk_step_mae.png", dpi=140)
    plt.close(figure)

    fixed_path = artifact_dir / "fixed_val_samples.json"
    fixed = json.loads(fixed_path.read_text(encoding="utf-8")) if fixed_path.exists() else []
    path_by_id = {int(path.stem.removeprefix("episode_")): path for path in paths}
    for index, record in enumerate(fixed[:3]):
        path = path_by_id[record["episode_id"]]
        sample = NeroEpisodicDataset(
            [path], config["camera_names"], stats,
            alignment_offset=config["alignment_offset"], image_size=config["image_size"],
            fixed_start_ts=record["start_ts"],
            coordinate_conditioning=bool(config.get("coordinate_conditioning", False)),
            joint_lookahead_steps=int(config.get("joint_lookahead_steps", 0)),
            joint_target_lead_steps=config.get("joint_target_lead_steps"),
            trim_after_gripper_open_steps=config.get("trim_after_gripper_open_steps"),
        )[0]
        image, qpos, action = sample[:3]
        with torch.no_grad():
            prediction = policy(qpos[None].to(device), image[None].to(device))[0].cpu()
        pred_real = denormalize_actions(prediction, stats).numpy()
        gt_real = denormalize_actions(action[:chunk], stats).numpy()
        _plot_sample(gt_real, pred_real, artifact_dir / f"sample_{index}_pred_vs_gt.png")

    out_of_range_count = int(joint_out_of_range.sum() + gripper_below_zero + gripper_above_max)
    episode_lengths = []
    first_step = int(config.get("trim_start_steps", 0))
    for path in paths:
        path_first_step = episode_trim_start(
            path, first_step, config.get("trim_start_steps_by_parent")
        )
        with h5py.File(path, "r") as root:
            episode_lengths.append(
                max(0, episode_effective_end(
                    root, config.get("trim_after_gripper_open_steps")
                ) - path_first_step - chunk + 1)
            )
    all_chunks, all_current_targets = np.concatenate(prediction_chunks), np.concatenate(current_targets)
    decay_candidates = config.get("temporal_decay_candidates", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    temporal_ab = {
        str(decay): compare_first_vs_temporal(
            all_chunks, all_current_targets, episode_lengths, float(decay)
        )
        for decay in decay_candidates
    }
    result = {
        "checkpoint": str(checkpoint), "checkpoint_epoch": int(payload["epoch"]),
        "split": split_name, "episode_ids": split[split_name], "sample_count": len(samples),
        "normalized_global_mae": normalized_mae, "global_mae_real": global_mae,
        "per_joint_mae": dict(zip(LABELS, per_joint.tolist())),
        "chunk_step_mae": chunk_mae.tolist(),
        "prediction_min": pred_min.tolist(), "prediction_max": pred_max.tolist(),
        "gt_min": gt_min.tolist(), "gt_max": gt_max.tolist(),
        "non_finite_count": non_finite_count, "out_of_range_count": out_of_range_count,
        "joint_out_of_range_count": dict(zip(LABELS[:7], joint_out_of_range.tolist())),
        "gripper_below_zero_count": gripper_below_zero,
        "gripper_above_max_count": gripper_above_max,
        "gripper_raw_mae_m": gripper_abs_sum / gripper_count,
        "gripper_clipped_mae_m": gripper_clipped_abs_sum / gripper_count,
        "gripper_binary_mae_m": gripper_binary_abs_sum / gripper_count,
        "gripper_binary_accuracy": gripper_correct / gripper_count,
        "gripper_binary_accuracy_by_chunk_step": np.divide(
            gripper_step_correct, gripper_step_count,
            out=np.zeros_like(gripper_step_correct), where=gripper_step_count > 0,
        ).tolist(),
        "safe_prediction_min": safe_prediction_min.tolist(),
        "safe_prediction_max": safe_prediction_max.tolist(),
        "postprocessed_out_of_range_count": postprocess_rejections,
        "max_predicted_joint_delta_rad": max_delta,
        "temporal_ab": temporal_ab,
    }
    (artifact_dir / "offline_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    dataset_stats_path = artifact_dir / "dataset_stats.json"
    if dataset_stats_path.exists():
        dataset_stats = json.loads(dataset_stats_path.read_text(encoding="utf-8"))
    else:
        dataset_stats = build_summary(
            Path(config["manifest"]), split, first_step,
            config.get("trim_start_steps_by_parent"),
            config.get("trim_after_gripper_open_steps"),
        )
        dataset_stats_path.write_text(
            json.dumps(dataset_stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    safety = {
        "train_qpos_min": dataset_stats.get("train_qpos", {}).get("min"),
        "train_qpos_max": dataset_stats.get("train_qpos", {}).get("max"),
        "train_action_min": dataset_stats.get("train_action", {}).get("min"),
        "train_action_max": dataset_stats.get("train_action", {}).get("max"),
        "val_prediction_min": result["prediction_min"],
        "val_prediction_max": result["prediction_max"],
        "joint_limits_rad": NERO_JOINT_LIMITS_RAD.tolist(),
        "out_of_range_count": out_of_range_count,
        "joint_out_of_range_count": result["joint_out_of_range_count"],
        "gripper_below_zero_count": gripper_below_zero,
        "gripper_above_max_count": gripper_above_max,
        "gripper_clipped_mae_m": result["gripper_clipped_mae_m"],
        "gripper_binary_mae_m": result["gripper_binary_mae_m"],
        "gripper_binary_accuracy": result["gripper_binary_accuracy"],
        "gripper_binary_accuracy_by_chunk_step": result["gripper_binary_accuracy_by_chunk_step"],
        "postprocessed_prediction_min": result["safe_prediction_min"],
        "postprocessed_prediction_max": result["safe_prediction_max"],
        "postprocessed_out_of_range_count": postprocess_rejections,
        "non_finite_count": non_finite_count,
        "max_predicted_joint_delta_rad": max_delta,
        "recording_frequency_hz": 20.0,
        "recommended_day6_inference_frequency_hz": 20.0,
    }
    (artifact_dir / "safety_summary.json").write_text(
        json.dumps(safety, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"安全汇总：{artifact_dir / 'safety_summary.json'}；未连接机械臂")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/act_day5/policy_best_val.ckpt"))
    parser.add_argument("--config", type=Path, default=Path("configs/act_nero_day5.yaml"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config, args.split, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
