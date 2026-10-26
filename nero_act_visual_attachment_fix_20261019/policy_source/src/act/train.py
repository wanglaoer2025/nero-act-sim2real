"""Train the NERO ACT policy on a fixed episode-level Day 5 split."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import pickle
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.act.dataset_stats import load_or_create_split
from src.act.nero_dataset import (
    NeroEpisodicDataset,
    compute_norm_stats,
    episode_effective_end,
    episode_trim_start,
    valid_episode_paths,
)
from src.act.nero_policy import NeroActPolicy


def load_config(path: Path, device_override: str | None = None) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    requested = device_override or config.get("device", "auto")
    config["device"] = "cuda" if requested == "auto" and torch.cuda.is_available() else (
        "cpu" if requested == "auto" else requested
    )
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但 torch.cuda.is_available() 为 false")
    return config


def fixed_window_samples(paths: list[Path], chunk_size: int, *, first_step: int = 0,
                         stride: int = 1,
                         first_steps_by_parent: dict[str, int] | None = None,
                         trim_after_gripper_open_steps: int | None = None,
                         grasp_focus_before_close_steps: int = 0,
                         grasp_focus_repeats: int = 1,
                         post_grasp_repeats: int = 1,
                         ) -> list[tuple[int, int]]:
    """Return every complete action window at a fixed stride."""
    if (chunk_size <= 0 or first_step < 0 or stride <= 0
            or grasp_focus_before_close_steps < 0 or grasp_focus_repeats <= 0
            or post_grasp_repeats <= 0):
        raise ValueError("chunk_size/stride 必须为正，first_step 不得为负")
    samples = []
    for path_index, path in enumerate(paths):
        path_first_step = episode_trim_start(path, first_step, first_steps_by_parent)
        with h5py.File(path, "r") as root:
            end_step = episode_effective_end(root, trim_after_gripper_open_steps)
            last_start = end_step - chunk_size
            close_steps = np.flatnonzero(root["action"][:end_step, 7] < 0.0475)
        if last_start < path_first_step:
            raise ValueError(f"{path} 在裁剪后不足一个完整 action window")
        episode_samples = [
            (path_index, start)
            for start in range(path_first_step, last_start + 1, stride)
        ]
        samples.extend(episode_samples)
        if grasp_focus_before_close_steps and grasp_focus_repeats > 1:
            if not len(close_steps):
                raise ValueError(f"{path} 没有闭爪动作，无法加权抓取前窗口")
            focus_start = max(
                path_first_step, int(close_steps[0]) - grasp_focus_before_close_steps
            )
            focused = [
                sample for sample in episode_samples
                if focus_start <= sample[1] <= close_steps[0]
            ]
            samples.extend(focused * (grasp_focus_repeats - 1))
        if post_grasp_repeats > 1:
            if not len(close_steps):
                raise ValueError(f"{path} 没有闭爪动作，无法加权抓取后窗口")
            post_grasp = [
                sample for sample in episode_samples if sample[1] > close_steps[0]
            ]
            samples.extend(post_grasp * (post_grasp_repeats - 1))
    return samples


def fixed_validation_samples(paths: list[Path], samples_per_episode: int,
                             chunk_size: int, first_step: int = 0,
                             first_steps_by_parent: dict[str, int] | None = None,
                             trim_after_gripper_open_steps: int | None = None,
                             ) -> tuple[list[tuple[int, int]], list[dict]]:
    samples, records = [], []
    for path_index, path in enumerate(paths):
        path_first_step = episode_trim_start(path, first_step, first_steps_by_parent)
        with h5py.File(path, "r") as root:
            length = episode_effective_end(root, trim_after_gripper_open_steps)
        last_full_chunk = max(0, length - chunk_size)
        if last_full_chunk < path_first_step:
            raise ValueError(f"{path} 在裁剪后不足一个完整 action window")
        for start in np.linspace(
            path_first_step, last_full_chunk, samples_per_episode, dtype=int
        ):
            samples.append((path_index, int(start)))
            records.append({"episode_id": int(path.stem.removeprefix("episode_")), "start_ts": int(start)})
    return samples, records


def validate_episode_groups(paths: list[Path], metadata_key: str) -> dict[str, int]:
    """Require the same number of trajectories for every task-position group."""
    groups = []
    for path in paths:
        with h5py.File(path, "r") as root:
            value = root.attrs.get(metadata_key)
        if value is None or str(value).strip() == "":
            raise ValueError(f"{path} 缺少分组属性 {metadata_key}")
        groups.append(str(value))
    counts = dict(sorted(Counter(groups).items()))
    if len(counts) < 2 or len(set(counts.values())) != 1:
        raise ValueError(f"各位置轨迹数量必须相同，当前为 {counts}")
    return counts


def validate_fixed_starts(paths: list[Path], expected_degrees: list[float],
                          tolerance_rad: float) -> None:
    expected = np.radians(np.asarray(expected_degrees, dtype=np.float32))
    if expected.shape != (7,) or tolerance_rad <= 0:
        raise ValueError("fixed_start_joints_deg 必须为 7 维，容差必须为正")
    failures = []
    for path in paths:
        with h5py.File(path, "r") as root:
            actual = root["observations/qpos"][0]
        error = float(np.max(np.abs(actual - expected)))
        if error > tolerance_rad:
            failures.append(f"{path.name}={error:.4f}")
    if failures:
        raise ValueError("episode 未从统一预备位姿开始：" + ", ".join(failures))


def run_epoch(policy: NeroActPolicy, loader: DataLoader, device: torch.device,
              *, training: bool) -> dict[str, float]:
    policy.train(training)
    totals = {"loss": 0.0, "l1": 0.0, "kl": 0.0, "gripper_bce": 0.0}
    samples = 0
    for image, qpos, action, is_pad, _ in loader:
        image, qpos, action, is_pad = [value.to(device) for value in (image, qpos, action, is_pad)]
        with torch.set_grad_enabled(training):
            losses = policy(qpos, image, action, is_pad)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("训练出现 NaN/Inf；请检查 episode、normalization 与 padding")
            if training:
                policy.optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.1)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("gradient 出现 NaN/Inf")
                policy.optimizer.step()
        batch = len(image)
        samples += batch
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch
    return {key: value / samples for key, value in totals.items()}


def save_checkpoint(path: Path, policy: NeroActPolicy, config: dict, split: dict,
                    epoch: int, metrics: dict) -> None:
    torch.save({
        "model": policy.state_dict(), "optimizer": policy.optimizer.state_dict(),
        "config": config, "split": split, "epoch": epoch, "metrics": metrics,
    }, path)


def save_curves(history: list[dict], output_dir: Path) -> None:
    for metric in ("loss", "l1", "kl", "gripper_bce"):
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot([row[f"train_{metric}"] for row in history], label="train")
        axis.plot([row[f"val_{metric}"] for row in history], label="val")
        axis.set(xlabel="epoch", ylabel=metric, title=f"ACT train/val {metric}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"train_val_{metric}.png", dpi=140)
        plt.close(figure)


def train(config: dict) -> dict:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if config["device"] == "cpu":
        torch.set_num_threads(int(config.get("cpu_threads", 4)))
    group_key = config.get("position_group_key")
    split = load_or_create_split(
        Path(config["manifest"]), Path(config["split_file"]), seed=seed,
        val_fraction=float(config["val_fraction"]), group_key=group_key,
    )
    train_paths = valid_episode_paths(Path(config["manifest"]), split["train"])
    val_paths = valid_episode_paths(Path(config["manifest"]), split["val"])
    all_paths = train_paths + val_paths
    if group_key:
        validate_episode_groups(all_paths, str(group_key))
    if config.get("fixed_start_joints_deg") is not None:
        validate_fixed_starts(
            all_paths, config["fixed_start_joints_deg"],
            float(config.get("fixed_start_tolerance_rad", 0.03)),
        )
    trim_start_steps = int(config.get("trim_start_steps", 0))
    trim_by_parent = config.get("trim_start_steps_by_parent")
    training_stride_steps = int(config.get("training_stride_steps", 1))
    coordinate_conditioning = bool(config.get("coordinate_conditioning", False))
    joint_target_lead_steps = config.get("joint_target_lead_steps")
    trim_after_open = config.get("trim_after_gripper_open_steps")
    stats_path = config.get("normalization_stats")
    if stats_path:
        with Path(stats_path).open("rb") as stream:
            stats = pickle.load(stream)
        for key, size in (
            ("qpos_mean", int(config["state_dim"])),
            ("qpos_std", int(config["state_dim"])),
            ("action_mean", int(config["action_dim"])),
            ("action_std", int(config["action_dim"])),
        ):
            values = np.asarray(stats.get(key), dtype=np.float32)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError(f"normalization {key} 必须是有限 float32[{size}]")
            if key.endswith("std") and np.any(values <= 0):
                raise ValueError(f"normalization {key} 必须大于 0")
            stats[key] = values
    else:
        stats = compute_norm_stats(
            train_paths, coordinate_conditioning, trim_start_steps, trim_by_parent,
            joint_target_lead_steps, trim_after_open,
            previous_gripper_conditioning=bool(config.get('previous_gripper_conditioning',False)),
        )
    if config.get('task_displacement_features',False):
        for key,saved in [('qpos_mean','kinematic_state_mean'),('qpos_std','kinematic_state_std')]:
            if not np.array_equal(stats[key],np.asarray(config[saved],dtype=np.float32)):
                raise ValueError('Analytic geometry statistics differ from training observation normalization')
    train_starts = fixed_window_samples(
        train_paths, int(config["chunk_size"]), first_step=trim_start_steps,
        stride=training_stride_steps, first_steps_by_parent=trim_by_parent,
        trim_after_gripper_open_steps=trim_after_open,
        grasp_focus_before_close_steps=int(config.get("grasp_focus_before_close_steps", 0)),
        grasp_focus_repeats=int(config.get("grasp_focus_repeats", 1)),
        post_grasp_repeats=int(config.get("post_grasp_repeats", 1)),
    )
    train_dataset = NeroEpisodicDataset(
        train_paths, config["camera_names"], stats,
        alignment_offset=config["alignment_offset"], image_size=config["image_size"],
        sample_starts=train_starts,
        coordinate_conditioning=coordinate_conditioning,
        previous_gripper_conditioning=bool(config.get('previous_gripper_conditioning',False)),
        joint_lookahead_steps=int(config.get("joint_lookahead_steps", 0)),
        joint_target_lead_steps=joint_target_lead_steps,
        trim_after_gripper_open_steps=trim_after_open,
    )
    val_starts, val_records = fixed_validation_samples(
        val_paths, int(config["fixed_val_samples_per_episode"]), int(config["chunk_size"]),
        trim_start_steps, trim_by_parent, trim_after_open,
    )
    val_dataset = NeroEpisodicDataset(
        val_paths, config["camera_names"], stats,
        alignment_offset=config["alignment_offset"], image_size=config["image_size"],
        sample_starts=val_starts,
        coordinate_conditioning=coordinate_conditioning,
        previous_gripper_conditioning=bool(config.get('previous_gripper_conditioning',False)),
        joint_lookahead_steps=int(config.get("joint_lookahead_steps", 0)),
        joint_target_lead_steps=joint_target_lead_steps,
        trim_after_gripper_open_steps=trim_after_open,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(config["eval_batch_size"]), shuffle=False, num_workers=0,
    )

    checkpoint_dir = Path(config["checkpoint_dir"])
    artifact_dir = Path(config["artifact_dir"])
    log_path = Path(config["training_log"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "dataset_stats.pkl").open("wb") as stream:
        pickle.dump(stats, stream)
    (checkpoint_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    split_text = json.dumps(split, indent=2)
    (checkpoint_dir / "split.json").write_text(split_text, encoding="utf-8")
    (artifact_dir / "fixed_val_samples.json").write_text(
        json.dumps(val_records, indent=2), encoding="utf-8"
    )

    device = torch.device(config["device"])
    policy = NeroActPolicy(config).to(device)
    if config.get("init_checkpoint"):
        payload = torch.load(
            Path(config["init_checkpoint"]), map_location=device, weights_only=False,
        )
        previous = payload.get("config", {})
        for key in (
            "state_dim", "action_dim", "chunk_size", "camera_names", "image_size",
            "hidden_dim", "dim_feedforward", "nheads", "enc_layers", "dec_layers",
            "backbone",
        ):
            if previous.get(key) != config.get(key):
                raise ValueError(
                    f"init checkpoint {key}={previous.get(key)!r} 与训练配置 "
                    f"{config.get(key)!r} 不匹配"
                )
        policy.load_state_dict(payload["model"], strict=True)
        print(f"已加载初始化权重：{config['init_checkpoint']}")
    history = []
    best_val_loss = best_val_l1 = float("inf")
    best_epoch = best_l1_epoch = -1
    started = time.monotonic()
    fields = [
        "epoch", "train_loss", "train_l1", "train_kl", "train_gripper_bce",
        "val_loss", "val_l1", "val_kl", "val_gripper_bce", "lr", "elapsed_s",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, int(config["num_epochs"]) + 1):
            train_metrics = run_epoch(policy, train_loader, device, training=True)
            fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(seed)
                val_metrics = run_epoch(policy, val_loader, device, training=False)
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "lr": policy.optimizer.param_groups[0]["lr"],
                "elapsed_s": time.monotonic() - started,
            }
            history.append(row)
            writer.writerow(row)
            stream.flush()
            if row["val_loss"] < best_val_loss:
                best_val_loss, best_epoch = row["val_loss"], epoch
                save_checkpoint(checkpoint_dir / "policy_best_val.ckpt", policy, config, split, epoch, row)
            if row["val_l1"] < best_val_l1:
                best_val_l1, best_l1_epoch = row["val_l1"], epoch
                save_checkpoint(checkpoint_dir / "policy_best_l1.ckpt", policy, config, split, epoch, row)
            if epoch == 1 or epoch % max(1, int(config["num_epochs"]) // 20) == 0:
                print(f"epoch {epoch}/{config['num_epochs']} train={row['train_loss']:.4f}/{row['train_l1']:.4f} "
                      f"val={row['val_loss']:.4f}/{row['val_l1']:.4f} "
                      f"best_loss={best_val_loss:.4f}@{best_epoch} "
                      f"best_l1={best_val_l1:.4f}@{best_l1_epoch}")

    save_checkpoint(checkpoint_dir / "policy_last.ckpt", policy, config, split,
                    int(config["num_epochs"]), history[-1])
    save_curves(history, artifact_dir)
    summary = {
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "best_l1_epoch": best_l1_epoch, "best_val_l1": best_val_l1,
        "best_metrics": history[best_epoch - 1], "last_metrics": history[-1],
        "best_l1_metrics": history[best_l1_epoch - 1],
        "train_episodes": split["train"], "val_episodes": split["val"],
        "train_windows": len(train_starts),
        "training_stride_steps": training_stride_steps,
        "trim_start_steps": trim_start_steps,
        "trim_start_steps_by_parent": trim_by_parent,
        "trim_after_gripper_open_steps": trim_after_open,
        "grasp_focus_before_close_steps": int(config.get("grasp_focus_before_close_steps", 0)),
        "grasp_focus_repeats": int(config.get("grasp_focus_repeats", 1)),
        "post_grasp_repeats": int(config.get("post_grasp_repeats", 1)),
        "init_checkpoint": config.get("init_checkpoint"),
        "normalization_stats": config.get("normalization_stats"),
        "joint_target_lead_steps": joint_target_lead_steps,
        "elapsed_s": time.monotonic() - started,
    }
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/act_nero_day5.yaml"))
    parser.add_argument("--epochs", type=int, help="覆盖配置中的 epoch 数，仅用于短测试")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config, args.device)
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error("--epochs 必须大于 0")
        config["num_epochs"] = args.epochs
    print(
        f"ACT Day5：device={config['device']} epochs={config['num_epochs']} "
        f"joint target lead={config.get('joint_target_lead_steps', 'recorded action')} steps；"
        f"trim={config.get('trim_start_steps', 0)} stride={config.get('training_stride_steps', 1)}；"
        "不会连接机械臂"
    )
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
