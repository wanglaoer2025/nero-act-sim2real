"""Run NERO ACT dataset/model smoke checks and a small offline overfit test."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.act.nero_dataset import (
    NeroEpisodicDataset,
    compute_norm_stats,
    denormalize_actions,
    valid_episode_paths,
)
from src.act.nero_policy import NeroActPolicy


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    device = config.get("device", "auto")
    config["device"] = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    return config


def _save_artifacts(history: list[dict], image: torch.Tensor, gt: np.ndarray,
                    prediction: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_dir / "sample_image.png", image[0].permute(1, 2, 0).cpu().numpy())
    for key in ("loss", "l1", "kl"):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot([row[key] for row in history])
        axis.set(xlabel="step", ylabel=key, title=f"ACT overfit {key}")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"{key}_loss.png", dpi=140)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(gt[:, 0], label="GT J1")
    axis.plot(prediction[:, 0], label="Pred J1")
    axis.set(xlabel="chunk step", ylabel="rad", title="Denormalized prediction vs GT")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_vs_gt.png", dpi=140)
    plt.close(figure)


def run(config: dict, *, episode_count: int, steps: int, smoke_only: bool = False) -> dict:
    if config["device"] == "cpu":
        torch.set_num_threads(int(config.get("cpu_threads", 4)))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    ids = list(config["overfit_episode_ids"][:episode_count])
    paths = valid_episode_paths(Path(config["manifest"]), ids)
    coordinate_conditioning = bool(config.get("coordinate_conditioning", False))
    joint_target_lead_steps = config.get("joint_target_lead_steps")
    stats = compute_norm_stats(
        paths, coordinate_conditioning, int(config.get("trim_start_steps", 0)),
        config.get("trim_start_steps_by_parent"), joint_target_lead_steps,
    )
    dataset = NeroEpisodicDataset(
        paths, config["camera_names"], stats,
        alignment_offset=config["alignment_offset"], image_size=config["image_size"],
        coordinate_conditioning=coordinate_conditioning,
        joint_lookahead_steps=int(config.get("joint_lookahead_steps", 0)),
        joint_target_lead_steps=joint_target_lead_steps,
    )
    loader = DataLoader(dataset, batch_size=min(config["batch_size"], len(dataset)), shuffle=True, num_workers=0)
    image, qpos, action, is_pad, start_ts = next(iter(loader))
    print(f"episodes：{[path.name for path in paths]}；start_ts：{start_ts.tolist()}")
    print(f"sample image/qpos/action/is_pad：{tuple(image[0].shape)} / {tuple(qpos[0].shape)} / "
          f"{tuple(action[0].shape)} / {tuple(is_pad[0].shape)}")
    print(f"batch image/qpos/action/is_pad：{tuple(image.shape)} / {tuple(qpos.shape)} / "
          f"{tuple(action.shape)} / {tuple(is_pad.shape)}")
    print(f"image dtype/range：{image.dtype} / {image.min().item():.3f}..{image.max().item():.3f}")
    print(f"qpos range：{qpos.min().item():.3f}..{qpos.max().item():.3f}")
    print(f"action range：{action.min().item():.3f}..{action.max().item():.3f}")
    print(f"qpos sample：{qpos[0].round(decimals=4).tolist()}")
    print(f"前 5 个 action：{action[0, :5].round(decimals=4).tolist()}")
    print(f"is_pad 前 20 项：{is_pad[0, :20].tolist()}")
    smoke_dir = Path(config["artifact_dir"])
    smoke_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(smoke_dir / "sample_image.png", image[0, 0].permute(1, 2, 0).numpy())

    device = torch.device(config["device"])
    policy = NeroActPolicy(config).to(device)
    image_b, qpos_b, action_b, pad_b = [value.to(device) for value in (image, qpos, action, is_pad)]
    with torch.no_grad():
        smoke = policy(qpos_b, image_b, action_b, pad_b)
        inference = policy(qpos_b, image_b)
    expected = (len(image), config["chunk_size"], config["action_dim"])
    if tuple(smoke["a_hat"].shape) != expected or tuple(inference.shape) != expected:
        raise RuntimeError(f"ACT output shape 错误：train={smoke['a_hat'].shape}, inference={inference.shape}")
    if not all(torch.isfinite(value).all() for value in (smoke["loss"], inference)):
        raise RuntimeError("ACT forward 出现 NaN/Inf")
    print(f"train/inference output：{tuple(smoke['a_hat'].shape)} / {tuple(inference.shape)}")
    if smoke_only:
        return {"smoke_only": True, "output_shape": list(inference.shape)}

    fixed = NeroEpisodicDataset(
        [paths[0]], config["camera_names"], stats,
        alignment_offset=config["alignment_offset"], image_size=config["image_size"], fixed_start_ts=0,
        coordinate_conditioning=coordinate_conditioning,
        joint_lookahead_steps=int(config.get("joint_lookahead_steps", 0)),
        joint_target_lead_steps=joint_target_lead_steps,
    )[0]
    fixed_image, fixed_qpos, fixed_action, fixed_pad, _ = fixed
    policy.eval()
    with torch.no_grad():
        initial_norm = policy(fixed_qpos[None].to(device), fixed_image[None].to(device))[0]
    initial_real = denormalize_actions(initial_norm.cpu(), stats).numpy()
    gt_real = denormalize_actions(fixed_action[:config["chunk_size"]], stats).numpy()
    initial_prediction_mae = float(np.mean(np.abs(initial_real - gt_real)))

    history = []
    iterator = iter(loader)
    policy.train()
    for step in range(steps):
        try:
            image_b, qpos_b, action_b, pad_b, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image_b, qpos_b, action_b, pad_b, _ = next(iterator)
        image_b, qpos_b, action_b, pad_b = [value.to(device) for value in (image_b, qpos_b, action_b, pad_b)]
        losses = policy(qpos_b, image_b, action_b, pad_b)
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(f"step {step} loss 出现 NaN/Inf")
        policy.optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.1)
        policy.optimizer.step()
        history.append({key: float(losses[key].detach()) for key in ("loss", "l1", "kl")})
        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            print(f"step {step + 1}/{steps}：loss={history[-1]['loss']:.6f} "
                  f"l1={history[-1]['l1']:.6f} kl={history[-1]['kl']:.6f}")

    policy.eval()
    with torch.no_grad():
        pred_norm = policy(fixed_qpos[None].to(device), fixed_image[None].to(device))[0]
    pred_real = denormalize_actions(pred_norm.cpu(), stats).numpy()
    mae = float(np.mean(np.abs(pred_real - gt_real)))
    if not np.isfinite(pred_real).all():
        raise RuntimeError("反归一化 prediction 包含 NaN/Inf")

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"policy_{episode_count}ep_last.ckpt"
    stats_path = checkpoint_dir / f"dataset_stats_{episode_count}ep.pkl"
    torch.save({"model": policy.state_dict(), "config": config, "episode_ids": ids}, checkpoint)
    with stats_path.open("wb") as stream:
        pickle.dump(stats, stream)
    artifact_dir = Path(config["artifact_dir"]) / f"{episode_count}ep"
    _save_artifacts(history, fixed_image, gt_real, pred_real, artifact_dir)
    metrics = {
        "episode_ids": ids, "steps": steps,
        "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
        "initial_l1": history[0]["l1"], "final_l1": history[-1]["l1"],
        "initial_kl": history[0]["kl"], "final_kl": history[-1]["kl"],
        "initial_window_loss": float(np.mean([row["loss"] for row in history[:min(10, len(history))]])),
        "final_window_loss": float(np.mean([row["loss"] for row in history[-min(10, len(history)) :]])),
        "initial_window_l1": float(np.mean([row["l1"] for row in history[:min(10, len(history))]])),
        "final_window_l1": float(np.mean([row["l1"] for row in history[-min(10, len(history)) :]])),
        "initial_prediction_mae_real": initial_prediction_mae,
        "prediction_mae_real": mae,
        "prediction_min": float(pred_real.min()), "prediction_max": float(pred_real.max()),
        "checkpoint": str(checkpoint), "stats": str(stats_path),
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/act_nero.yaml"))
    parser.add_argument("--episode-count", type=int, choices=(1, 3), default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    steps = config["steps"] if args.steps is None else args.steps
    if steps <= 0:
        parser.error("--steps 必须大于 0")
    run(config, episode_count=args.episode_count, steps=steps, smoke_only=args.smoke_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
