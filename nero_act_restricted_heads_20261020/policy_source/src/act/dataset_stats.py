"""Create the fixed Day 5 episode split and summarize valid NERO data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from src.act.nero_dataset import episode_effective_end, episode_trim_start, valid_episode_paths


def episode_id(path: Path) -> int:
    return int(path.stem.removeprefix("episode_"))


def load_or_create_split(manifest: Path, output: Path, *, seed: int = 42,
                         val_fraction: float = 0.2, force: bool = False,
                         group_key: str | None = None) -> dict:
    valid_paths = valid_episode_paths(manifest)
    valid_ids = [episode_id(path) for path in valid_paths]
    if output.exists() and not force:
        split = json.loads(output.read_text(encoding="utf-8"))
    else:
        rng = np.random.default_rng(seed)
        if group_key:
            grouped: dict[str, list[int]] = {}
            for path in valid_paths:
                with h5py.File(path, "r") as root:
                    value = root.attrs.get(group_key)
                if value is None:
                    raise ValueError(f"{path} 缺少分组属性 {group_key}")
                grouped.setdefault(str(value), []).append(episode_id(path))
            train_ids, val_ids = [], []
            for name, ids in sorted(grouped.items()):
                if len(ids) < 2:
                    raise ValueError(f"分组 {name!r} 至少需要 2 条轨迹才能划分 train/val")
                shuffled = rng.permutation(ids).tolist()
                val_count = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
                val_ids.extend(shuffled[:val_count])
                train_ids.extend(shuffled[val_count:])
        else:
            shuffled = rng.permutation(valid_ids).tolist()
            val_count = max(1, round(len(shuffled) * val_fraction))
            val_ids, train_ids = shuffled[:val_count], shuffled[val_count:]
        split = {
            "seed": seed,
            "group_key": group_key,
            "train": sorted(train_ids),
            "val": sorted(val_ids),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(split, indent=2), encoding="utf-8")
    train, val = set(split["train"]), set(split["val"])
    if train & val or train | val != set(valid_ids) or not train or not val:
        raise ValueError("split 必须无重叠地覆盖 manifest 中全部 valid episodes")
    return split


def _range_summary(values: list[np.ndarray]) -> dict:
    merged = np.concatenate(values, axis=0)
    return {"min": merged.min(axis=0).tolist(), "max": merged.max(axis=0).tolist()}


def build_summary(manifest_path: Path, split: dict, trim_start_steps: int = 0,
                  trim_start_steps_by_parent: dict[str, int] | None = None,
                  trim_after_gripper_open_steps: int | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid = [row for row in manifest["episodes"] if row["status"] == "valid"]
    rejected = [row for row in manifest["episodes"] if row["status"] != "valid"]
    lengths = np.asarray([row["num_steps"] for row in valid])
    durations = np.asarray([row["duration_s"] for row in valid])
    frequencies = np.asarray([row["actual_hz"] for row in valid])
    qpos, action, gripper = [], [], []
    for path in valid_episode_paths(manifest_path, split["train"]):
        first_step = episode_trim_start(
            path, trim_start_steps, trim_start_steps_by_parent
        )
        with h5py.File(path, "r") as root:
            end_step = episode_effective_end(root, trim_after_gripper_open_steps)
            if end_step <= first_step:
                raise ValueError(f"{path} 裁剪后为空")
            qpos.append(root["observations/qpos"][first_step:end_step])
            action.append(root["action"][first_step:end_step])
            gripper.append(root["observations/gripper_state"][first_step:end_step])
    describe = lambda values: {
        "min": float(values.min()), "mean": float(values.mean()), "max": float(values.max())
    }
    return {
        "valid_episode_count": len(valid),
        "rejected_episode_count": len(rejected),
        "rejected": [{"path": row["path"], "reasons": row.get("reasons", [])} for row in rejected],
        "train_episode_count": len(split["train"]),
        "val_episode_count": len(split["val"]),
        "episode_length": describe(lengths),
        "duration_s": describe(durations),
        "actual_frequency_hz": describe(frequencies),
        "train_qpos": _range_summary(qpos),
        "train_action": _range_summary(action),
        "train_gripper_state": _range_summary(gripper),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/act_nero_day5.yaml"))
    parser.add_argument("--force-split", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split = load_or_create_split(
        Path(config["manifest"]), Path(config["split_file"]), seed=int(config["seed"]),
        val_fraction=float(config["val_fraction"]), force=args.force_split,
        group_key=config.get("position_group_key"),
    )
    summary = build_summary(
        Path(config["manifest"]), split, int(config.get("trim_start_steps", 0)),
        config.get("trim_start_steps_by_parent"),
        config.get("trim_after_gripper_open_steps"),
    )
    output = Path(config["artifact_dir"]) / "dataset_stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"split": split, **summary}, indent=2, ensure_ascii=False))
    print(f"数据统计：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
