"""Plot qpos, actions, and same-cycle tracking error for one ACT episode."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from src.act.episode_io import load_episode


def plot_episode(path: Path, output_dir: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nero-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode = load_episode(path)
    qpos, actions, timestamps = episode["qpos"], episode["actions"], episode["timestamps"]
    time_s = timestamps - timestamps[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    outputs = []

    for suffix, values, ylabel, labels in (
        ("qpos", qpos, "joint angle (rad)", [f"J{i}" for i in range(1, 8)]),
        ("action", actions, "target (rad / m)", [*[f"J{i}" for i in range(1, 8)], "gripper"]),
        ("tracking_error", actions[:, :7] - qpos, "action - qpos (rad)", [f"J{i}" for i in range(1, 8)]),
    ):
        figure, axis = plt.subplots(figsize=(11, 6))
        for index, label in enumerate(labels):
            axis.plot(time_s, values[:, index], label=label, linewidth=1)
        axis.set(xlabel="time (s)", ylabel=ylabel, title=f"{stem} - {suffix}")
        axis.grid(alpha=0.25)
        axis.legend(ncol=4, fontsize=8)
        figure.tight_layout()
        output = output_dir / f"{stem}_{suffix}.png"
        figure.savefig(output, dpi=140)
        plt.close(figure)
        outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/act_day3"))
    args = parser.parse_args()
    try:
        outputs = plot_episode(args.episode, args.output_dir)
    except (OSError, KeyError, ValueError) as exc:
        print(f"绘图失败：{exc}")
        return 1
    print("已生成：")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
