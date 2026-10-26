"""Generate a dual-camera ACT demonstration with the YOLO pick-place policy."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from src.act.episode_io import build_leader_actions, save_episode
from src.act.record_episode import ActRgbCameras, _episode_metrics
from src.utils.arm import confirm_action


DEFAULT_SECONDARY_CAMERA = (
    "/dev/v4l/by-id/"
    "usb-DHZJ-251231-J_DICOTA_4K_01.00.00-video-index0"
)


class YoloEpisodeRecorder:
    """Sample robot feedback and dual RGB while an existing policy moves the arm."""

    def __init__(self, output: Path, *, frequency: float = 20.0,
                 action_lead_ms: float = 0.0,
                 secondary_camera_device: str | None = DEFAULT_SECONDARY_CAMERA,
                 secondary_camera_rotation_degrees: int = 0,
                 metadata: dict | None = None, camera=None) -> None:
        if output.exists():
            raise FileExistsError(f"输出已存在：{output}")
        if not 0 < frequency <= 20 or action_lead_ms < 0:
            raise ValueError("frequency 必须在 (0,20] Hz，action_lead_ms 不得为负")
        self.output = output
        self.frequency = float(frequency)
        self.lead_steps = round(action_lead_ms / 1000.0 * frequency)
        self.camera = camera or ActRgbCameras(
            640, secondary_camera_device, secondary_camera_rotation_degrees
        )
        self.metadata = metadata or {}
        self.robot = self.gripper = None
        self.gripper_target_m = 0.095
        self.images: dict[str, list[np.ndarray]] = {}
        self.qpos: list[np.ndarray] = []
        self.gripper_states: list[np.ndarray] = []
        self.gripper_targets: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.skews: list[float] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: Exception | None = None
        self.finished = False

    def start(self, robot, gripper, initial_gripper_target_m: float) -> None:
        self.robot, self.gripper = robot, gripper
        self.gripper_target_m = float(initial_gripper_target_m)
        self.camera.open()
        self.images = {name: [] for name in self.camera.cameras}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"ACT 自动录制开始：{self.output}；相机={list(self.images)}")

    def set_gripper_target(self, target_m: float) -> None:
        self.gripper_target_m = float(target_m)

    def _run(self) -> None:
        period = 1.0 / self.frequency
        next_tick = time.monotonic()
        try:
            while not self.stop_event.is_set():
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self.stop_event.wait(delay)
                if self.stop_event.is_set():
                    break
                next_tick += period
                frames = self.camera.capture_rgb()
                joints = self.robot.get_joint_angles()
                gripper = self.gripper.get_gripper_status()
                if joints is None or gripper is None or gripper.msg.mode != "width":
                    continue
                qpos = np.asarray(joints.msg, dtype=np.float32)
                width = np.asarray([gripper.msg.value], dtype=np.float32)
                if qpos.shape != (7,) or not np.isfinite(qpos).all() or not np.isfinite(width).all():
                    continue
                for name, frame in frames.items():
                    self.images[name].append(frame)
                self.qpos.append(qpos)
                self.gripper_states.append(width)
                self.gripper_targets.append(
                    np.asarray([self.gripper_target_m], dtype=np.float32)
                )
                self.timestamps.append(time.monotonic())
                self.skews.append(self.camera.last_skew_s)
        except Exception as exc:
            self.error = exc
            self.stop_event.set()

    def finish(self, *, save: bool) -> Path | None:
        if self.finished:
            return self.output if save and self.output.exists() else None
        self.finished = True
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.camera.close()
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("ACT 录制线程未能停止")
        if self.error is not None:
            raise RuntimeError(f"ACT 后台录制失败：{self.error}") from self.error
        if not save:
            print("ACT 自动录制已丢弃：YOLO 搬运未完整成功。")
            return None
        if len(self.timestamps) < max(10, self.lead_steps + 1):
            raise RuntimeError(f"ACT 有效帧不足：{len(self.timestamps)}")

        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        qpos = np.stack(self.qpos)
        gripper_states = np.stack(self.gripper_states)
        targets = np.stack(self.gripper_targets)
        images = {name: np.stack(values) for name, values in self.images.items()}
        actions = build_leader_actions(qpos, targets, self.lead_steps)
        metrics = _episode_metrics(timestamps, self.frequency)
        save_episode(
            self.output, images, qpos, actions, timestamps,
            gripper_states=gripper_states,
            metadata={
                **self.metadata,
                "control_frequency_hz": self.frequency,
                "demonstration_mode": "yolo_teacher_policy",
                "action_semantics": "future_measured_qpos_plus_gripper_target",
                "action_lead_steps": self.lead_steps,
                "gripper_action_lead_steps": 0,
                "image_format": "RGB",
                "camera_pair_skew_mean_ms": float(np.mean(self.skews) * 1000),
                "camera_pair_skew_p95_ms": float(np.percentile(self.skews, 95) * 1000),
                "camera_pair_skew_max_ms": float(np.max(self.skews) * 1000),
                **metrics,
            },
        )
        print(f"ACT 自动示教已保存：{self.output}；frames={len(timestamps)}；"
              f"Hz={metrics['actual_frequency_hz']:.2f}")
        return self.output


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/act_dual_fixed"))
    parser.add_argument(
        "--position-group", required=True, choices=("1", "2", "3"),
        help="杯子位置编号；三个编号必须采集相同数量的成功轨迹",
    )
    parser.add_argument("--weights", default="/home/user/projects/yolo/best.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--detection-timeout-s", type=float, default=30.0)
    parser.add_argument("--source-offset-mm", nargs=3, type=float, default=(0, -17, 0))
    parser.add_argument("--destination-offset-mm", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--workspace-mm", nargs=6, type=float,
                        default=(-650, 0, -300, 300, 0, 500))
    parser.add_argument("--transit-speed-percent", type=int, default=30)
    parser.add_argument("--precision-speed-percent", type=int, default=5)
    parser.add_argument("--cup-height-mm", type=float, default=0.0)
    parser.add_argument("--secondary-camera-device", default=DEFAULT_SECONDARY_CAMERA)
    parser.add_argument("--secondary-camera-rotation-deg", type=int,
                        choices=(0, 90, 180, 270), default=0)
    parser.add_argument(
        "--action-lead-ms", type=float, default=0.0,
        help="保持 0；训练阶段只对七个关节统一前视 8 帧",
    )
    args = parser.parse_args(argv)
    if args.episode_id < 0:
        parser.error("episode-id 不得为负")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output = args.output_dir / f"episode_{args.episode_id}.hdf5"
    if output.exists():
        print(f"拒绝覆盖已有 episode：{output}")
        return 1
    if not confirm_action(
        "YOLO 自动执行并录制 ACT 示教",
        "确认杯子、蓝卡、双相机、工作空间和实体急停均已就绪；确认后整条轨迹不会暂停。",
        "AUTO_RECORD",
    ):
        return 0

    from src.demo.yolo_pick_place import main as run_yolo

    forwarded = [
        "--weights", args.weights,
        "--source-class", "cup", "--destination-class", "blue", "--pick-place",
        "--auto-lock", "--detection-timeout-s", str(args.detection_timeout_s),
        "--stable-frames", str(args.stable_frames), "--confidence", str(args.confidence),
        "--workspace-mm", *map(str, args.workspace_mm),
        "--source-offset-mm", *map(str, args.source_offset_mm),
        "--destination-offset-mm", *map(str, args.destination_offset_mm),
        "--transit-speed-percent", str(args.transit_speed_percent),
        "--precision-speed-percent", str(args.precision_speed_percent),
        "--act-output", str(output), "--act-frequency", "20",
        "--act-action-lead-ms", str(args.action_lead_ms),
        "--act-position-group", args.position_group,
        "--act-secondary-camera-device", args.secondary_camera_device,
        "--act-secondary-camera-rotation-deg", str(args.secondary_camera_rotation_deg),
        "--cup-height-mm", str(args.cup_height_mm),
    ]
    try:
        return run_yolo(forwarded, preconfirmed=True)
    except Exception as exc:
        print(f"YOLO 自动示教失败，未生成可用 episode：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
