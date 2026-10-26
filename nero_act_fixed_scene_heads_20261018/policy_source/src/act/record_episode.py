"""Record one kinesthetic ACT demonstration from NERO and a ZED camera.

The operator drags the arm in Leader mode. At 20 Hz we store actual arm qpos,
the SDK's distinct Leader target feedback, gripper state/target, and RGB.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

from src.act.episode_io import build_leader_actions, save_episode, validate_episode
from src.act.robot_wrapper import NeroRobotWrapper
from src.utils.arm import confirm_action
from src.utils.vision import rotate_image_180


DEFAULT_FREQUENCY_HZ = 20.0
MAX_FREQUENCY_HZ = 20.0


class ZedRgbCamera:
    """Small ZED wrapper returning upright, contiguous RGB frames."""

    def __init__(self, width: int = 640) -> None:
        self.width = width
        self.camera = sl.Camera()
        self.image = sl.Mat()
        self.runtime = sl.RuntimeParameters()
        self.resolution: sl.Resolution | None = None

    def open(self) -> None:
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.NONE
        status = self.camera.open(init)
        if status > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED 打开失败：{status}")

        info = self.camera.get_camera_information()
        native = info.camera_configuration.resolution
        height = max(1, round(self.width * native.height / native.width))
        self.resolution = sl.Resolution(self.width, height)

    def capture_rgb(self) -> np.ndarray:
        if self.resolution is None:
            raise RuntimeError("ZED 尚未打开")
        status = self.camera.grab(self.runtime)
        # ZED warnings compare below SUCCESS and still provide a readable frame.
        if status > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED 取帧失败：{status}")
        self.camera.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU, self.resolution)
        frame = np.asarray(self.image.get_data()).copy()
        frame = rotate_image_180(frame)
        if frame.ndim != 3:
            raise RuntimeError(f"相机图像维度异常：{frame.shape}")
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"相机通道数异常：{frame.shape[2]}")
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def close(self) -> None:
        self.camera.close()


class UsbRgbCamera:
    """OpenCV USB camera normalized to the same frame size as the ZED stream."""

    def __init__(self, device: str, width: int = 640, rotation_degrees: int = 0) -> None:
        self.device = device
        self.width = width
        self.height = round(width * 9 / 16)
        self.rotation_degrees = rotation_degrees
        self.camera: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.camera = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            raise RuntimeError(f"USB 相机打开失败：{self.device}")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            self.camera.read()

    def capture_rgb(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("USB 相机尚未打开")
        ok, frame = self.camera.read()
        if not ok or frame is None:
            raise RuntimeError(f"USB 相机取帧失败：{self.device}")
        rotations = {
            0: None,
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        code = rotations[self.rotation_degrees]
        if code is not None:
            frame = cv2.rotate(frame, code)
        scale = min(self.width / frame.shape[1], self.height / frame.shape[0])
        resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        y = (self.height - resized.shape[0]) // 2
        x = (self.width - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None


class LatestFrameCamera:
    """Read one blocking camera continuously so multiple cameras run in parallel."""

    def __init__(self, camera) -> None:
        self.camera = camera
        self.frame: np.ndarray | None = None
        self.timestamp = 0.0
        self.error: Exception | None = None
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def open(self) -> None:
        self.camera.open()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.latest(timeout=5.0)

    def _capture_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                frame = self.camera.capture_rgb()
                with self.condition:
                    self.frame = frame
                    self.timestamp = time.monotonic()
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def latest(self, timeout: float = 0.2) -> tuple[np.ndarray, float]:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.frame is None and self.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("等待相机首帧超时")
                self.condition.wait(remaining)
            if self.error is not None:
                raise RuntimeError(f"后台相机取帧失败：{self.error}") from self.error
            if time.monotonic() - self.timestamp > 0.2:
                raise RuntimeError("相机最新帧已超过 200 ms，拒绝写入过期图像")
            return self.frame, self.timestamp

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.camera.close()


class ActRgbCameras:
    """One synchronized-enough capture point for ACT's fixed camera list."""

    def __init__(self, width: int, secondary_device: str | None = None,
                 secondary_rotation_degrees: int = 0) -> None:
        self.cameras = {"camera_0": LatestFrameCamera(ZedRgbCamera(width))}
        if secondary_device:
            self.cameras["camera_1"] = LatestFrameCamera(
                UsbRgbCamera(secondary_device, width, secondary_rotation_degrees)
            )
        self.last_skew_s = 0.0

    def open(self) -> None:
        opened = []
        try:
            for camera in self.cameras.values():
                camera.open()
                opened.append(camera)
        except Exception:
            for camera in opened:
                camera.close()
            raise

    def capture_rgb(self) -> dict[str, np.ndarray]:
        samples = {name: camera.latest() for name, camera in self.cameras.items()}
        stamps = [stamp for _, stamp in samples.values()]
        self.last_skew_s = max(stamps) - min(stamps)
        return {name: frame for name, (frame, _) in samples.items()}

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="录制一条 NERO ACT 示教轨迹")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="仅检查相机与内存数据格式")
    mode.add_argument("--execute", action="store_true", help="连接机械臂并录制真实示教")
    parser.add_argument("--output-dir", type=Path, default=Path("data/act"))
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--task-name", default="pick_place")
    parser.add_argument("--position-group", choices=("1", "2", "3"),
                        help="可选的杯子位置编号，用于均衡训练集")
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument(
        "--action-lead-ms", type=float, default=0.0,
        help="实验性离线前视；默认 0，保留同周期原始 Leader action",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument(
        "--secondary-camera-device",
        help="第二相机设备，建议使用 /dev/v4l/by-id/...-video-index0 稳定路径",
    )
    parser.add_argument(
        "--secondary-camera-rotation-deg", type=int, choices=(0, 90, 180, 270), default=0,
        help="第二相机顺时针旋转角度",
    )
    parser.add_argument("--gripper-open-mm", type=float, default=95.0)
    parser.add_argument("--gripper-close-mm", type=float, default=0.0)
    parser.add_argument("--gripper-force-n", type=float, default=1.0)
    parser.add_argument(
        "--gripper-hold-s", type=float, default=1.0,
        help="夹爪动作时保持 Follower 的时间；期间请停止拖动",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        args.dry_run = True
    if not 0 < args.frequency <= MAX_FREQUENCY_HZ:
        parser.error(f"--frequency 必须在 (0, {MAX_FREQUENCY_HZ}] Hz 内")
    if args.max_duration <= 0 or args.action_lead_ms < 0:
        parser.error("--max-duration 必须大于 0，--action-lead-ms 不得为负")
    if args.episode_id < 0 or args.camera_width <= 0:
        parser.error("--episode-id 不得为负，--camera-width 必须大于 0")
    if not 0 <= args.gripper_close_mm <= args.gripper_open_mm <= 100:
        parser.error("夹爪宽度必须满足 0 <= close <= open <= 100 mm")
    if args.gripper_force_n <= 0 or args.gripper_hold_s <= 0:
        parser.error("--gripper-force-n 和 --gripper-hold-s 必须大于 0")
    return args


def _show(frames_rgb: dict[str, np.ndarray], state: str, detail: str) -> int:
    frames = []
    for name, frame_rgb in frames_rgb.items():
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, name, (18, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        frames.append(frame)
    frame = np.hstack(frames)
    cv2.putText(frame, state, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, detail, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.imshow("ACT episode recorder", frame)
    return cv2.waitKey(1) & 0xFF


def _preview(camera: ActRgbCameras, duration_s: float, frequency: float,
             *, show=_show, clock=time.monotonic, sleep=time.sleep) -> int:
    deadline = clock() + duration_s
    frames_seen = 0
    while clock() < deadline:
        frames = camera.capture_rgb()
        frames_seen += 1
        if show(frames, "DRY RUN", "q/Esc: close preview") in (ord("q"), 27):
            break
        sleep(min(1.0 / frequency, max(0.0, deadline - clock())))
    return frames_seen


def _print_plan(args: argparse.Namespace, output: Path, lead_steps: int) -> None:
    print("\nACT Day2 轨迹录制计划")
    print(f"  模式：{'实机录制' if args.execute else 'dry-run（不连接机械臂）'}")
    print(f"  输出：{output}")
    print(f"  采样频率：{args.frequency:.1f} Hz")
    print(f"  最长时长：{args.max_duration:.1f} s")
    print(f"  action 前视：{lead_steps} 帧（约 {lead_steps / args.frequency * 1000:.0f} ms）")
    print("  相机：camera_0=ZED LEFT" + (
        f"；camera_1={args.secondary_camera_device}，旋转 {args.secondary_camera_rotation_deg}°"
        if args.secondary_camera_device else "（单相机）"
    ))


def _check_output_location(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出已存在：{output}；更换 --episode-id 或显式使用 --overwrite")
    parent = output.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise PermissionError(f"输出目录不可写：{output.parent}")
    print(f"  输出目录检查：可写（现有父目录 {parent}）")


def _dry_run(args: argparse.Namespace, camera: ActRgbCameras, lead_steps: int) -> int:
    count = max(40, lead_steps + 2)
    captured = {name: [] for name in camera.cameras}
    timestamps = []
    period = 1.0 / args.frequency
    next_tick = time.monotonic()
    for _ in range(count):
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        next_tick += period
        frames = camera.capture_rgb()
        timestamps.append(time.monotonic())
        for name, frame in frames.items():
            captured[name].append(frame)
    qpos = np.zeros((count, 7), dtype=np.float32)
    gripper = np.full((count, 1), args.gripper_open_mm / 1000.0, dtype=np.float32)
    actions = build_leader_actions(qpos, gripper, lead_steps)
    images = {name: np.stack(values) for name, values in captured.items()}
    timestamps = np.asarray(timestamps, dtype=np.float64)
    validate_episode(images, qpos, actions, timestamps, gripper)
    metrics = _episode_metrics(timestamps, args.frequency)
    print("\ndry-run 检查通过")
    for name, values in images.items():
        print(f"  {name}：shape={values.shape}, dtype={values.dtype}, color=RGB")
    print(f"  实测采集频率：{metrics['actual_frequency_hz']:.2f} Hz；"
          f"late cycles：{metrics['late_cycles']}")
    print(f"  最后一组双相机帧时间差：{camera.last_skew_s * 1000:.1f} ms")
    print(f"  双相机实时预览：{args.max_duration:.1f} s；按 q 或 Esc 可提前关闭")
    _preview(camera, args.max_duration, args.frequency)
    print(f"  qpos/action/gripper：{qpos.shape} / {actions.shape} / {gripper.shape}")
    print("  未连接机械臂、未发送夹爪命令、未写入 HDF5。")
    return 0


def _episode_metrics(timestamps: np.ndarray, frequency: float) -> dict[str, float | int]:
    dt = np.diff(timestamps)
    if not dt.size:
        return {"actual_frequency_hz": 0.0, "dt_mean_s": 0.0, "dt_p95_s": 0.0,
                "dt_max_s": 0.0, "late_cycles": 0}
    return {
        "actual_frequency_hz": float(1.0 / np.mean(dt)),
        "dt_mean_s": float(np.mean(dt)),
        "dt_p95_s": float(np.percentile(dt, 95)),
        "dt_max_s": float(np.max(dt)),
        "late_cycles": int(np.count_nonzero(dt > 1.5 / frequency)),
    }


def _record(args: argparse.Namespace, camera: ActRgbCameras, output: Path, lead_steps: int) -> int:
    if not confirm_action(
        "录制 ACT 示教轨迹",
        "机械臂将使能并进入 Leader 零重力模式；请扶稳机械臂并准备急停。",
        "RECORD",
    ):
        return 0

    robot = NeroRobotWrapper()
    connected = False
    leader_mode = False
    emergency = False
    cancelled = False
    images: dict[str, list[np.ndarray]] = {name: [] for name in camera.cameras}
    qpos_rows: list[np.ndarray] = []
    leader_rows: list[np.ndarray] = []
    gripper_rows: list[np.ndarray] = []
    gripper_targets: list[np.ndarray] = []
    timestamps: list[float] = []
    camera_skews: list[float] = []
    target_width_m = args.gripper_open_mm / 1000.0
    follower_until: float | None = None

    try:
        robot.connect()
        connected = True
        from src.act.replay_episode import move_to_replay_start
        from src.arm.prepare import START_JOINTS_DEG

        fixed_start = np.radians(np.asarray(START_JOINTS_DEG, dtype=np.float32))
        move_to_replay_start(
            robot, fixed_start, speed_percent=10, timeout_s=45.0,
            tolerance_rad=0.02,
        )
        target_width_m = float(robot.get_gripper_state()[0])
        print("\n状态：READY")
        print(f"当前夹爪宽度：{target_width_m * 1000:.1f} mm")
        print("已到统一预备位姿；按 r 后进入 Leader 并立即录制，请勿提前拖动。")
        while True:
            frames = camera.capture_rgb()
            key = _show(frames, "READY", "r: record | q: cancel | e: emergency")
            if key in (ord("r"), ord("R")):
                robot.enter_leader_mode()
                leader_mode = True
                break
            if key in (ord("q"), ord("Q"), ord("x"), ord("X"), 27):
                cancelled = True
                return 0
            if key in (ord("e"), ord("E")):
                emergency = True
                robot.emergency_stop()
                return 2

        print("状态：RECORDING")
        print("空格/q：完成并保存；x/Esc：取消；e：急停；o/c：打开/闭合夹爪。")
        print("按 o/c 前先停止拖动；程序会短暂进入 Follower，夹爪动作后自动恢复 Leader。")
        period = 1.0 / args.frequency
        start = time.monotonic()
        next_tick = start
        while True:
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_tick += period

            now = time.monotonic()
            if follower_until is not None and now >= follower_until:
                robot.enter_leader_mode()
                leader_mode = True
                follower_until = None
                print("夹爪动作阶段结束，已恢复 Leader，可以继续拖动。")

            frames = camera.capture_rgb()
            camera_skews.append(camera.last_skew_s)
            stamp = time.monotonic()
            teaching_qpos = (
                robot.get_arm_qpos()
                if follower_until is not None
                else robot.get_leader_qpos()
            )
            gripper = robot.get_gripper_state()
            for name, frame in frames.items():
                images[name].append(frame)
            # Leader feedback is live while dragging; normal feedback is live
            # during the short Follower hold used for gripper commands.
            qpos_rows.append(teaching_qpos.copy())
            leader_rows.append(teaching_qpos)
            gripper_rows.append(gripper.copy())
            gripper_targets.append(np.array([target_width_m], dtype=np.float32))
            timestamps.append(stamp)

            elapsed = stamp - start
            key = _show(
                frames,
                "GRIPPER HOLD" if follower_until is not None else "RECORDING",
                f"frames={len(timestamps)} time={elapsed:.1f}s "
                f"gripper={float(gripper[0]) * 1000:.1f}/{target_width_m * 1000:.1f}mm",
            )
            if key in (ord("o"), ord("O")) and follower_until is None:
                target_width_m = args.gripper_open_mm / 1000.0
                robot.enter_follower_mode()
                leader_mode = False
                robot.open_gripper(target_width_m, args.gripper_force_n)
                follower_until = time.monotonic() + args.gripper_hold_s
                print(
                    f"收到 OPEN：夹爪目标 {args.gripper_open_mm:.1f} mm；"
                    f"Follower 保持 {args.gripper_hold_s:.1f} s"
                )
            elif key in (ord("c"), ord("C")) and follower_until is None:
                target_width_m = args.gripper_close_mm / 1000.0
                robot.enter_follower_mode()
                leader_mode = False
                robot.close_gripper(target_width_m, args.gripper_force_n)
                follower_until = time.monotonic() + args.gripper_hold_s
                print(
                    f"收到 CLOSE：夹爪目标 {args.gripper_close_mm:.1f} mm；"
                    f"Follower 保持 {args.gripper_hold_s:.1f} s"
                )
            elif key in (ord("q"), ord("Q"), ord(" ")):
                break
            elif key in (ord("x"), ord("X"), 27):
                cancelled = True
                return 0
            elif key in (ord("e"), ord("E")):
                emergency = True
                robot.emergency_stop()
                return 2
            if elapsed >= args.max_duration:
                print("达到最长录制时长，自动结束。")
                break

        if len(timestamps) <= lead_steps:
            raise RuntimeError(f"有效帧只有 {len(timestamps)}，至少需要 {lead_steps + 1} 帧")

        robot.enter_follower_mode()
        leader_mode = False
        print("状态：SAVING")
        image_arrays = {name: np.stack(values).astype(np.uint8, copy=False)
                        for name, values in images.items()}
        qpos_array = np.stack(qpos_rows).astype(np.float32, copy=False)
        leader_array = np.stack(leader_rows).astype(np.float32, copy=False)
        gripper_array = np.stack(gripper_rows).astype(np.float32, copy=False)
        target_array = np.stack(gripper_targets).astype(np.float32, copy=False)
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        actions = build_leader_actions(leader_array, target_array, lead_steps)
        metrics = _episode_metrics(timestamp_array, args.frequency)
        metadata = {
            "task_name": args.task_name,
            "control_frequency_hz": args.frequency,
            "image_format": "RGB",
            "image_rotation_degrees": 180,
            "camera_names": list(image_arrays),
            "camera_rotations_degrees": {
                "camera_0": 180,
                **({"camera_1": args.secondary_camera_rotation_deg}
                   if args.secondary_camera_device else {}),
            },
            "demonstration_mode": "leader_kinesthetic_with_follower_gripper_hold",
            "action_semantics": "leader_qpos_plus_gripper_target",
            "action_lead_steps": lead_steps,
            "gripper_action_lead_steps": 0,
            "action_lead_ms": lead_steps / args.frequency * 1000.0,
            "fixed_start_joints_deg": START_JOINTS_DEG,
            **({"position_group": args.position_group} if args.position_group else {}),
            "qpos_unit": "rad",
            "gripper_unit": "m",
            "timestamp_clock": "monotonic",
            "acquisition_order": "image,timestamp,mode_specific_qpos,gripper_state",
            "gripper_follower_hold_s": args.gripper_hold_s,
            "camera_pair_skew_mean_ms": float(np.mean(camera_skews) * 1000),
            "camera_pair_skew_p95_ms": float(np.percentile(camera_skews, 95) * 1000),
            "camera_pair_skew_max_ms": float(np.max(camera_skews) * 1000),
            **metrics,
        }
        save_episode(
            output,
            image_arrays,
            qpos_array,
            actions,
            timestamp_array,
            gripper_states=gripper_array,
            metadata=metadata,
            overwrite=args.overwrite,
        )
        print("状态：DONE")
        print(f"  HDF5：{output}")
        print(f"  相机：{list(image_arrays)}")
        print(f"  帧数/时长/实际频率：{len(timestamps)} / {timestamp_array[-1] - timestamp_array[0]:.2f} s / "
              f"{metrics['actual_frequency_hz']:.2f} Hz")
        return 0
    except KeyboardInterrupt:
        emergency = True
        if connected:
            robot.emergency_stop()
        print("\n收到 Ctrl-C，已急停且不会保存不完整轨迹。")
        return 130
    except Exception:
        emergency = True
        if connected:
            try:
                robot.emergency_stop()
            except Exception as stop_error:
                print(f"警告：异常后的电子急停命令失败：{stop_error}")
        raise
    finally:
        if connected and leader_mode and not emergency:
            try:
                robot.enter_follower_mode()
            except Exception as exc:  # best-effort safety cleanup
                print(f"警告：恢复 Follower 失败：{exc}")
        if connected:
            robot.disconnect()
        if cancelled:
            print("状态：IDLE（已取消，未保存文件）")


def main() -> int:
    args = parse_args()
    output = args.output_dir / f"episode_{args.episode_id}.hdf5"
    lead_steps = max(0, round(args.action_lead_ms / 1000.0 * args.frequency))
    _print_plan(args, output, lead_steps)
    camera = ActRgbCameras(
        args.camera_width, args.secondary_camera_device, args.secondary_camera_rotation_deg
    )
    try:
        _check_output_location(output, args.overwrite)
        camera.open()
        if args.dry_run:
            return _dry_run(args, camera, lead_steps)
        return _record(args, camera, output, lead_steps)
    except Exception as exc:
        print(f"录制失败：{exc}")
        return 1
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
