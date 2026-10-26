> 当前分发是ACT重训模型依赖快照。模型结果与用法见 [包根README](../README.md)；下面历史硬件命令不属于本包执行入口。

# NERO 机械臂 + 双目摄像头

这是一个基于 `pyAgxArm` 的 NERO 机械臂项目，包含固件检查、机械臂复位/失能、关节运动、直线运动、夹爪和抓取搬运示例。项目当前默认使用 Linux SocketCAN 的 `can0`。

> 安全提示：运行任何运动脚本前，请固定底座、清空工作空间并准备好实体急停。不要在机械臂状态未知时发送运动命令；失能或复位时应托住机械臂，防止关节卸力后下落。

## 先看这里

| 你要做什么 | 从哪里开始 |
| --- | --- |
| 安装环境、连接 CAN、运行现成命令 | 本文档 |
| 判断某个文档是否仍然有效 | [文档索引](docs/README.md) |
| 开发或修改代码 | [贡献与工程规范](CONTRIBUTING.md) |
| 让新的编码 Agent 接手项目 | [AGENTS.md](AGENTS.md) |
| 排查连接、运动、标定、视觉或 ACT 问题 | [故障排查总览](docs/robot-troubleshooting.md) |
| 继续 ACT 数据、训练或实机验证 | [ACT 文档索引](docs/act/README.md) |

`README.md` 只负责用户入口和可运行命令；项目内部约定不要继续堆到这里。新功能的文档归属规则见[文档索引](docs/README.md)。

## 项目结构

```text
.
├── AGENTS.md                    # 新 Agent 必读的项目地图、状态与安全边界
├── CONTRIBUTING.md              # 文件摆放、接口、命名、测试和提交规范
├── README.md
├── docs/
│   ├── README.md                  # 文档唯一索引
│   ├── development.md              # 位姿、单位、坐标系和开发约定
│   ├── robot-troubleshooting.md    # 机械臂问题、根因、修复与验收总览
│   ├── act/
│   │   ├── README.md              # ACT 当前入口；区分计划和实施报告
│   │   ├── act-day1-report.md      # ACT 关节控制适配与真机验收步骤
│   │   ├── act-day2-report.md      # ACT Leader 示教采集、HDF5 和检查流程
│   │   ├── act-day3-report.md      # 数据验收、可视化与安全 replay
│   │   └── act-day6-report.md      # ACT 实机推理、安全层与分阶段验收
│   ├── vision-grasp-roadmap.md     # 相机自主抓取开发路线
│   └── transparent-beaker-pouring-implementation-plan.md # 透明器皿倾倒方案
├── src/
│   ├── act/                # ACT 数据、模型、训练、回放和实机适配
│   ├── arm/                # 使能、失能、准备、复位和夹爪命令
│   ├── calibration/        # camera→base、手眼和 TCP 标定命令
│   ├── demo/               # 已完成、可独立运行的能力演示与基线流程
│   │   ├── arm_*.py        # 关节、直线、坐标轴和抓取动作演示
│   │   ├── stereo_test.py  # ZED RGB-D 点选与采集演示
│   │   ├── yolo_detect.py  # YOLO 二维检测演示
│   │   ├── yolo_pick_place.py # YOLO RGB-D 监督抓取放置基线
│   │   ├── beaker_shaking.py # 识别、抓取和连续摇晃烧杯
│   │   └── beaker_pouring.py # 分阶段透明器皿倾倒
│   ├── utils/              # 机械臂、倾倒、视觉和语音公共辅助模块
│   └── voice/              # 语音识别和抓取可运行入口
├── scripts/                # CAN 配置与固件检查脚本
├── config/                 # 现场标定和运行参数；真实机器值通常不提交
├── configs/                # 可复现的 ACT 训练/推理实验 YAML
├── tests/                  # 默认不连接硬件的自动测试
└── data, logs, artifacts, checkpoints/ # 本地生成物；不提交 Git
```

完整文档导航统一维护在[文档索引](docs/README.md)，不要再从文件名猜测阅读顺序。
注意：`src/calibration/` 是标定源码；顶层 `calibration/` 是本机生成的标定样本，不提交 Git。
仓库路径命名规范见 [CONTRIBUTING.md](CONTRIBUTING.md#命名规范)。2026-09-12
起 Python 命令统一使用 `snake_case`；旧 CamelCase 模块不再保留兼容壳。

| 旧模块 | 新模块 |
| --- | --- |
| `scripts.checkFirmware` | `scripts.check_firmware` |
| `src.armEnable` / `src.armDisable` / `src.armPrepare` / `src.armReset` | `src.arm.enable` / `src.arm.disable` / `src.arm.prepare` / `src.arm.reset` |
| `src.handEyeCalibration` / `src.autoHandEyeCalibration` | `src.calibration.hand_eye` / `src.calibration.auto_hand_eye` |
| `src.calibrateCameraToBase` / `src.calibrateTcp` | `src.calibration.camera_to_base` / `src.calibration.tcp` |
| `src.transparentPouring` | `src.demo.beaker_pouring` |
| `src.voiceControl` / `src.voicePickPlace` | `src.voice.recognize` / `src.voice.pick_place` |
| `src.stereoTest` / `src.yoloDetect` / `src.yoloPickPlace` | `src.demo.stereo_test` / `src.demo.yolo_detect` / `src.demo.yolo_pick_place` |
| `src.demo.gripper_demo` | `src.arm.gripper` |
| `src.demo.<CamelCase>` | `src.demo.<snake_case>` |

本次调整不保留旧路径兼容模块；请按上表同步外部脚本和服务配置。

当前 `cup → book` 空杯监督抓取与放置已于 2026-09-01 实机跑通一次，项目处于监督抓取的重复性验收期。可复现参数、命令、验收结果和当前限制见[实机验收记录](docs/development.md#2026-09-01-cup-to-book-实机验收记录)。

## 开始使用

### 1. 准备环境

本项目目前使用 Ubuntu 24.04。创建虚拟环境并安装官方 SDK：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "python-can>3.3.4" h5py
python -m pip install "git+https://github.com/agilexrobotics/pyAgxArm.git"

# RTX 4090 + NVIDIA 535 驱动：安装匹配的官方 CUDA 12.1 PyTorch wheel
python -m pip install --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

本机验证组合为 NVIDIA driver `535.288.01`、PyTorch `2.5.1+cu121`、torchvision `0.20.1+cu121`。PyTorch wheel 自带所需 CUDA runtime，不要求把系统 CUDA Toolkit 改成同一小版本。可用下面的命令确认 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

安装和平台支持以 [pyAgxArm 官方说明](https://github.com/agilexrobotics/pyAgxArm#install) 为准。

### 2. 连接并启用 CAN

确认 CAN-H、CAN-L 和电源连接正确，再启动 `can0`，详见[激活单个CAN模块](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/can_user.md#2-%E6%BF%80%E6%B4%BB%E5%8D%95%E4%B8%AA-can-%E6%A8%A1%E5%9D%97)：

```bash
source ~/nero/bin/activate
sh scripts/find_all_can_port.sh
bash scripts/can_activate.sh can0 1000000
candump can0
```

candump 后终端应当持续输出类似下文内容：
```
can0  251   [8]  FF D5 00 CD 00 00 00 2A
can0  252   [8]  00 00 F6 DC 00 00 04 14
can0  253   [8]  00 00 00 07 00 00 00 00
can0  254   [8]  00 00 FC 50 00 00 02 FB
```

如果 `can0` 已经启动但参数不对，先执行 `sudo ip link set can0 down`，再重新设置。首次连接可参考 [NERO 官方 CAN 指南](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/first_time_user_guide_can.md)。

### 3. 选择固件配置

本机械臂固件为 `NeroFW.V120`。

运行前检查脚本顶部的 `FIRMWARE` 或 `NeroFW.V120`：

| NERO 固件 | SDK 配置 |
| --- | --- |
| ≤ 1.10 | `NeroFW.DEFAULT` |
| 1.11 | `NeroFW.V111` |
| 1.12 | `NeroFW.V112` |
| ≥ 1.20 | `NeroFW.V120` |

固件 ≤ 1.11 还需要在 Web 上位机开启 CAN 推送，或在使能时调用 `set_normal_mode()`；1.12 及以上无需该调用。

### 4. 从项目根目录运行

先执行只读检查，再运行低风险示例：

```bash
python -m scripts.check_firmware
python -m src.arm.gripper open
python -m src.demo.arm_circle
python -m src.demo.arm_rotate
python -m src.demo.arm_square
python -m src.demo.arm_grab
```

抓取前必须先在 `src/demo/arm_grab.py` 中测量并修改：

- `PICK_POSE_MM_DEG`：实际抓取位姿；
- `LIFT_HEIGHT_MM`：安全抬升高度；
- `TRANSFER_X_MM` / `TRANSFER_Y_MM`：搬运距离；
- 夹爪宽度、夹持力和运动速度。

所有硬件动作都使用统一格式显示“操作、风险、确认口令”。运动、抓取、夹爪和模式切换使用对应的大写口令；直接回车或输入不匹配会取消当前阶段。录音和标定命令菜单等非运动交互仍使用普通按键。

## 其他命令

```bash
# ZED RGB-D：鼠标点选三维点，s 保存样本，q 或 Esc 退出
python -m src.demo.stereo_test

# YOLOv8 通用类别预览，不连接机械臂；首次运行自动下载 yolov8n.pt
python -m src.demo.yolo_pick_place

# 根据实测对应点计算相机到机械臂基坐标系的外参
python -m src.calibration.camera_to_base calibration/camera_base_points.json

# 固定相机手眼标定（当前标定板黑色方框实测边长为 95.5 mm）
python -m src.calibration.hand_eye --marker-size-mm 95.5 --marker-id 0

# 自动运动手眼标定（相对路径参考 arm_square，首次运行全程守住急停）
python -m src.calibration.auto_hand_eye --marker-size-mm 95.5 --marker-id 0

# 零力拖动标定夹爪 TCP，并记录杯子横向抓取法兰姿态
python -m src.calibration.tcp

# 无针尖：使用尺测 TCP，只记录横向姿态；仅供高位无接触测试
python -m src.calibration.tcp --measured-tcp-xyz-mm 135 0 0

# 分别低速点动 base +X/+Y/+Z 30 mm，确认现场物理方向
python -m src.demo.base_axes

# 空载圆周调试：默认预览单条约 355° 的连续圆弧
python -m src.demo.arm_circle --arc continuous --speed-percent 30 --execute

# MoveIt 连续多圈圆周：需先按下方说明启动 MoveIt；默认仅规划
python -m src.demo.arm_circle_moveit --turns 3 --period-s 1.0

# 根据最近一次超限日志离线寻找安全余量内最快的摇晃参数；不连接硬件
python -m src.demo.shake_limit_explorer --observed-acceleration-rad-s2 5.957

# 两个半圆模式只用于诊断控制器的连续 move_c 行为
python -m src.demo.arm_circle --arc positive_y --execute
python -m src.demo.arm_circle --arc negative_y --execute
python -m src.demo.arm_circle --arc full --execute

# 单独使能机械臂并保持当前位置，不发送运动目标
python -m src.arm.enable

# 夹爪：默认只预览；实机执行时还需输入 OPEN 或 CLOSE
python -m src.arm.gripper open --execute
python -m src.arm.gripper close --width-mm 0 --force-n 0.8 --execute

# 从当前位置直接 move_j 到夹爪朝 base -X 的准备姿态（约 -219、-91、479 mm）
python -m src.arm.prepare

# 在实时画面中显示当前标定平移 RMSE 的实际尺度（不控制机械臂）
python -m src.demo.calibration_error_scale

# 失能：关节会卸力，运行前必须托住机械臂
python -m src.arm.disable

# 急停复位：确认实体急停已释放并托住机械臂
python -m src.arm.reset

# 运行不连接硬件的测试
python -m unittest discover -s tests

# ACT Day 1：默认只预览单关节流式轨迹，不连接、不运动
python -m src.act.test_joint_stream --dry-run

# ACT Day 1 真机测试必须从 5 Hz 开始，并输入 STREAM；完整步骤见报告
python -m src.act.test_joint_stream \
  --execute --joint-index 0 --frequency 5 --duration 5 --amplitude-rad 0.02

# ACT Day 2：检查相机与内存 schema，不连接机械臂、不写数据
python -m src.act.record_episode --dry-run --frequency 20

# ACT Day 2：Leader 拖动示教；必须输入 RECORD，按 r 开始、空格结束
python -m src.act.record_episode \
  --execute --episode-id 0 --frequency 20 --task-name pick_place

# 检查一条 HDF5 episode 的 shape、范围、频率和时间戳
python -m src.act.episode_inspect data/act/episode_0.hdf5

# ACT Day 3：批量严格验收并生成本地 manifest
python -m src.act.validate_episode data/act/*.hdf5 \
  --manifest data/act/manifest.json

# 生成关节、action 和同周期误差曲线
python -m src.act.episode_plot data/act/episode_1.hdf5 \
  --output-dir artifacts/act_day3

# Replay 默认 dry-run；真机必须显式 --execute，并先从前20帧、0.25x开始
python -m src.act.replay_episode data/act/episode_1.hdf5 \
  --dry-run --start-step 0 --end-step 20 --speed-scale 0.25

# 无法手动找到示教起点时，可在确认整段扫掠空间安全后低速移动到起点
python -m src.act.replay_episode data/act/episode_1.hdf5 \
  --execute --move-to-start --start-step 0 --end-step 20 --speed-scale 0.25

# ACT Day 4：离线检查 NERO dataset 与官方 ACT 的输入输出，不连接机械臂
python -m src.act.train_smoke_test --smoke-only --episode-count 1

# 用单条 valid episode 做小样本 overfit；checkpoint/artifacts 默认不提交 Git
python -m src.act.train_smoke_test --episode-count 1 --steps 300

# 查看反归一化 action chunk，并执行离线限位/步长检查
python scripts/act_predict_debug.py \
  checkpoints/act_day4_overfit/policy_1ep_last.ckpt \
  checkpoints/act_day4_overfit/dataset_stats_1ep.pkl \
  --episode-id 1 --start-ts 0

# ACT Day 5：生成固定的 16/4 episode split 和 train-only 数据统计
python -m src.act.dataset_stats --config configs/act_nero_day5.yaml

# 使用 RTX 4090 正式训练；全程不连接机械臂
python -m src.act.train --config configs/act_nero_day5.yaml --device cuda

# 使用 best-val checkpoint 对完整 validation set 做离线评估
python -m src.act.eval_offline \
  --checkpoint checkpoints/act_day5/policy_best_val.ckpt \
  --config configs/act_nero_day5.yaml \
  --split val

# ACT Day 6：真实相机/关节反馈推理 50 周期；会使能机械臂，但绝不发送目标
python -m src.act.run_act_real \
  --config configs/act_nero_day6.yaml --dry-run --max-steps 50

# 如果初始 qpos 被拒绝，先经两次确认低速移动到一条有效示教的起点
python -m src.act.replay_episode data/act/episode_1.hdf5 \
  --execute --move-to-start --move-only --start-step 0 --end-step 1

# dry-run 日志通过后，第一轮只允许 10 个低速控制周期
python -m src.act.run_act_real \
  --config configs/act_nero_day6.yaml --execute \
  --max-steps 10 --speed-scale 0.25

# 前两阶段验收后再启用时序聚合；超过 10 step 必须显式声明 --extended
python -m src.act.run_act_real \
  --config configs/act_nero_day6.yaml --execute --temporal-ensemble \
  --max-steps 50 --speed-scale 0.25 --extended
```

### MoveIt 连续圆周 demo

先用模拟机械臂验收。终端 1 启动 MoveIt，并保持该终端运行：

```bash
source /opt/ros/jazzy/setup.bash
source ~/projects/nero_ros2_ws/install/setup.bash
ros2 launch agx_arm_moveit demo.launch.py \
  arm_type:=nero effector_type:=agx_gripper follow:=false
```

终端 2 先只规划；确认输出为 `规划完成` 后，再执行模拟轨迹：

```bash
cd ~/projects/nero
source ~/nero/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/projects/nero_ros2_ws/install/setup.bash
python -m src.demo.arm_circle_moveit --turns 3 --period-s 1.0
python -m src.demo.arm_circle_moveit --turns 3 --period-s 1.0 --execute
```

单独的真机圆周 demo 当前仍只用于规划；抓杯与摇匀由下方受监督入口执行。
真机 launch 必须保留平滑 `move_j`，并使用昨天实测正常的 200 Hz 插值：

```bash
ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py \
  can_port:=can0 arm_type:=nero effector_type:=agx_gripper \
  fast_mode:=false controller_rate_hz:=200 speed_percent:=75 \
  auto_control_gate:=true use_rviz:=false
```

真机在终端 2 只能使用真实关节反馈做规划：

```bash
python -m src.demo.arm_circle_moveit \
  --joint-state-topic /feedback/joint_states --turns 3 --period-s 1.0
```

ACT 各阶段的完整定义、验收结论和风险记录在 `docs/act/act-day1-report.md` 至 `docs/act/act-day6-report.md`。checkpoint 不能绕过 Day 6 安全适配层直接控制机械臂。

双相机 ACT 的相机摆位、录制、检查、训练和实机推理命令见 [docs/act/dual-camera-recording.md](docs/act/dual-camera-recording.md)。新一轮固定起点数据单独保存在 `data/act_dual_fixed`，不要与旧数据混用。

使用现有 YOLO 搬运策略自动生成一条无中途键盘暂停的双相机 ACT 示教：

```bash
python -m src.act.record_yolo_episode --episode-id 0 \
  --position-group 1
```

在重训前离线审查全部闭爪窗口，并用当前 checkpoint 检查 close 预测时序；该命令不会连接
相机、CAN 或机械臂：

```bash
python -m src.act.grasp_transition_audit \
  data/act_dual_fixed/episode_*.hdf5 \
  --config configs/act_nero_dual_xyz_grasp_focus_day6.yaml \
  --output-dir artifacts/act_dual_xyz_grasp_focus_day5/grasp_transition_audit \
  --device cpu
```

固定蓝卡新数据的当前候选模型使用
`configs/act_nero_dual_fixed_destination_day6.yaml`。先做 50 周期 dry-run 并保存实际模型输入；
通过后只做 10 周期、关闭 temporal ensemble 的低速实机测试：

```bash
python -m src.act.run_act_real \
  --config configs/act_nero_dual_fixed_destination_day6.yaml \
  --dry-run --max-steps 50 --device cuda --save-model-input-every 10

# 实验性 ACT + adaptive MPC：仍为 dry-run，不发送运动命令
python -m src.act.run_act_real \
  --config configs/act_nero_dual_fixed_destination_day6.yaml \
  --dry-run --max-steps 50 --device cuda --adaptive-mpc \
  --save-model-input-every 10

python -m src.act.run_act_real \
  --config configs/act_nero_dual_fixed_destination_day6.yaml \
  --execute --max-steps 10 --speed-scale 0.25 --device cuda \
  --save-model-input-every 5
```

不要在检查这两次 CSV、summary 和模型输入前增加步数或启用 temporal ensemble。该候选只完成
离线验收，尚未通过真机完整抓放。

`adaptive_mpc.enabled` 默认关闭。启用后，TCP 距目标超过 150 mm 时原样使用 ACT，进入
150 mm 后才渐进启用 MPC，50 mm 内完全介入；MPC 每关节最多只在 ACT 目标上修正
0.005 rad，且不会改写夹爪预测。输出继续经过原有安全过滤、抓取门禁和 watchdog。先检查
CSV 中的 `mpc_active`、`mpc_blend`、`mpc_correction_norm_rad`、`mpc_fallback` 和求解耗时；
未通过 dry-run 时不得用于 `--execute`。

## 语音控制

语音模块使用 `arecord` 采集麦克风音频，`faster-whisper` 离线转写中文，再通过固定白名单解析待抓物和颜色卡片。未知、否定或不完整的语句不会执行。

```bash
source /home/user/nero/bin/activate
python -m pip install faster-whisper

# 仅测试麦克风、转写和基础白名单，不控制机械臂
python -m src.voice.recognize --device pipewire --duration 7

# 先用文字验证结构化指令和生成的搬运命令
python -m src.voice.pick_place \
  --text "把黄色杯子放在蓝色卡片上" \
  --source-offset-mm 0 -17 0

# 麦克风识别成功后，再启动保留人工确认的抓取流程
python -m src.voice.pick_place \
  --device pipewire \
  --duration 7 \
  --transit-speed-percent 50 \
  --precision-speed-percent 5 \
  --source-offset-mm 0 -17 0 \
  --destination-offset-mm 0 0 0 \
  --execute
```

PipeWire 占用 USB 麦克风时应使用 `--device pipewire`，不要直接打开独占的 `hw:N,M`。程序兼容 Whisper 偶尔输出的繁体颜色词，并支持 `cup`、`box`、`bottle` 与红/蓝/黄卡片别名；只有已经配置运动几何的物体允许进入执行阶段。执行模式仍需在视觉窗口按 `m`，并在终端确认 `PICK` 和 `GRASP`。

## 常见问题

### 找不到 `pyAgxArm`

确认已经激活 `.venv`，然后使用同一个解释器执行 `python -m pip show pyAgxArm`。不要混用系统 `pip` 和虚拟环境里的 `python`。

### `can0` 不存在或 `connect()` 失败

执行 `ip link show` 检查接口。确认 USB-CAN 已识别、驱动已加载且脚本中的通道为 `can0`。

### `candump can0` 没有数据

依次检查波特率是否为 `1000000`、CAN-H/L 是否接反、端子是否接触良好，以及 `NeroFW` 是否与真实固件匹配。旧固件还要确认 CAN 推送已开启。

### 连接后读取不到状态或关节角

`connect()` 后设备可能需要短暂初始化；示例已保留必要等待。如果持续无反馈，请先停止动作脚本，用 `candump can0` 检查总线，并确保没有另一个程序同时占用或控制机械臂。

### 机械臂进入 `EMERGENCY_STOP`

先排除碰撞和路径阻挡，释放实体/Web 急停并托住机械臂，然后运行 `python -m src.arm.reset`。不要在原因不明时反复复位。

### 运动超时或目标不可达

不要直接放宽容差或提高速度。先检查目标单位是否正确（位置为米、姿态和关节角为弧度）、目标是否超出工作空间、路径是否经过奇异位姿。抓取示例的用户输入则使用毫米和度，由工具函数转换为 SDK 单位。

控制器可能在断开程序后继续保留上一次 `NO_SOLUTION`、奇异点或目标越界状态。共享的 `move_joint_and_wait()` 会允许新的安全 `move_j` 覆盖这三类历史规划状态；急停、碰撞、关节通信异常等状态仍会拒绝运动。新脚本应复用该函数，不要在发送恢复关节运动前调用只接受 `NORMAL` 的 `check_arm_normal()`。

机械臂从失能状态启动时，驱动器使能成功与主控制器完成制动器释放之间可能存在短暂延迟。共享的 `enable_arm()` 会等待关节反馈和控制器状态连续就绪，运动脚本不应在 `enable()` 返回后立即自行发送目标。

### 断开连接后机械臂为什么仍保持使能

`disconnect()` 只断开 CAN 通信，不等于失能。示例不会自动失能，以免机械臂突然下落；需要卸力时请托住机械臂并单独运行 `python -m src.arm.disable`。

### CAN正常连接，机械臂保持使能状态，为什么无法控制机械臂/机械臂不返回信号？

可以使用`candump`检查输出，若无输出可以插拔机械臂USB后重新连接。


# ZED 双目摄像头

## 开始使用

### 安装环境

官方SDK下载[路径](https://www.stereolabs.com/en-fr/developers/release#CUDA12)。

在已经配置好的nero虚拟环境中，查看zstd安装情况：
```
zstd --version

# 若已经安装，则跳过安装步骤
sudo apt install zstd

chmod +x ZED_SDK_UbuntuXX_cudaYY.Y_vZ.Z.Z.zstd.run
./ZED_SDK_UbuntuXX_cudaYY.Y_vZ.Z.Z.zstd.run
```

官方 api [接口](https://www.stereolabs.com/docs/api/python)

## 推荐：ArUco 手眼标定

项目已经提供一个不依赖 ROS 的小型 `eye-to-hand` 标定程序，思路与 `easy_handeye` 相同：相机固定不动，标记跟随机械臂运动，同时采集法兰位姿和标记在相机中的位姿，联合求解 `T_base_camera`。

### 准备

1. 程序可自动识别 `DICT_5X5_50` 和 `DICT_4X4_50`，当前标定板使用 ID `0`；一次标定只能使用同一种标定板，并保持打印比例不缩放。
2. 用卡尺测量黑色正方形外边界的实际边长；命令中的 `--marker-size-mm` 必须使用这个实测值，不是整张纸的宽度。
3. 将标记刚性固定在法兰或夹爪不随开合移动的部分。安装位置可以偏离夹爪中心，但采样期间不能松动。
4. 固定 ZED、机械臂底座和桌面；让标记在全部采样姿态下完整可见。

### 采集与求解

```bash
source /home/user/nero/bin/activate
cd /home/user/work
python -m src.calibration.hand_eye --marker-size-mm 95.5 --marker-id 0
```

程序会在输入 `DRAG` 确认后使能机械臂并进入 Leader 零力拖动模式。切换前必须托住机械臂、清空工作区并准备好急停；不要同时启动另一个 Python CAN 客户端。手动拖到目标姿态后按 `s`，程序会切换到 Follower 位置保持模式；法兰反馈连续稳定 `0.5 秒`后，程序丢弃两帧并使用第三张新画面，再核对拍照前后的法兰变化不超过 `0.5 mm` 和 `0.1°`。无论采样成功或被拒绝，程序都会在提示 1 秒后恢复 Leader；整个过程必须持续托住机械臂。稳定等待期间预览会暂时停在最后一帧，但该帧不会用于采样。

- `s`：保存当前一组法兰/标记位姿；样本会立即写入 `calibration/handeye_时间.json`；
- `c`：使用已采集样本求解；
- `q` 或 `Esc`：退出。

退出或发生可处理异常时，程序会尝试恢复 Follower 位置保持模式。恢复期间继续托住机械臂，直到确认关节重新保持；断电、强制杀进程等情况不能依赖该自动恢复。

至少采集 15 组，建议 15–20 组。标记应覆盖工作区的近、中、远位置，并明显改变 Roll、Pitch、Yaw，不能只做平移或只绕一个轴旋转。按 `c` 时，法兰位置至少要在两个 XYZ 方向各覆盖 `100 mm`；覆盖不足时不会求解。相机虽然倒装，程序只把预览旋转 180°；检测和相机内参仍使用 ZED 原始图像坐标。

采样时只接受经过亚像素角点优化的标准 ArUco 解码，法兰位姿直接使用 Follower 中持续更新的控制器反馈，默认拒绝重投影 RMSE 大于 `2 px` 的帧。按 `c` 后使用 MAD 稳健求解，最多剔除 `25%` 的明显离群样本，并把原始样本编号写入 `fit.excluded_sample_indices`；离群过多或最终平移 RMSE 超过当前开发门限 `10 mm`、旋转 RMSE 超过 `2°` 时不会覆盖已有外参。

结果写入 `config/camera_to_base.json`，可直接由 `src.demo.stereo_test` 加载。终端显示的“一致性误差”只能发现采样之间不一致，不能代替验收；必须再用至少 3 个未参与标定的独立点比较 `base XYZ` 与实测值。当前抓取允许误差为 `5 mm`，外参误差应尽量控制在 `2–3 mm`。

`src.demo.stereo_test` 会拒绝加载平移 RMSE 超过当前开发门限 `10 mm` 或旋转 RMSE 超过 `2°` 的手眼标定文件。该门限只表示允许加载进行开发验证，不代表已经达到最终 `5 mm` 抓取目标。

截至 2026-08-30，最新自动标定文件为 `calibration/auto_handeye_20260830_143806.json`：40组有效样本中稳健求解采用35组，统计剔除样本 `[36, 37, 38, 39, 40]`；平移 RMSE `5.994 mm`、最大值 `10.510 mm`，旋转 RMSE `0.938°`、最大值 `1.447°`。平均重投影误差 `0.178 px`，最大值 `0.515 px`。结果已写入 `config/camera_to_base.json` 并通过当前开发门限，但仍须使用至少3个未参与标定的独立点验证实际 `base XYZ`。

这套方法会同时求出标记相对法兰的未知安装变换，因此不需要先标定针尖 TCP。标定结束后可以拆掉 ArUco；只要 ZED 和机械臂底座没有移动，`T_base_camera` 仍然有效。实际抓取仍需单独标定夹爪抓取中心 TCP。

### 自动运动采样

确认相机、标定板安装和无障碍工作区后，可运行：

```bash
python -m src.calibration.auto_hand_eye --marker-size-mm 95.5 --marker-id 0
```

程序先进入 `demo.arm_square` 使用的弯曲准备姿态，再依次执行40个已经实机验证为可达、稳定且能识别标定板的相对 `move_p` 目标；原99点候选路径中的59个失败点已删除。默认 XYZ 范围约为 `220 × 320 × 180 mm`，最低点保持为已验证的相对高度 `dz=-140 mm`。可用 `--range-scale 0.5` 到 `1.2` 调整 XY 范围，Z 不随该参数缩放；只有默认 `1.0` 的40个点经过实机验证。单点异常仍会跳过，主动按 `Ctrl+C` 会尝试发送电子急停。

## 备用：针尖对应点标定流程

无法把 ArUco 刚性固定到末端时，可以继续使用针尖和 RGB-D 对应点方法。该方法分为两步，顺序不能颠倒：先确定临时针尖相对法兰的 TCP 偏移，再建立 ZED 相机坐标系到机械臂基坐标系的外参。

### 1. 标定临时针尖 TCP

1. 将刚性小针固定在法兰或夹爪的固定部分，针可以偏离夹爪几何中心，但不能松动或弯曲；不要安装在会随夹爪开合移动的位置。
2. 在桌面固定一个带小孔或锥形凹槽的点，避免针尖接触时滑动。
3. 使用至少 6 个明显不同的法兰姿态，让针尖从不同方向轻触同一个固定点。必须改变 Roll、Pitch、Yaw，不能只平移。
4. 根据各次法兰反馈位姿求解针尖相对法兰的 TCP 偏移。接触时不要继续下压，调试期间保持低速并准备好急停。

这一步使用“同一个点、多个姿态”。SDK 返回的是法兰位姿；针尖 TCP 未标定前，不能直接把法兰 XYZ 当作针尖的 `base XYZ`。

### 2. 标定相机到机械臂基坐标系

1. 在工作区中心、四角和边缘准备至少 6 个标定点，另外保留至少 3 个不参与求解的验证点；最好加入两三个不同高度的点。
2. 用已标定的针尖依次接触每个点，根据法兰位姿和 TCP 偏移计算该点的 `base XYZ`。
3. 移开针尖，在 `stereo_test` 中点击同一个物理点，确认窗口显示有效的 `camera XYZ` 后按 `s` 保存。
4. 将每个点的 `camera_xyz_mm` 和 `base_xyz_mm` 写入 `calibration/camera_base_points.json`。文件格式见[视觉自主抓取开发路线](docs/vision-grasp-roadmap.md#42-相机外参)。
5. 求解外参：

```bash
python -m src.calibration.camera_to_base calibration/camera_base_points.json
```

结果默认保存到 `config/camera_to_base.json`。至少使用 3 个独立验证点检查误差；当前充电头抓取的总误差目标为 `5 mm` 以内，因此外参误差应尽量控制在 `2–3 mm`。

### 3. 验证并换回夹爪 TCP

外参通过独立点验证后，可以取下标定针。相机外参描述的是“ZED 相机 → 机械臂基座”，取下针不会使它失效；实际抓取前仍需测量或标定夹爪两指之间的抓取中心 TCP。

摄像头、相机支架或机械臂底座发生移动后，必须重新进行相机外参标定；桌面移动后要重新测量桌面高度和工作区边界。外参通过验证前，视觉坐标不得用于驱动机械臂。

### 4. 标定实际夹爪 TCP 和横向姿态

`src.calibration.tcp` 使用“同一固定点、多个法兰姿态”的枢轴法，同时记录杯子横向抓取时的法兰 RPY：

```bash
python -m src.calibration.tcp
```

需要用刚性夹具让针尖准确代表计划使用的夹爪抓取中心。针尖如果只是在抓取中心之外随意固定，程序得到的是针尖 TCP，不能用于杯子抓取。准备一个固定的锥形凹点，输入 `DRAG` 进入 Leader 后：

- 每次让针尖轻触同一凹点，明显改变 Roll、Pitch、Yaw，停止后输入 `s`；至少采集 8 组；
- 将夹爪摆成水平、从基坐标 `+X` 一侧朝 `-X` 接近杯子的姿态，停止后输入 `p`；
- 输入 `c` 求解并保存到 `config/tcp_offset.json`；输入 `q` 退出。

程序退出时会恢复 Follower 位置保持模式。只有拟合 RMSE、最大误差以及独立点验证都满足抓取要求后，才能把该 TCP 用于视觉运动。

没有针尖时，可用 `--measured-tcp-xyz-mm 135 0 0` 暂存尺测结果并只记录横向 RPY。当前夹爪从法兰局部 `+X` 方向伸出，`135 mm` 来自法兰安装面到两指抓取中心约 `130–140 mm` 的测量，因此配置会写入 `"approximate": true` 且没有拟合误差。必须依次完成高位悬停和杯面前停止验证，确认实机偏差可接受后，才允许在人工监督下抓杯；它不能代替正式 TCP 标定。

输入 `p` 后程序会根据粗测 TCP 向量显示“法兰到 TCP”的伸出方向在 base 中的单位向量；本任务必须接近 `[-1, 0, 0]`。如果方向偏离 `-X` 超过 `15°`，程序会拒绝保存。

## YOLOv8 目标定位与监督搬运

先在 nero 虚拟环境安装推理依赖。若两类物体都在通用数据类别中，可直接使用默认 `yolov8n.pt`；首次运行由 Ultralytics 自动下载，不需要自训练 `best.pt`。

```bash
source /home/user/nero/bin/activate
python -m pip install ultralytics

# 自训练模型纯二维预览：只检测 bottle，不读取深度或连接机械臂
python -m src.demo.yolo_detect \
  --weights /home/user/projects/yolo/best.pt \
  --classes bottle \
  --confidence 0.15 \
  --imgsz 960

# 省略 --classes 时显示模型识别到的全部类别
python -m src.demo.yolo_detect \
  --weights /home/user/projects/yolo/best.pt

# 用本次双类别模型在 ZED 实时画面中识别烧杯和玻璃蒸发皿
python -m src.demo.yolo_detect \
  --weights /home/user/projects/yolo/runs/detect/vessels_train_fixed/weights/best.pt \
  --classes beaker crystallizing_dish \
  --confidence 0.50 \
  --imgsz 960

# 多烧杯实验：保留每个 beaker 检测框，并在画面和终端显示总数
python -m src.demo.yolo_detect \
  --weights /home/user/projects/yolo/runs/detect/beaker_train-2/weights/best.pt \
  --classes beaker \
  --confidence 0.20 \
  --imgsz 960

# 多烧杯杯底中心实验：输出未经经验补偿的 base XYZ，不连接机械臂
python -m src.demo.multi_beaker_localize \
  --weights /home/user/projects/yolo/runs/detect/beaker_train-2/weights/best.pt \
  --table-z-base-mm 0 \
  --beaker-height-mm 95 \
  --beaker-diameter-mm 70 \
  --confidence 0.20 \
  --imgsz 960

# 冻结上一步的三个稳定坐标，先只打印抓取对齐目标
# 默认应用 config/transparent_pouring.json 中的源位置补偿（当前 Y=-15 mm）
python -m src.demo.multi_beaker_pregrasp \
  --center-mm B1 -475.98 237.57 0 \
  --center-mm B2 -490.65 31.22 0 \
  --center-mm B3 -511.46 -164.14 0

# 一次完成“相机稳定定位三个烧杯 → 冻结坐标 → 打印抓取对齐目标”
# 三个目标全部变绿后按 p；按 q/Esc 取消。此命令默认仍不连接机械臂
python -m src.demo.multi_beaker_pregrasp \
  --weights /home/user/projects/yolo/runs/detect/beaker_train-2/weights/best.pt \
  --table-z-base-mm 0

# 目标核对无误后才允许实机；需完整重启 MoveIt 驱动并准备急停
# 首次输入 MOVE3，之后直接回车逐步继续，输入 q 停止
# 默认转场/近桌/检查速度为 15%/5%/3%
# 转场高出抓取点 200 mm；非目标烧杯以带 5 mm 余量的圆柱参与碰撞检查
# 在上述命令末尾添加：--execute --allow-approximate-tcp

# 通用类别预览：显示模型识别到的类别和 base XYZ
python -m src.demo.yolo_pick_place

# 指定待抓物和承接物后，连续 8 帧三维坐标波动不超过 5 mm 可按 m 锁定
python -m src.demo.yolo_pick_place \
  --source-class cup \
  --destination-class book

# 粗测 TCP 下的高位悬停验证：从预先固定的初始位姿到杯子 +X 高位（不下降）
python -m src.demo.yolo_pick_place \
  --source-class cup \
  --destination-class book \
  --hover-only \
  --workspace-mm -650 0 -300 300 0 500

# 侧向接近验证：从高位下降到杯侧，再沿 -X 接近并在杯面前 50 mm 停止
python -m src.demo.yolo_pick_place \
  --source-class cup \
  --destination-class book \
  --side-approach-only \
  --workspace-mm -650 0 -300 300 0 500

# 监督抓取与放置：首次确认输入 PICK，到杯面前再次输入 GRASP
python -m src.demo.yolo_pick_place \
  --source-class cup \
  --destination-class book \
  --pick-place \
  --workspace-mm -650 0 -300 300 0 500
```

无接触安全段默认以 `30%` 速度运动，单段等待 `60 s`；可用
`--transit-speed-percent` 和 `--motion-timeout-s` 调整。接近杯子的直线段固定为
`5%`。横向姿态来自 `config/tcp_offset.json`：TCP 伸出方向必须朝 base `-X`，两指开合方向必须沿 base `Y`。当前安装对应 `Roll=-90°、Pitch=0°、Yaw=180°`；程序只有在位置和姿态都到达容差后才报告完成。

视觉表面点到杯心的修正与杯子物理半径已经分离：实机验证采用 `--surface-center-correction-mm 1`，而 `--source-radius-mm 40` 仅用于计算杯面前的安全停止位置。不要再把杯子半径直接用作视觉修正距离。

`--pick-place` 默认按杯高 `220 mm`、夹爪张开 `95 mm`、夹紧目标 `80 mm`、夹持力 `0.5 N` 运行。程序连接后记录启动关节角并打印启动法兰位姿；若启动 Y 不大于 0 会给出警告。放置后沿 base `+X` 退出 `50 mm`、垂直抬高，再用 `move_j` 返回启动位置。运行前应把机械臂放在已验证且偏向 base `+Y` 的安全初始位置，并确认返回关节路径无障碍。首次运行应使用空杯并全程守住急停。

程序在正向 ZED 左目图像上检测目标，取检测框中心区域的有效点云中位数，经 `config/camera_to_base.json` 转换为 `base XYZ`。按 `r` 清空稳定帧，按 `q` 退出。影子模式不会连接机械臂，应先用尺或已知点验证整个工作区的输出。

独立点验证、夹爪 TCP 偏移和俯抓法兰姿态都确认后，才添加 `--execute`。下面的尖括号必须替换为实测值；工作空间顺序为 `Xmin Xmax Ymin Ymax Zmin Zmax`，全部使用毫米：

```bash
python -m src.demo.yolo_pick_place \
  --weights /path/to/best.pt \
  --source-class charger \
  --destination-class target \
  --execute \
  --flange-rpy-deg <roll> <pitch> <yaw> \
  --tcp-offset-mm <x> <y> <z> \
  --workspace-mm <xmin> <xmax> <ymin> <ymax> <zmin> <zmax>
```

执行模式仍需在终端输入 `PICK` 才会连接机械臂。流程为打开夹爪、`move_p` 到待抓物上方、`move_l` 下降/夹持/抬升，再 `move_l` 到承接物上方；到达后保持夹持，不会自动下降或松爪。首次实机运行使用空夹爪和较大的 `--destination-clearance-mm` 验证路径，并全程守住急停。

## 透明烧杯摇晃与倾倒

两个完成的演示入口分别为 `src.demo.beaker_shaking` 和
`src.demo.beaker_pouring`。它们共享一套透明器皿感知与抓取规划，倾倒 demo 固定关闭
摇晃，摇晃 demo 固定使用 MoveIt 连续圆周轨迹。感知支持四种方式：

1. 预训练 YOLO-seg 直接输出 mask；
2. SAM 2.1 由用户依次点击源烧杯和目标器皿，无需训练；
3. 现有 YOLO 检测框自动提示 SAM 2.1，检测失败时仍可点击覆盖。
4. 烧杯 YOLO 检测框与固定 base 水平面求交，完全不读取深度。

程序默认只做 **dry-run**；只有显式添加 `--execute` 并选择分阶段动作后才会连接机械臂。当前 `/home/user/projects/yolo/best.pt` 是检测框模型，不能单独提供透明器皿 mask，但可以在第 3 种方式中作为 SAM 的提示器。

实际配置 `config/transparent_pouring.json` 已记录烧杯直径/高度 `70/95 mm` 和蒸发皿外径/高度 `100/56 mm`。带液体阶段还必须实测有效内口直径，并填入 `target.inner_diameter_mm`。

倾倒方向当前固定为 base `+Y`，对应配置 `pour.direction_base: [0, 1, 0]`。烧杯直接移动到蒸发皿 `-Y` 方向的斜上方倾倒起点，近侧杯口位于蒸发皿外缘外并朝向中心，再绕 base `-X` 轴倾斜并沿原路径回正；倾倒前不再向蒸发皿下移。当前水平退让 `55 mm`、杯口最小垂直高度 `110 mm`，规划器还会自动抬高圆弧，保证倾倒起点不低于抓杯后的抬升高度。

### 推荐：烧杯 YOLO + 固定桌面（完全不使用深度）

烧杯模型只负责检测框；程序根据杯底/杯口中心轴、已知抓取高度、相机外参和
`base Z` 水平面计算抓取点。蒸发皿第一阶段固定放置，使用实测的 base XY。
`--target-base-xy-mm` 后的两个值是蒸发皿中心的真实基坐标；`0 0` 表示机械臂
基座原点，并不表示自动检测，通常会导致倾倒法兰路径越界。

也可以用原六类别模型的 `blue` 类直接定位蓝色纸片中心，并将其作为倾倒目标：

```bash
python -m src.demo.beaker_pouring \
  --depthless \
  --source-detector-weights /home/user/projects/yolo/runs/detect/beaker_train-2/weights/best.pt \
  --source-class beaker \
  --target-detector-weights /home/user/projects/yolo/best.pt \
  --target-class blue \
  --table-z-base-mm 0 \
  --confidence 0.20 \
  --execute --execution-stage pour --pour-angle-deg 60
```

实际倒水时，蒸发皿中心必须与蓝色纸片中心重合；纸片只是定位标记。首次仍应
使用空烧杯验证完整轨迹，可用 `--target-offset-mm dx dy dz` 修正固定视觉偏差。
空杯流程的转场速度默认为 `75%`，近桌接近速度降为 `15%`。`liquid` 阶段进一步将转场/接近速度限制为 `30%/15%`，倾倒和回正圆弧按当前要求使用 `100%`。
倾倒和回正各使用一次控制器原生 `move_c` 圆弧，不再按每 `5°` 停止等待。
每条 `move_c` 前会先向当前位置发送零位移 `move_l` 并等待 `MOVE_L` 反馈，再重新进入 `MOVE_C`，避免连续圆弧被固件当成上一条已完成指令而忽略。
倾倒 demo 抓住并垂直抬升烧杯后直接转场到目标，不执行摇晃动作。摇晃 demo
保持烧杯姿态不变、杯口向上，抓取 TCP 以 `10 mm` 半径连续转圈。

`src.demo.beaker_shaking` 重建了 2026-09-15 中午实测正常的 MoveIt 组合：
`fast_mode=false`、200 Hz 插值、高位转场 75%、近桌 15%、最终抓取 5%。抓起后
保持杯口向上，以 10 mm 半径连续转 3 圈，每圈 1 秒，再放回原位。程序在运动前
读取驱动模式和控制频率；不是上述组合会直接拒绝。第一条只检测和生成计划，
第二条才连接已启动的 ROS MoveIt；画面稳定后按 `p` 会先关闭相机窗口。

```bash
python -m src.demo.beaker_shaking \
  --source-detector-weights /home/user/projects/yolo/runs/detect/vessels_train_fixed/weights/best.pt \
  --source-class beaker --table-z-base-mm 0 --confidence 0.20 \
  --allow-approximate-tcp

python -m src.demo.beaker_shaking \
  --source-detector-weights /home/user/projects/yolo/runs/detect/vessels_train_fixed/weights/best.pt \
  --source-class beaker --table-z-base-mm 0 --confidence 0.20 \
  --gripper-contact-force-n 0.5 --gripper-force-n 0.8 \
  --allow-approximate-tcp --execute
```

执行命令固定加入 `--moveit --shake-cycles 3 --shake-period-s 1.0`，无需重复填写。

终端 1 可以用脚本单独启动并管理 MoveIt 驱动。脚本不会运行 Python；它会拒绝已有
的同名驱动，并在本终端按 `Ctrl+C` 后清理本次 launch 的全部 ROS 子进程：

```bash
bash scripts/start_moveit_driver.sh
```

驱动显示就绪后，在终端 2 手动运行上面的 `python -m src.demo.beaker_shaking ...`
命令；Python 退出后，再回到终端 1 按 `Ctrl+C` 关闭驱动。
如果原终端已经消失但驱动节点仍在，先检查再清理残留进程：

```bash
bash scripts/stop_moveit_driver.sh --dry-run
bash scripts/stop_moveit_driver.sh
```

```bash
python -m src.demo.beaker_pouring \
  --depthless \
  --source-detector-weights /home/user/projects/yolo/runs/detect/beaker_train-2/weights/best.pt \
  --source-class beaker \
  --table-z-base-mm <桌面Z> \
  --target-base-xy-mm <蒸发皿X> <蒸发皿Y> \
  --confidence 0.20 \
  --plan-output captures/pouring_plan.json
```

窗口连续稳定后按 `p` 只生成计划，不连接机械臂。先核对画面中的 `base XYZ`
和计划坐标；如存在近似恒定偏差，可增加
`--source-offset-mm <dx> <dy> <dz>`。首次悬停测试再添加
`--execute --execution-stage pregrasp`，不得直接进行带水倾倒。

当前桌面实测为 `base Z=0 mm`，一次检测结果为烧杯底部中心
`[-518, -184, 0] mm`。无需先提供蒸发皿位置即可进行源烧杯验证：

```bash
# 第一阶段：空夹爪移动到烧杯 +X 侧 100 mm，不接触烧杯
python -m src.demo.beaker_pouring \
  --depthless \
  --source-detector-weights /home/user/projects/yolo/runs/detect/beaker_train-2/weights/best.pt \
  --source-class beaker \
  --table-z-base-mm 0 \
  --confidence 0.20 \
  --execute \
  --execution-stage pregrasp

# 第二阶段：只在第一阶段对齐合格后运行；抓起、摇晃空烧杯并放回
python -m src.demo.beaker_shaking \
  --source-detector-weights /home/user/projects/yolo/runs/detect/beaker_train-2/weights/best.pt \
  --source-class beaker \
  --table-z-base-mm 0 \
  --confidence 0.20 \
  --source-offset-mm <dx> <dy> <dz> \
  --execute \
  --gripper-contact-force-n 0.5 \
  --gripper-force-n 0.8 \
  --gripper-force-step-n 0.1 \
  --allow-approximate-tcp
```

两条命令都要等待坐标稳定后在窗口按 `p`，再按终端提示确认。烧杯高度为
`100 mm`、抓取比例为 `0.4`，几何目标原本是杯底向上 `40 mm`。实机发现该高度没有给夹爪本体和标定误差留下足够桌面净空，因此规划器将 TCP 最低高度限制为 `55 mm`。下降线原本位于烧杯 base `+X` 侧 `50 mm`，扣除约 `37.5 mm` 杯半径后只剩约 `12.5 mm`，夹爪外形和视觉误差可能使下降过程压到杯子。现在高位点和低位预抓取点一起沿 base `+X` 后退到距杯心 `100 mm`，先在该位置垂直下降，再水平接近。进入任何动作前，MoveIt 模式会预检高位、抓取和圆周边界目标；若更远的预抓取点返回 `NO_IK_SOLUTION (-31)`，程序会在运动前停止，不会退回危险的近杯下降。相关距离与高度可通过配置中的 `grasp.pregrasp_distance_mm`、`grasp.min_tcp_height_above_table_mm` 和 `grasp.pregrasp_vertical_clearance_mm` 调整。用直尺测量夹爪两指中心相对这些目标
的偏差，并按 `offset = 物理真值 - 视觉检测值` 填入补偿。例如夹爪相对烧杯
实际中心多偏向 `+Y 8 mm`，则使用 `--source-offset-mm 0 -8 0`。至少在工作区
不同位置重复 5 次并取中位补偿；若偏差随位置变化，不能用固定 offset 掩盖，
应重新检查相机外参或桌面 Z。

先运行完全不使用相机的轨迹测试：

```bash
python -m src.demo.beaker_pouring \
  --config config/transparent_pouring.json \
  --mock-base-mm -500 50 0 -450 -100 0 \
  --plan-output captures/pouring_plan.json
```

### 推荐：SAM 2.1 手动提示

这种方式不依赖烧杯类别。程序启动后，先在源烧杯内部点击一次，再在目标器皿内部点击一次；SAM 会生成两个 mask。

未提供 `--weights` 时，程序默认使用 `sam2.1_t.pt` 进入该模式，因此下面命令中的 `--sam-weights` 可以省略。显式写出是为了让运行方式更清楚。

```bash
# ZED 实时定位；首次运行会下载约 75 MB 的轻量权重
python -m src.demo.beaker_pouring \
  --sam-weights sam2.1_t.pt \
  --config config/transparent_pouring.json \
  --plan-output captures/pouring_plan.json

# 单张图片验证；点击两个器皿后按 Enter 接受
python -m src.demo.beaker_pouring \
  --sam-weights sam2.1_t.pt \
  --config config/transparent_pouring.json \
  --image captures/transparent_scene.png \
  --save-preview captures/transparent_preview.png
```

### 自动检测框 + SAM 2.1

如果现有模型能把源烧杯识别为 `cup`、目标器皿识别为 `bottle`，可以自动产生提示框：

```bash
python -m src.demo.beaker_pouring \
  --sam-weights sam2.1_t.pt \
  --detector-weights /home/user/projects/yolo/best.pt \
  --source-class cup \
  --target-class bottle \
  --config config/transparent_pouring.json \
  --plan-output captures/pouring_plan.json
```

如果现场使用的检测类别不同，应把 `cup` 和 `bottle` 替换为模型中的真实类别。如果两个透明器皿只能被识别成同一类别，自动方式无法确定角色，应使用手动提示。

实时窗口操作：

- 左键第一次点击源烧杯，第二次点击目标器皿；再次点击会开始一组新提示；
- 右键清空手动提示，恢复自动检测；
- `r`：清除缓存的 SAM mask、重新拟合桌面并重新识别；
- `p`：源和目标连续稳定后生成并校验 dry-run 轨迹；
- `q` / `Esc`：退出。

固定场景中 SAM mask 会被缓存。移动器皿后必须按 `r` 或重新点击；当前 PyTorch CUDA 环境已经修复，可使用 RTX 4090 推理。

也可以用 COCO 分割模型的近似类别做快速对照测试：

```bash
python -m src.demo.beaker_pouring \
  --weights yolo11n-seg.pt \
  --source-class cup \
  --target-class bowl \
  --config config/transparent_pouring.json
```

进入实机动作前，必须依次完成：mask 人工检查、整个工作区三维坐标验收、空夹爪悬停、空杯抓取、空杯低角度倾倒、空杯完整倾倒，最后才允许少量水测试。实机入口必须按照 `pregrasp → grasp → pour → liquid` 的顺序逐级验收。

### 少量液体倾倒（完成前置验收后）

`liquid` 阶段复用已验证的抓取和圆弧路径，转场时始终保持烧杯竖直，并在目标上方先下降再倾倒。首次最多使用 `30 mL`，并需要额外输入 `LIQUID` 确认。

当前两类器皿检测模型可同时用于烧杯和玻璃结晶皿。相同权重只执行一次推理，程序结合类别、宽高比和框重叠联合分配两个角色；无法得到明确且互不重叠的一对器皿时不会进入规划：

```bash
python -m src.demo.beaker_pouring \
  --depthless \
  --source-detector-weights /home/user/projects/yolo/runs/detect/vessels_train_fixed/weights/best.pt \
  --target-detector-weights /home/user/projects/yolo/runs/detect/vessels_train_fixed/weights/best.pt \
  --source-class beaker \
  --target-class crystallizing_dish \
  --table-z-base-mm 0 \
  --confidence 0.50 \
  --execute --execution-stage liquid \
  --liquid-volume-ml 30 \
  --pour-angle-deg 60 \
  --gripper-force-n 0.8 \
  --allow-approximate-tcp
```

默认仍会拒绝使用 `approximate: true` 的 TCP；当前现场可用 `--allow-approximate-tcp` 显式接受粗测值并进行人工监督验收。未填写 `target.inner_diameter_mm` 时程序只能按外径中心估算液流落点并显示强警告；这两项都不能跳过 `pregrasp → grasp → pour → liquid` 的逐级确认。
