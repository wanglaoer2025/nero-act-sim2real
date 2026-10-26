# ACT Day 2 — 示教数据采集报告

本文记录 NERO 单臂 Leader 拖动示教的 `record → save → inspect` 数据链。离线实现、相机 dry-run 和三条真实 pick-place episode 验收已经完成。

## 1. Day 1 输入

| 项目 | 结论 |
| --- | --- |
| arm qpos | 7 个机械臂关节，`float32 (7,)`，单位 rad |
| gripper state | 独立开口宽度，`float32 (1,)`，单位 m |
| 完整 action | 7 个关节目标 + 1 个夹爪宽度目标，`float32 (8,)` |
| 频率 | 20 Hz；Day 1 已用小幅 J1 轨迹完成实机验证 |
| 示教方式 | Leader 零重力拖动；`o/c` 触发短暂 Follower 保持并控制夹爪 |
| 已知延迟 | Day 1 流式控制中关节目标到反馈约 0.15–0.20 s；不在采集阶段预先补偿 |

## 2. Demonstration 定义

### Observation

每个时刻保存：

- `camera_0[t]`：ZED 左目图像，倒装校正后转换为 RGB；
- `qpos[t]`：拖动阶段使用 `get_leader_joint_angles()`；夹爪 Follower 保持阶段使用 `get_joint_angles()`。两者都是各自模式下的实时物理关节角；
- `gripper_state[t]`：夹爪实际开口宽度，独立于 qpos；
- `timestamp[t]`：图像取回后立即读取的 `time.monotonic()`。

### Action

NERO SDK 的 Leader 反馈既是当前被拖动机械臂的物理关节状态，也是可用于控制 Follower 的示教目标。第一版保留同一采样周期的原始示教目标：

```text
action[t] = concat(leader_qpos[t], gripper_target[t])
```

默认 `--action-lead-ms 0`，不在采集阶段用 Day 1 的控制延迟移动标签。这样保留原始 Leader action，后续可以依据真实 episode 的时间戳离线评估时移。参数仍允许进行显式实验；启用前视时末尾用最后一个 action 填充，不截断数据集。

夹爪 target 是程序通过 `o/c` 实际下发的连续宽度（m），夹持力是固定安全参数，不进入第一版 action。

## 3. 录制流程

状态机为：

```text
IDLE → READY → RECORDING → SAVING → DONE
```

- 默认或 `--dry-run`：只打开相机、构造模拟 buffer 并校验 shape，不连接机械臂、不控制夹爪、不写 HDF5；
- `--execute`：输入 `RECORD` 后才连接、使能并进入 Leader；
- 连接后先等待 1 秒让 CAN 后台反馈初始化，再检查通信并使用 8 秒使能窗口；该流程与已验证的 `src.arm.enable` 一致；
- 若控制器反馈 `EMERGENCY_STOP`，程序会在 enable 前明确退出；不会自动清除急停，必须人工排除原因、释放实体急停并运行 `src.arm.reset`；
- READY：`r/R` 开始，`q/Q/x/X/Esc` 取消，`e/E` 电子急停；
- RECORDING：`o/O` 打开夹爪，`c/C` 闭合夹爪，`空格/q/Q` 完成，`x/X/Esc` 取消，`e/E` 急停；按开合键前必须停止拖动，程序会进入 Follower 保持（默认 1 s）、发送夹爪命令并继续采样，然后自动恢复 Leader；窗口会显示实际/目标夹爪宽度；
- 正常完成先恢复 Follower 位置保持再保存；取消不生成正式文件；异常、Ctrl-C 或反馈失败会请求电子急停且不保存不完整数据。

录制循环使用累计 `next_tick` 维持固定周期，而不是每轮固定 sleep。实际 dt、平均频率、P95、最大 dt 和 late cycles 会写入文件属性。

## 4. HDF5 Schema

```text
episode_N.hdf5
├── /action                              float32 [T, 8]
├── /timestamp                           float64 [T]
└── /observations
    ├── /qpos                            float32 [T, 7]
    ├── /gripper_state                   float32 [T, 1]
    └── /images/camera_0                 uint8   [T, H, W, 3]
```

根属性包含 `sim=False`、task、requested/actual frequency、action 语义与前视量、RGB/旋转信息、单位、采集顺序和周期统计。图像按帧分块并用 HDF5 LZF 无损压缩。qpos 与 action 前 7 维都来自同周期 Leader 反馈；ACT action chunk 的后续行仍包含未来轨迹。

## 5. 控制频率

- requested：20 Hz；代码拒绝超过 Day 1 已验证的 20 Hz；
- 三条有效 episode 的实际频率为 20.034–20.047 Hz；
- 各条 P95 dt 为 0.0507–0.0512 s，最大 dt 为 0.0518–0.0566 s，均无 late cycle。

## 6. Camera

- 来源：ZED `VIEW.LEFT`；双目深度在 Day 2 不启用；
- 输入模式：HD720 @ 30 FPS；默认缩放宽度 640，高度按相机原始宽高比计算；
- 保存格式：`uint8 RGB`；
- 安装修正：保存前旋转 180°；
- 时间戳：图像取回后的主机 monotonic 时间，不是 ZED 硬件曝光时间。

2026-09-08 已完成真实 ZED 的只读 dry-run：相机 S/N 39602891 以 HD720 @ 30 打开，模拟 episode 得到 `uint8 RGB (10, 360, 640, 3)`，qpos/action/gripper 分别为 `(10, 7)/(10, 8)/(10, 1)`；未连接机械臂且未写 HDF5。

## 7. Gripper

- state：实际连续宽度，`(1,)`，m；
- action：键盘 `o/c` 下发的目标连续宽度，作为 action 第 8 维；NERO Leader 模式会忽略夹爪控制帧，因此程序会短暂切到 Follower 执行；
- 初始 target：进入 READY 后读取的实际夹爪宽度，避免凭空假设已经张开；
- force：默认 1 N，可由命令行调整，但当前不作为训练维度。

## 8. Recorded Episodes

| Episode | T | Duration | Actual Hz | Valid | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| `episode_001.hdf5` | 242 | 12.020 s | 20.049 | 无效 | RGB/时间正常；普通 qpos 全程冻结；夹爪无事件，不作为训练数据 |
| `episode_3.hdf5` | 356 | 17.720 s | 20.034 | 有效 | 无 late cycle；3 次夹爪目标切换；实际宽度约 0–94.9 mm；模式切换附近关节连续 |
| `episode_4.hdf5` | 260 | 12.920 s | 20.047 | 有效 | 无 late cycle；夹爪 94.9→29.4→94.9 mm；切换处最大关节步长约 0.0039 rad |
| `episode_5.hdf5` | 303 | 15.070 s | 20.040 | 有效 | 无 late cycle；夹爪 94.9→29.3→94.9 mm；切换处最大关节步长约 0.0015 rad |

`episode_3/4/5.hdf5` 合计 919 帧、45.710 s，是当前三条有效真机示教。三条数据都记录到了真实夹爪闭合和重新张开；闭合后宽度稳定在约 28.3–29.4 mm。Leader/Follower 切换附近没有发现明显关节跳动。最终任务是否在画面中完整成功仍由现场操作者负责确认。

## 9. 真机采集和检查

先在机械臂旁完成一次相机 dry-run：

```bash
source /home/user/nero/bin/activate
cd /home/user/projects/nero
python -m src.act.record_episode --dry-run --frequency 20
```

确认画面方向、RGB 和 shape 正常后，清空工作区、扶稳机械臂并准备实体急停，依次录制三条短而完整的同一任务示教：

```bash
python -m src.act.record_episode --execute --episode-id 0 --frequency 20 --task-name pick_place
python -m src.act.record_episode --execute --episode-id 1 --frequency 20 --task-name pick_place
python -m src.act.record_episode --execute --episode-id 2 --frequency 20 --task-name pick_place
```

每条录制完成后立即检查：

```bash
python -m src.act.episode_inspect data/act/episode_0.hdf5
python -m src.act.episode_inspect data/act/episode_1.hdf5
python -m src.act.episode_inspect data/act/episode_2.hdf5
```

只保留画面正常、任务完整、夹爪事件正确、关节连续、长度一致、时间戳单调且没有 NaN/Inf 的 episode。失败数据不要用 `--overwrite` 掩盖，应先人工判定并另存到 rejected 目录或使用新的 episode id。

## 10. 已知时间对齐风险

1. ZED 图像与 NERO SDK 反馈没有共享硬件时钟；当前顺序是 `image → timestamp → leader qpos → gripper state`。
2. `timestamp` 是主机取帧完成时间，不是曝光时间；相机内部缓冲可能引入未知延迟。
3. SDK 反馈调用很快不代表返回值一定是同一时刻的新 CAN 帧。
4. Day 1 的 150–200 ms 是 `move_j` 流式控制延迟，不应直接当作 Leader 示教标签时移；当前原样保留同周期 action。
5. ZED 为 30 FPS、采集为 20 Hz；调度压力和重复/陈旧画面需要在真实 episode 中检查。

这些风险不会被 mock 测试证明消失。训练前应绘制 qpos/action、检查夹爪事件对应画面，并用真实 episode 估计最佳时移。

## 11. 离线验收与 Day 3 建议

Day 2 离线侧已经覆盖 HDF5 保存/读取、shape、长度、NaN/Inf、时间戳单调性、默认禁止覆盖、同周期 Leader action、冻结 qpos 检测、CAN 初始化等待，以及可选前视的末端填充。全仓库 66 项离线测试全部通过。`episode_inspect.py` 会在发现结构或数值错误时以非零状态退出。

Day 3 最优先事项是人工复核三条有效 episode 的完整画面和任务成功性，然后扩大到同一任务的 20–30 条高质量示教，并从真实数据估计相机—关节—action 的时间偏移。训练或回放前还应实现带关节限位、单步变化限制和急停保护的 replay/safety layer。
