# ACT Day 1 — NERO SDK 接入报告

本文记录 2026-09-08 完成的静态 SDK 调查、ACT 控制封装和离线测试。尚未执行真实机械臂的连续运动测试，因此所有实机延迟、稳定性和推荐频率都保留为待测项，不以 mock 结果代替。

## 1. 现有 SDK 映射

| ACT 概念 | NERO API | 单位 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| arm qpos | `get_joint_angles().msg` | rad | `(7,)` | SDK 原始值是 Python `list[float]`；wrapper 转为 `np.float32` |
| arm target | `move_j([J1..J7])` | rad | `(7,)` | 只发送目标，不等待到位 |
| gripper state | `get_gripper_status().msg.value` | m | `(1,)` | ACT 第一版只使用开口宽度；SDK 同时提供力反馈 |
| gripper command | `move_gripper_m(value, force)` | m、N | 标量 | 项目实机抓取 baseline 使用 95 mm 张开、0 mm 闭合、1 N 保持力 |
| emergency stop | `electronic_emergency_stop()` | - | - | SDK 定义为带阻尼快速减速 |

本机 SDK 证据位于虚拟环境的 `pyAgxArm/api/constants.py` 和 `protocols/can_protocol/drivers/nero/default/driver.py`：NERO `_JOINT_NUMS = 7`，关节名为 `joint1` 到 `joint7`，七个关节均有独立限位、CAN 反馈/控制通道，并全部参与七段 MDH 正运动学。由此确认 J7 是机械臂本体关节，不是夹爪位置。夹爪通过独立的 `AGX_GRIPPER` effector 初始化。

## 2. Arm DoF / Action Dimension

- `get_joint_angles()`：七维 `[J1, J2, J3, J4, J5, J6, J7]`，单位 rad。
- `arm_qpos_dim = 7`。
- `arm_action_dim = 7`。
- `gripper_state_dim = 1`，建议表示为开口宽度（m），训练前再统一归一化到 `[0, 1]`。
- 第一版完整 observation qpos 建议：7 个本体关节 + 1 个夹爪宽度，`qpos_dim = 8`。
- 第一版完整 action 建议：7 个关节目标 + 1 个夹爪目标，`action_dim = 8`。

不要把第七关节替换成夹爪，也不要把夹爪力同时塞进第一版 action。夹爪力先作为固定安全参数，等基础示教稳定后再评估是否扩维。

夹爪 demo 默认张开到机械上限 100 mm，现有 `demo/yolo_pick_place.py` 则使用 95 mm，给上限留出 5 mm 余量。ACT wrapper 沿用已经跑通抓取流程的 95 mm。

## 3. Wrapper 行为

`src.act.robot_wrapper.NeroRobotWrapper` 完成以下约束：

- 创建机械臂后、连接 CAN 前初始化夹爪；
- 连接后使能并进入 Follower 控制模式；
- qpos 只接受有限的七维数据，统一返回 `np.ndarray(dtype=float32)`；
- 反馈只进行默认 150 ms 的短重试，缺失时抛出 `TimeoutError`；
- `send_arm_qpos()` 直接调用 `move_j()`，不调用任何等待到位函数；
- 正常 `disconnect()` 不失能机械臂。

## 4. Streaming Test

### 离线 dry-run

已实现并通过：

```bash
python -m src.act.test_joint_stream --dry-run
```

该命令不会连接 CAN，也不会发送运动目标，只展示七维输入约束和单关节正弦轨迹。若只想读取真实 qpos，可加 `--read-hardware`；它会要求输入 `CONNECT`，连接并使能机械臂，但不会发送目标。

### 5 Hz / 10 Hz / 20 Hz

| 频率 | 真机结果 | 状态 |
| --- | --- | --- |
| 5 Hz | 25 次命令/反馈，无缺失或异常；实际 5.00 Hz | 初步通过 |
| 10 Hz | 50 次命令/反馈，无缺失或异常；实际 10.00 Hz | 通过 |
| 20 Hz | 100 次命令/反馈，无缺失或异常；实际 20.00 Hz | 通过 |

实机命令见“真机验收步骤”。脚本记录每步目标、反馈、`move_j` 调用耗时、反馈读取耗时，并输出 loop mean/P95/max、实际频率、反馈缺失和异常；CSV 写入 `logs/act_joint_stream_*.csv`。

2026-09-08 首次 5 Hz 尝试成功完成真实 qpos 读取，但旧版脚本在结束回位阶段超时。该版回位循环每 0.1 秒重复发送 `q0`，可能不断重启控制器规划，而且只在回位成功后写日志，无法据此评价 5 Hz streaming。现已改为只发送一次回位目标、随后只读反馈，并在回位前保存 CSV；这次尝试标记为“无效实验”，需重新执行。

修复后 5 Hz 实验日志为 `logs/act_joint_stream_20260908_103631.csv`，机械臂正常完成测试并回到起点：

- 25 次命令均取得反馈，`None feedback = 0`，无异常；
- 实际循环频率 5.00 Hz，loop dt mean/P95/max 为 0.200003/0.200011/0.200071 s；
- `move_j()` latency mean/max 为 0.000449/0.000545 s；
- feedback latency mean/max 为 0.000108/0.000150 s；
- J1 原始 mean absolute error 为 0.004021 rad，RMSE 为 0.005407 rad，最大误差为 0.014514 rad；
- 目标与反馈零延迟相关系数约 0.927；反馈后移一个采样周期（约 0.2 s）后相关系数约 0.978、RMSE 降至 0.003201 rad，误差主要来自跟随滞后；
- J1 反馈峰峰范围约 0.040003 rad，与目标幅度一致；未测试关节的最大峰峰漂移约 0.000209 rad（J3）。

因此 5 Hz 下没有发现丢命令、反馈中断或多关节串扰，`move_j()` 可继续作为低频 ACT joint target 候选。约 0.2 s 的响应滞后必须在后续测试和 ACT temporal aggregation 设计中考虑。

10 Hz 实验日志为 `logs/act_joint_stream_20260908_104051.csv`，机械臂同样正常回到起点：

- 50 次命令均取得反馈，实际循环频率 10.00 Hz，无异常；
- loop dt mean/P95/max 为 0.100001/0.100070/0.100116 s；
- `move_j()` latency mean/max 为 0.000510/0.000844 s；
- feedback latency mean/max 为 0.000117/0.000163 s；
- J1 mean absolute error 为 0.002819 rad，RMSE 为 0.004560 rad，最大误差为 0.015413 rad；
- 目标与反馈零延迟相关系数约 0.951；反馈后移两个采样周期（约 0.2 s）后相关系数约 0.979、RMSE 为 0.003320 rad；
- J1 反馈峰峰范围约 0.040073 rad；未测试关节的最大峰峰漂移约 0.000105 rad。

5 Hz 的最佳滞后为一个采样、10 Hz 为两个采样，两者都对应约 0.2 s，说明主要延迟来自控制器/机械响应而非 Python 调度。10 Hz 相比 5 Hz 没有出现误差、抖动或串扰恶化，可进入 20 Hz 小幅对照测试。

20 Hz 实验日志为 `logs/act_joint_stream_20260908_104439.csv`，机械臂正常回到起点：

- 100 次命令均取得反馈，实际循环频率 19.9997 Hz，无异常；
- loop dt mean/P95/max 为 0.050001/0.050057/0.050119 s；
- `move_j()` latency mean/max 为 0.000392/0.000701 s；
- feedback latency mean/max 为 0.000100/0.000267 s；
- J1 mean absolute error 为 0.002412 rad，RMSE 为 0.004583 rad，最大误差为 0.016181 rad；
- 目标与反馈零延迟相关系数约 0.952；反馈后移 3–4 个采样周期（约 0.15–0.20 s）后相关系数约 0.972；
- J1 反馈峰峰范围约 0.040090 rad；未测试关节的最大峰峰漂移约 0.000140 rad。

20 Hz 下调用延迟远小于 50 ms 控制周期，反馈完整，跟踪误差与 10 Hz 基本一致。第一版 ACT 因此采用 20 Hz 作为采集和控制频率；该结论只覆盖当前平滑、小幅、单关节轨迹。

## 5. `move_j` 适用性

- 静态源码结论：`move_j()` 校验并组装消息，然后调用 `_send_msgs()`，没有等待机械臂到位，因此 Python 调用语义是非等待式。
- 真机结论：5/10/20 Hz 小幅单关节测试中连续目标均能生效，没有丢反馈或明显串扰，调用本身不阻塞；反馈约滞后 0.15–0.20 s。
- 当前建议频率：第一版 ACT 使用 20 Hz；图像、qpos、gripper state 和 action 使用同一时间基准记录。
- 最终 verdict：`move_j()` 适合作为第一版 ACT 的 joint target 接口，但 Safety Layer 必须限制逐步关节变化，并对 action chunk 做平滑/时间聚合。

SDK 还提供 `move_js()`，但其源码明确警告“无平滑、无轨迹规划、极高机械冲击/振荡风险”，因此 Day 1 不调用它，也不能因为名字像 servo 就直接替换。未发现一个已文档化的“停止当前运动但保持使能”普通接口。

另一个安全发现是：SDK 虽在配置中提供 NERO 每个关节的标称限位，但驱动的 `joint_limits_enabled` 默认是 `False`，`move_j()` 默认只按通用 ±2π 裁剪。首次读取的 J2 为 1.8309 rad（约 104.9°），高于配置预设的 100°，但不会被默认 `move_j()` 裁剪，因此这不是该次回位失败的原因。后续 Safety Layer 必须显式检查关节限位；在确认零点和真实硬件限位前，不应贸然打开 SDK 自动裁剪。

## 6. 安全 / Stop 语义

- `electronic_emergency_stop()`：发送带阻尼急停命令，用于异常和人工中断。
- `disable()`：关节电机失能，机械臂可能卸力或下坠，只能在人工托住并明确确认时使用。
- `disconnect()`：停止 SDK 后台线程并关闭 CAN 通信；不是停止运动命令，也不保证机械臂保持安全状态。
- 脚本正常结束会先下发起始 qpos 并检查回到容差内，再断开；异常或 `Ctrl+C` 会触发电子急停。

## 7. Leader Mode

本机 SDK 将 `set_leader_mode()` 明确定义为“leader arm zero force drag mode”。`get_leader_joint_angles()` 与普通关节反馈一样返回七维 `[J1..J7]`，单位 rad；项目的 TCP 标定程序也已经用它完成稳定拖动采样。

Day 2 推荐使用 **Leader/drag teaching 采集机械臂示教**，原因是它直接产生与 ACT arm qpos/action 对齐的七维关节序列，且项目已有实机经验。夹爪不能从本体七维数据中推断，应在示教期间用键盘或手柄单独控制开/合，并同步记录夹爪宽度反馈。若拖动速度、反馈频率或夹爪同步不满足要求，再退回完整 keyboard/gamepad teleop。

## 8. 真机验收步骤

先确认底座固定、工作区清空、J1 在当前位置可安全小幅运动并准备实体急停：

```bash
# 1. 默认 dry-run，不连接机械臂
python -m src.act.test_joint_stream --dry-run

# 2. 可选：只连接并读取七维 qpos，不发送运动目标
python -m src.act.test_joint_stream --dry-run --read-hardware

# 3. 第一轮仅测 J1、5 Hz、±0.02 rad、5 秒
python -m src.act.test_joint_stream \
  --execute --joint-index 0 --frequency 5 --duration 5 --amplitude-rad 0.02

# 4. 只有 5 Hz 通过后才依次改为 10、20
python -m src.act.test_joint_stream \
  --execute --joint-index 0 --frequency 10 --duration 5 --amplitude-rad 0.02
python -m src.act.test_joint_stream \
  --execute --joint-index 0 --frequency 20 --duration 5 --amplitude-rad 0.02
```

执行模式需要输入精确口令 `STREAM`。若出现振动、异响、状态报警、轨迹突跳或不能返回起点，应立即按实体急停，不再提高频率，并保留 CSV 和终端输出用于分析。

## 9. Day 1 验收与剩余问题

已完成：

- wrapper API 与七维/有限值校验；
- 夹爪反馈、开合与参数范围封装；
- 默认安全 dry-run 和显式实机确认；
- mock 单元测试，不连接 CAN；
- SDK 源码级关节维数、Leader 模式和停止语义调查。
- 5/10/20 Hz 小幅单关节真机 streaming 测试，均无反馈丢失或异常；
- `move_j()` 非等待式行为和第一版 20 Hz 控制频率确认。

后续仍需验证：

- Leader 模式下图像、关节和夹爪反馈能否稳定同步到 20 Hz；
- 多关节示教轨迹以及 ACT action chunk 下的误差和安全边界；
- SDK 反馈对象的消息时间戳/新鲜度检查，避免把读取缓存耗时误当成传感器延迟；
- 示教时夹爪事件与图像/关节时间戳的同步策略。

Day 2 应从一段短 Leader 示教记录器开始：同步保存时间戳、左目 RGB、七维 arm qpos、单维 gripper width 和键盘/手柄夹爪动作；在此之前不要开始 ACT 训练。
