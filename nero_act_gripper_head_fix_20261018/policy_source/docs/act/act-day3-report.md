# ACT Day 3 — Dataset Validation & Replay Report

本文记录 NERO 示教数据的严格离线验收、可视化、replay 安全层和时间对齐分析。当前环境未执行真机运动，因此不宣称 replay 已成功；真机闭环仍需操作者按本文步骤逐级验收。

## 1. Day 1/2 已确认输入

| 项目 | 真实定义 |
| --- | --- |
| arm qpos | 7 维 `float32`，单位 rad |
| gripper state | 独立 1 维实际宽度，单位 m |
| action | 7 维 Leader joint target + 1 维夹爪宽度目标 |
| recording frequency | requested 20 Hz |
| demonstration | Leader 拖动；夹爪动作期间短暂进入 Follower |
| timestamp | `float64 time.monotonic()`，图像取回后记录 |
| episode path | `data/act/episode_N.hdf5` |

## 2. Episode Validation

验收命令：

```bash
python -m src.act.validate_episode data/act/*.hdf5 \
  --manifest data/act/manifest.json
```

默认门槛包括完整 schema、dtype/shape/长度、有限值、严格递增时间戳、非空白图像抽样、NERO 标称关节限位，以及相邻 action 最大变化不超过 0.08 rad。

| Episode | T | Duration (s) | Hz | Max action step (rad) | Late | Offline status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `episode_001` | 242 | 12.020 | 20.049 | 0.0297 | 0 | rejected：qpos 冻结但 action 运动 |
| `episode_1` | 273 | 13.571 | 20.043 | 0.0242 | 0 | valid |
| `episode_2` | 228 | 11.321 | 20.052 | 0.0196 | 0 | valid |
| `episode_3` | 356 | 17.720 | 20.034 | 0.0283 | 0 | valid |
| `episode_4` | 260 | 12.920 | 20.047 | 0.0189 | 0 | valid |
| `episode_5` | 303 | 15.070 | 20.040 | 0.0281 | 0 | valid |
| `episode_6` | 220 | 10.921 | 20.053 | 0.0323 | 0 | valid |
| `episode_7` | 283 | 14.070 | 20.043 | 0.0302 | 0 | valid |
| `episode_8` | 230 | 11.420 | 20.052 | 0.0245 | 0 | valid |
| `episode_9` | 197 | 9.770 | 20.061 | 0.0234 | 0 | valid |
| `episode_10` | 280 | 13.921 | 20.042 | 0.0328 | 0 | valid |
| `episode_11` | 292 | 14.522 | 20.038 | 0.0251 | 0 | valid |
| `episode_12` | 270 | 13.423 | 20.041 | 0.0234 | 0 | valid |
| `episode_13` | 261 | 12.972 | 20.043 | 0.0383 | 0 | valid |
| `episode_14` | 235 | 11.670 | 20.051 | 0.0266 | 1 | valid；仅一次 late cycle |
| `episode_15` | 228 | 11.321 | 20.050 | 0.0223 | 0 | valid |
| `episode_16` | 268 | 13.320 | 20.045 | 0.0276 | 0 | valid |
| `episode_17` | 261 | 12.970 | 20.046 | 0.0394 | 0 | valid |
| `episode_18` | 246 | 12.220 | 20.048 | 0.0310 | 0 | valid |
| `episode_19` | 274 | 13.622 | 20.041 | 0.0206 | 0 | valid |
| `episode_20` | 294 | 14.621 | 20.039 | 0.0324 | 0 | valid |

20 条有效数据合计 5259 帧、261.366 秒；频率范围为 20.034–20.061 Hz。抽样图像标准差为 73.75–85.12，没有发现空白帧。离线 valid 仍需人工确认画面中的任务确实成功，并经 replay 验收后才能成为最终训练集。

## 3. Qpos / Action 与 Tracking Error

`episode_plot.py` 会分别生成 qpos、action 和 `action[:7] - qpos`：

```bash
python -m src.act.episode_plot data/act/episode_10.hdf5 \
  --output-dir artifacts/act_day3
```

已生成 episode 1、10、20 的代表性图。曲线连续，未发现超过 0.0394 rad 的单步变化。20 条新数据的同周期 arm tracking MAE/P95/max 都是 0，因为 recorder 将同一份模式相关关节反馈同时保存为 qpos 和 Leader action。这证明数据没有发生额外复制错位，但该数值不是 Follower 控制误差。

## 4. Time Alignment

对 `E(lag) = mean(abs(action[t] - qpos[t+lag]))`、lag `-3..+3` 的有效数据加权结果：

| lag (step) | -3 | -2 | -1 | 0 | +1 | +2 | +3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAE (rad) | 0.003063 | 0.002041 | 0.001021 | 0.000000 | 0.001021 | 0.002041 | 0.003063 |

离线最佳候选是 0 step，但这是采集 schema 的必然结果，不能推导 `move_j` 的物理响应 lag。训练 Dataset 暂时使用 `observation[t] → action[t:]`，不应用 `start_ts-1` 或固定 ±1/±2 hack。真机 replay 日志获得 command/feedback 后再评估控制 lag；Day 1 小幅流式测试提示物理响应约 3–4 帧（150–200 ms）。

## 5. Replay 实现与安全边界

`replay_episode.py` 已支持：

- 默认 dry-run，只有 `--execute` 才连接硬件；
- `[start-step, end-step)` 部分区间和 0.25–1.0x 时间缩放；
- HDF5 validation、8维 action、有限值、标称关节限位、0.08 rad 单步阈值；
- 当前位置到首帧默认最大差异 0.12 rad；超出时拒绝，绝不自动跳到起点；
- 可显式使用 `--move-to-start`，经独立 `MOVE_START` 确认后以默认 10% 速度发送一次起点关节目标，并等待误差小于 0.02 rad；该动作不含碰撞检测；
- replay 控制器速度上限默认 10%，可用 `--replay-speed-percent` 显式调整；`--speed-scale` 只缩放记录时间轴；
- 每帧读取真实 qpos，跟踪误差超过 0.25 rad、反馈缺失或其他异常时电子急停；
- 第8维作为夹爪 target，仅在目标发生变化时发送，默认力 1 N；
- CSV 记录 command、feedback、tracking error 和调用延迟。

已完成的离线 dry-run：

```text
episode_1, steps [0,20), speed 0.25x：preflight passed
```

## 6. 真机 Replay 验收步骤

先把机械臂安全地拖到该示教的起始姿态附近；脚本不会自动 move-to-start。每一步均在机械臂旁操作并准备实体急停：

```bash
source /home/user/nero/bin/activate
cd /home/user/projects/nero

# 1. 离线检查
python -m src.act.replay_episode data/act/episode_1.hdf5 \
  --dry-run --start-step 0 --end-step 20 --speed-scale 0.25

# 2. 第一轮：仅 20 帧、0.25x
python -m src.act.replay_episode data/act/episode_1.hdf5 \
  --execute --move-to-start --start-step 0 --end-step 20 --speed-scale 0.25

# 3. 前一步无异常后扩大到 50 帧
python -m src.act.replay_episode data/act/episode_1.hdf5 \
  --execute --start-step 0 --end-step 50 --speed-scale 0.25

# 4. 再验证 50 帧、0.5x
python -m src.act.replay_episode data/act/episode_1.hdf5 \
  --execute --start-step 0 --end-step 50 --speed-scale 0.5
```

只有动作、夹爪时序、路径和日志均正常后，才允许完整 episode 0.5x。出现异响、突跳、卡顿、碰撞趋势、跟踪误差停止或反馈中断时，不得扩大范围。

## 7. Valid / Rejected 规则

自动规则记录在 `data/act/manifest.json`。最终训练集必须同时满足：自动检查 valid、操作者确认画面任务完整、真机 replay 大致复现且未触发安全停止。

- 离线 valid：`episode_1`–`episode_20`。
- rejected：`episode_001`，原因是 qpos 全程冻结，不能表达示教 observation。
- 待完成：至少 1–3 条代表性 episode 的真机 replay 人工验收；完成前 manifest 的 valid 仅表示离线有效。

## 8. 测试结果与 Day 4 Readiness

全仓库 77 项离线测试通过，覆盖 validation、范围选择、速度缩放、自动移动起点、action shape、NaN/Inf、关节限位、单步突跳、反馈缺失和跟踪发散，mock 测试不会连接 CAN。

当前已满足 Day 4 数据结构、连续性、频率和离线工具前置条件，但尚未满足真机 replay 闭环条件。Day 4 最先做的不是训练，而是完成上述 20→50 帧的 episode 1 replay，检查 CSV；随后对一条完整 episode 验收夹爪事件。闭环通过后再开始单 episode overfit test。
