# ACT Day 6 — Real Robot Inference Report

Day 6 的软件链路已经完成离线实现和 mock 验收。真实相机 + 真实关节反馈 dry-run、短窗口运动和完整 rollout 尚未执行，因此本报告明确标记为“等待分阶段实机验收”，当前不能进入 Day 7 成功率测试。

## 1. Inputs from Day 5

| 项目 | 值 |
| --- | --- |
| checkpoint | `checkpoints/act_day5/policy_best_val.ckpt` |
| selected epoch | 291 |
| validation L1 | 0.152449 |
| state / action | 8 / 8（7 joints + gripper） |
| chunk size | 20 |
| camera | `camera_0`，RGB `[1,1,3,90,160]` |
| inference frequency | 20 Hz |

启动时严格比对 checkpoint 的 state/action 维数、chunk size、camera names 和 image size；权重使用 `strict=True` 加载。normalization 的四个数组必须为有限 `float32[8]`，标准差必须大于零。

## 2. Runtime Pipeline

```text
ZED 左目 RGB + arm/gripper feedback
  -> 与训练完全相同的 resize/scale 和 qpos normalization
  -> ACT 输出 [20,8]
  -> action denormalization
  -> first action / temporal ensemble
  -> SafetyFilter
  -> dry-run 日志，或显式 --execute 后发送 SDK 命令
```

`--dry-run` 是默认模式。它仍会打开相机、连接并使能机械臂以读取真实反馈，但不会调用 `move_j` 或夹爪命令。实机异常会停止循环；execute 模式还会触发电子急停。

## 3. Preprocessing / Normalization

训练 dataset 与实时入口共同调用 `preprocess_images()`：输入是旋转校正后的 RGB `uint8[N,H,W,3]`，转换为 `[N,3,90,160]`、缩放到 0..1；ImageNet normalization 仍由 policy 内部完成。qpos 和 action 使用 Day 5 train-only `dataset_stats.pkl`，不存在运行时重新统计。

## 4. Safety Layer

- 标称七关节限位来自 Day 3 验证器；轻微超限 clamp，超过限位 0.1 rad 直接拒绝；
- 单周期关节变化上限为 0.02 rad，速度上限为 0.4 rad/s，两者取更严格值；
- `--speed-scale 0.25` 时实际单步最大变化为 0.005 rad；
- 超出 demonstration action range 0.1 rad 时记录 anomaly；
- 初始姿态超出训练范围加 0.15 rad margin 时拒绝启动；
- 任意 NaN/Inf、反馈失败或跟踪误差超过 0.25 rad 时停止；
- 夹爪输出不再把网络连续值直接下发，而是通过 35/60 mm 迟滞映射到 0/95 mm，仅在状态变化时发送一次。

这些约束解决了 Day 5 原始夹爪预测 `-31.6..121.1 mm` 越界和阈值附近反复开关的问题。

## 5. Dry-Run Result

| 项目 | 状态 |
| --- | --- |
| mock camera/robot loops | 2，PASS |
| command sends | 0 |
| real ZED + qpos loops | 50，PASS |
| real model latency | mean 23.7 ms，P95 40.0 ms（旧首帧 cold start 213.6 ms） |
| real safety anomalies | 0 |

2026-09-09 首次真实 dry-run 在进入循环前被初始 OOD 检查拒绝，记录为 0 step；相机、CUDA checkpoint 和机械臂连接均正常。旧错误没有列出越界关节，现已改为输出每个越界关节的实测值与允许范围，并给 replay 工具增加只移动到示教起点、不回放 action/夹爪的 `--move-only` 模式。

移动到 episode 1 起点后的第二次 dry-run 完成 50 steps：实际 `19.9996 Hz`，无 joint-limit clamp、range anomaly、NaN 或反馈失败。239 个 delta clamps 来自机械臂在 dry-run 中保持不动，而模型稳定请求 J2/J3/J4/J7 离开起点；逐关节平均请求差为 `[0.0024, 0.0436, 0.0142, 0.0113, 0.0053, 0.0003, 0.0070] rad`，属于预期的速度限制，不是越界预测。first-action mean/P95/max jitter 为 `0.002209/0.004464/0.006153 rad`；temporal 候选为 `0.000854/0.002042/0.002698 rad`。

第二次 dry-run 唯一 late cycle 来自首帧 CUDA warm-up（model 213.6 ms；其余 P95 40.0 ms）。实机下发前已增加未计入控制循环的 CUDA warm-up，并把调度器改成超时后跳过错过的 deadline，禁止为“追赶频率”连续突发下发命令。该调度变更后需要重新做一次 50-loop dry-run。

执行命令：

```bash
source /home/user/nero/bin/activate
cd /home/user/projects/nero
python -m src.act.run_act_real \
  --config configs/act_nero_day6.yaml \
  --dry-run --max-steps 50
```

输入 `CONNECT` 后机械臂会使能但不会运动。若初始姿态被判为 OOD，应先取消并把机械臂移到示教起始区域，不要扩大 margin 绕过检查。

推荐使用已经通过验收的 episode 1 起点：

```bash
python -m src.act.replay_episode data/act/episode_1.hdf5 \
  --execute --move-to-start --move-only \
  --start-step 0 --end-step 1
```

依次输入 `REPLAY` 和 `MOVE_START`。该命令只低速发送一次七关节起点并等待到位，不控制夹爪、不继续回放；仍然没有碰撞规划，执行前必须人工确认从当前位置到目标关节姿态的完整扫掠路径。

## 6. First-Action-Only Rollout

| Rollout | Steps | Result | Safety Triggers | Notes |
| --- | ---: | --- | ---: | --- |
| mock | 2 | PASS | 0 | 已确认 dry-run 零发送 |
| real-1 | 10 | FAIL | delta clamp 54 | SDK 接收命令但 qpos 七关节全程不变 |

第一轮命令：

```bash
python -m src.act.run_act_real \
  --config configs/act_nero_day6.yaml \
  --execute --max-steps 10 --speed-scale 0.25
```

## 7. Temporal Aggregation

对 validation episodes 8、10、15、16 的全部 1006 个 observation 做了无跨 episode 边界的离线 A/B。first-action 基线为：MAE `0.011613`，平均/P95/最大关节 jitter 为 `0.008100 / 0.022391 / 0.077652 rad`，夹爪二值准确率 `93.44%`。

| decay | MAE | mean jitter (rad) | max jitter (rad) | gripper accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.013422 | 0.005624 | 0.021392 | 91.35% |
| 0.20 | 0.011490 | 0.005885 | 0.025773 | 91.95% |
| **0.50** | **0.011199** | **0.006441** | **0.039301** | **92.64%** |
| 1.00 | 0.011378 | 0.007016 | 0.057407 | 93.04% |

采用 decay `0.5`：相对 first-action，离线 MAE 降低约 3.6%，平均 jitter 降低约 20.5%，最大 jitter 降低约 49.4%；夹爪准确率降低约 0.8 个百分点，因此运行时仍必须保留夹爪迟滞。默认 first-action-only，只有基础短窗口通过后才加 `--temporal-ensemble`。

## 8. Full Rollout Results

尚未执行。至少完成一次 first-action 10-step 和一次 temporal 10-step 后，才允许使用 `--extended` 增加到 20–50 step；完整抓取放置还需要现场确认抓取、抬升、搬运和放置均成功。

## 9. Failure Analysis

首次 10-step execute 的 summary 虽显示 `passed`，但 CSV 复核发现 qpos 起点和终点完全一致；上一安全目标与反馈的最大差保持约 0.005 rad，因此该轮按 `CONTROL_NO_PROGRESS` 判定失败。旧 safety 每周期相对未移动的反馈生成目标，导致控制器连续收到同一个很小的目标。

已改为相对上一条安全命令按 0.005 rad 递进，同时限制命令最多领先实测关节 0.03 rad。新增 progress watchdog：累计请求至少 0.01 rad、反馈仍小于 0.001 rad 时停止，避免再次把“SDK 无报错”误判为闭环成功。

修复后第一次重试正确触发 watchdog：9 条已记录命令中目标递进到 0.03 rad，反馈仍只有 J1 约 `0.00005 rad` 的噪声变化。日志中的 `feedback_failures=1` 是旧分类条件把错误文本“关节反馈仅变化”误认为反馈丢失；现已更正为 `feedback_failures=0, no_progress_rejections=1`。由于 replay 在 10% 速度已实机工作，而 Day6 使用 5%，下一轮把控制器速度上限调整为 10%；动作本身仍受 0.005 rad/周期和最大领先 0.03 rad 双重限制。

10% 速度重试仍被 watchdog 拒绝：9 条记录中 J3/J5/J7 的安全目标跨度均为 0.025 rad，编码器只有约 `0.00005 rad` 噪声，说明命令在变化但机械臂没有执行。由此排除单纯 5% 速度死区；在继续修改 ACT 前，必须重新运行 Day1 的 5 Hz、J1、±0.02 rad 单关节流式测试，区分控制器状态异常和 20 Hz 多关节 `move_j` 重规划问题。

随后 Day1 单关节测试在 5 Hz 和 20 Hz 均成功，证明 CAN、Follower、编码器反馈、`move_j` 和 20 Hz 频率正常。增加 `--speed-percent 10` 后的 20 Hz 测试也成功：100 帧无错误或反馈丢失，实际频率 `19.9998 Hz`，J1 命令/反馈跨度 `0.04000/0.04009 rad`，误差 mean/P95/max 为 `0.00206/0.01072/0.01458 rad`。因此 10% 控制器速度不是无运动原因。

隔离测试与 Day6 的一个差异是：测试在 `set_speed_percent` 后等待 0.1 s，而 Day6 原先立即发送首条关节命令。NERO SDK 的设速操作会发送一帧 `move_mode=255` 的模式控制报文，因此所有 ACT 设速入口现已统一经 `NeroRobotWrapper.set_speed_percent()` 等待 0.1 s，再发送运动目标。加入等待后的 Day6 重试仍然无运动（9 帧、37 次 delta clamp、12 次 command-lead clamp），排除设速报文尚未稳定这一原因。

Day1 的历史回位问题已经证明：重复发送尚未完成的 `move_j` 会重启控制器轨迹规划。单关节正弦流可以在 20 Hz 工作，但 Day6 同时改变多个关节，规划负担和行为不同。当前采用保守过渡方案：模型、相机和安全检查继续运行在 20 Hz，普通 `move_j` 仅以 5 Hz 下发；不切换到 SDK 明确警告可能产生冲击和振荡的 `move_js`。日志新增 `command_sent` 和汇总 `commands_sent`，无进展检查仍然保留。

若 5 Hz 多目标测试仍无运动，使用单次目标诊断隔离控制器规划：

```bash
python -m src.act.run_act_real \
  --config configs/act_nero_day6.yaml \
  --execute --single-action-test --max-steps 50 --speed-scale 1.0
```

输入 `MOVE_ONCE` 后只发送第一条经过 safety 的实时预测关节目标（不控制夹爪），随后约 2.5 秒只采集图像和反馈。成功时 `commands_sent=1` 且 qpos 有变化；失败时程序输出 `arm_status/motion_status`。

单次预测目标已完成实机验证，说明 action 的单位、维度、关节顺序以及控制器对该目标的接受均正常。后续 5 Hz 多目标测试发送 3 条命令后仍无反馈进展，控制器状态保持 `NORMAL/MOVE_J/REACH_TARGET_POS_SUCCESSFULLY`。

按现场操作要求，Day6 普通运行失败现在会停止预测下发，读取当前 qpos，恢复 Follower 并只发送一次当前位置保持命令；只有读取或保持失败时才回退到电子急停。用户主动按 `Ctrl-C` 仍执行电子急停。该策略只应用于 Day6 推理，不改变示教和 replay 的既有急停行为。

保持策略实机验证成功。该轮日志同时发现下发由 20 Hz 降至 5 Hz 后，安全层仍按 20 Hz 和 `speed_scale=0.25` 计算增量，使每条命令只有约 `0.005 rad`；成功的单次目标测试则约为 `0.02 rad`。现按下发抽取倍数同步换算单步 scale（本配置为 `0.25 × 4 = 1.0`），因此 5 Hz 下每条最大仍为 `0.02 rad`，等效速度保持约 `0.1 rad/s`，且继续受 `max_delta_rad=0.02` 和 command-lead 限制。

修正步长后的日志显示，第 0/4/8 步分别在约 0/0.2/0.4 s 下发目标，但 10-step 测试总长只有约 0.43 s；成功的单次目标日志中，编码器首次达到 `0.001 rad` 变化需要约 0.67 s。因此固定 5 Hz 仍会在控制器开始运动前覆盖目标。当前改为反馈门控：任何时刻最多存在一条待完成目标，只有编码器进入 `0.005 rad` 到位容差后才允许发送下一条；超过 5 s 未到位则普通失败并保持当前位置。循环结束时也会等待最后一条目标完成，不再用固定 10-step 时长提前判定无进展。

## 10. Runtime Timing

首次 execute 的 model mean/P95/max 为 `22.6/38.4/48.3 ms`，send mean/max 为 `0.69/1.30 ms`，processing mean/P95 为 `29.2/49.3 ms`。一次 64 ms processing overrun 被安全跳过，10-step 短窗口实际频率为 17.50 Hz，没有发生追赶式突发下发。

## 11. Safety Trigger Statistics

首次 execute：joint-limit clamp 0、delta clamp 54、range anomaly 0、late cycle 1、NaN 0、feedback failure 0。该轮 qpos 无运动，不能因为其他统计正常而验收通过。

## 12. Regression Tests

新增测试覆盖：checkpoint/stats 维数不匹配、dry-run 零发送、NaN/Inf 拒绝、严重/轻微关节超限、速度与 delta clamp、命令领先限制、训练范围 anomaly、初始 OOD、夹爪迟滞、temporal 权重、episode 边界、超时后禁止突发追赶和设速后的控制器等待。ACT 相关 48 项测试通过；未连接 CAN。

## 13. Day 7 Readiness

**当前不具备 Day 7 条件。** 软件实现和离线 A/B 已通过，但仍缺少：50-loop 真实 dry-run、10-step first-action、10-step temporal、至少一次受监督完整 rollout，以及对应 timing/safety 统计。完成这些阶段且不存在频繁 clamp、反馈丢失或不可控运动后，才能开始 20 次正式成功率评估。
