# ACT Day 5 — Formal Training & Offline Evaluation Report

Day 5 已完成 NERO ACT 正式 GPU 训练和全验证集离线评估。初次训练发现夹爪连续回归越界且二值准确率仅 85.0%；随后加入 NERO 侧开/闭辅助 loss，并增加强制 action 安全投影后重新训练。全过程未连接 CAN、未执行机械臂动作。

## 1. Inputs from Day 1–4

| 项目 | 配置 |
| --- | --- |
| state | 7 joint qpos + 1 gripper width，共 8 维 |
| action | 7 joint target + 1 gripper target，共 8 维 |
| camera | `camera_0`，输入 `[B,1,3,90,160]` |
| alignment | offset 0，`observation[t] -> action[t:]` |
| chunk size | 20 |
| recording frequency | 20 Hz |
| model | 官方 ACT，ResNet-18，hidden 128，2 encoder + 2 decoder |

## 2. Dataset and Split

当前 manifest 有 20 条 valid demonstrations，达到 Day 5 最低数量要求；Day 5 新录制 0 条。旧 `episode_001` 共 1 条 rejected，原因是 qpos 全程冻结而 action 明显运动。

使用 seed 42 按 episode 固定为 80/20 split，同一 episode 不会跨集合：

- train（16）：1、2、3、4、5、6、7、9、11、12、13、14、17、18、19、20；
- val（4）：8、10、15、16。

split 保存于 `data/act/splits/day5_split.json`，checkpoint 内另存一份 `split.json`。训练入口会验证 split 无重叠、完整覆盖全部 valid ID，并拒绝 rejected episode。

数据统计：

| 指标 | min | mean | max |
| --- | ---: | ---: | ---: |
| episode length | 197 | 262.95 | 356 |
| duration (s) | 9.770 | 13.068 | 17.720 |
| actual frequency (Hz) | 20.034 | 20.045 | 20.061 |

结构、数值与连续性满足训练门槛。物体位置覆盖范围和每条任务是否真正成功仍属于人工视觉验收项；数据量刚好达到最低值，并不代表已经覆盖充分。

## 3. Normalization

normalization 只使用 16 条 train episodes；validation 完全复用 train stats。变长 episode 使用时间维 concatenate，不要求 episode 等长。

```text
qpos_mean   = [0.007738, 0.889212, -0.039668, 0.927866, -0.026313, -0.184168, 1.245445, 0.063146]
qpos_std    = [0.232881, 0.105763, 0.083322, 0.101271, 0.093997, 0.010000, 0.078438, 0.033543]
action_mean = [0.007738, 0.889212, -0.039668, 0.927866, -0.026313, -0.184168, 1.245445, 0.050226]
action_std  = [0.232881, 0.105763, 0.083322, 0.101271, 0.093997, 0.010000, 0.078438, 0.047364]
```

## 4. Formal Training Configuration

配置文件为 `configs/act_nero_day5.yaml`：

```text
epochs=300, batch_size=2, chunk_size=20
hidden_dim=128, dim_feedforward=512
nheads=8, encoder/decoder=2/2
kl_weight=10, gripper_bce_weight=1, lr=1e-4, backbone_lr=1e-5
pretrained ResNet-18=true, seed=42, device=RTX 4090 CUDA
```

每个 epoch 从每条训练 episode 随机抽一个 observation；validation 使用每条 val episode 的 5 个固定起点，共 20 个固定 samples。训练日志逐 epoch 写入 `logs/act_day5_training.csv`。

## 5. Training Result

修复后 300 epochs 耗时 136.97 秒，无 NaN/Inf、OOM、DataLoader hang 或非有限 gradient。

| epoch | train total | train L1 | train KL | train grip BCE | val total | val L1 | val grip BCE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 88.5132 | 0.8622 | 8.641 | 1.245 | 66.1569 | 0.7531 | 2.253 |
| 291（best） | 0.2980 | 0.1836 | 0.00280 | 0.0864 | 0.3407 | **0.1524** | 0.1711 |
| 300（last） | 0.3332 | 0.1708 | 0.00271 | 0.1354 | 0.3308 | 0.1556 | 0.1633 |

train 和 val 同时下降，末期有正常小幅波动。按照预定规则选择 val L1 最低的 epoch 291。辅助 BCE 没有修改官方 ACT 网络，只纠正 NERO 夹爪 action 的二值语义。

## 6. Full Validation Offline Evaluation

使用 epoch 291 checkpoint，对 val episodes 8、10、15、16 的全部 1006 个 observation 做 inference prior 预测。比较时只统计 chunk 内存在 GT 的 timestep。

- normalized global MAE：`0.160939`；
- denormalized global MAE：`0.014796`（7 个 rad 通道与 1 个 m 通道的混合平均，仅作总览）。

逐通道真实单位 MAE：

| 通道 | MAE | 单位 |
| --- | ---: | --- |
| J1 | 0.034509 | rad |
| J2 | 0.024857 | rad |
| J3 | 0.006578 | rad |
| J4 | 0.014576 | rad |
| J5 | 0.008653 | rad |
| J6 | 0.000374 | rad |
| J7 | 0.006758 | rad |
| gripper raw continuous | 0.022063 | m |
| gripper binary command | 0.009751 | m |

J1/J2 仍是误差最大的关节。夹爪二值准确率由 85.0% 提升到 89.8%；当前动作的 step 0 准确率为 93.4%，远期最低约 86.2%，说明剩余错误主要集中在开/闭事件的未来时刻。

## 7. Chunk-Step MAE

从 step 0 到 step 19：

```text
[0.011613, 0.011454, 0.011211, 0.011411, 0.012357,
 0.012492, 0.013431, 0.013491, 0.014414, 0.014781,
 0.015213, 0.015568, 0.016183, 0.016359, 0.016797,
 0.017395, 0.017716, 0.018028, 0.018409, 0.018774]
```

短期前 4 步基本稳定，之后随预测距离增加而逐渐变差；step 19 比 step 0 高约 62%。这符合长预测时域更难的预期，也说明 Day 6 应使用 temporal aggregation，而不是直接整段开环执行 20 个预测动作。

## 8. Prediction Safety Analysis

- 非有限预测：0；
- 七个关节超出标称限位：全部为 0；
- 最大预测关节相邻步长：`0.038051 rad`，低于当前离线门槛 `0.08 rad`；
- 七个关节的原始预测越界数：0；
- 辅助分类训练后的原始夹爪值是开/闭判别分数，反归一化范围为 `-0.031617..0.121089 m`，不可直接作为命令；
- 强制安全投影把夹爪判别结果映射为 `0` 或 `0.095 m`，处理后所有通道越界数为 0；
- 投影后夹爪 binary MAE 为 `0.009751 m`，全 chunk 准确率 89.8%，step 0 准确率 93.4%。

`src/act/action_safety.py` 已解决数值边界问题：任何非有限值、关节限位越界或超过 `0.08 rad` 的单步变化都会被拒绝，夹爪只允许输出示教使用的 0/95 mm。原始 checkpoint 输出仍不得绕过此函数直接发送。剩余约 10% 的远期夹爪状态错误需要 Day 6 temporal aggregation、状态迟滞和极短窗口验证，不能靠放宽安全阈值处理。

## 9. Outputs

```text
data/act/splits/day5_split.json
configs/act_nero_day5.yaml
logs/act_day5_training.csv

checkpoints/act_day5/
  policy_best_val.ckpt
  policy_last.ckpt
  dataset_stats.pkl
  config.yaml
  split.json

artifacts/act_day5/
  dataset_stats.json
  fixed_val_samples.json
  training_summary.json
  offline_eval.json
  safety_summary.json
  train_val_loss.png
  train_val_l1.png
  train_val_kl.png
  train_val_gripper_bce.png
  per_joint_mae.png
  chunk_step_mae.png
  sample_0..2_pred_vs_gt.png
```

模型与数据产物由 `.gitignore` 排除，可由配置和训练入口复现。

## 10. Selected Checkpoint

Day 6 候选 checkpoint：

```text
checkpoints/act_day5/policy_best_val.ckpt
epoch=291
selection metric=val L1 0.152449
```

选择的是 best-val 而不是 epoch 300 last checkpoint。该文件当前只是候选，不代表授权真机执行。

## 11. Regression Tests

Day 5 增加了 split 可复现/无泄漏、rejected 排除、固定 timestep dataset、padding-aware error aggregation，以及 action 安全投影测试。全仓库 83 项离线测试通过；`compileall` 与 `git diff --check` 通过。测试不会连接机械臂。

## 12. Day 6 Readiness

正式训练、checkpoint 选择、全 val 离线评估和数值安全投影已经完成。当前仍**不满足直接真机 inference 条件**，阻塞项是：

1. 尚未实现 temporal aggregation、夹爪状态迟滞与逐周期 feedback watchdog；
2. 夹爪 step 0 二值准确率为 93.4%，仍需离线时序聚合验证；
3. 20 条数据只达到最低数量，需要操作者确认物体位置变化和任务成功覆盖充分；
4. 尚未对 Day 6 rollout 初始姿态、停止条件和低速比例做分阶段验收。

Day 6 首先应实现一个完全离线的 rollout adapter：20 Hz inference、temporal aggregation、夹爪 clamp、关节限位、0.08 rad 最大步长和 watchdog。它必须先回放为日志/曲线，通过后才能做单步或极短窗口真机低速测试。
