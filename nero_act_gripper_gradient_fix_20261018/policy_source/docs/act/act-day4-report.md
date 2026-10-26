# ACT Day 4 — NERO 适配与小样本 Overfit 报告

本文记录官方 ACT 到 NERO 单臂、单相机数据的最小适配与离线验收。Day 4 全程未连接或驱动机械臂；生成的 checkpoint 仅用于证明学习链路，不可用于真机控制。

## 1. Day 1–3 输入结论

| 项目 | 当前定义 |
| --- | --- |
| arm qpos | 7 维 `float32`，单位 rad |
| 完整 state | `7 joint qpos + 1 gripper state = 8` |
| action | `7 joint target + 1 gripper target = 8` |
| gripper | 实际/目标开口宽度，单位 m，范围 `0..0.1` |
| camera | `camera_0`，ZED 左目 RGB |
| image tensor | `[B, 1, 3, 90, 160]`，`float32`，范围 `0..1` |
| 采样频率 | 20 Hz |
| alignment | `observation[t] -> action[t:]`，offset `0` |

`data/act/manifest.json` 中 `episode_1`–`episode_20` 共 20 条为 `valid`；旧的 `episode_001` 为 `rejected`，训练加载器会拒绝使用它。

## 2. 官方 ACT 来源与改动边界

模型代码来自[官方 `tonyzhaozh/act` 仓库](https://github.com/tonyzhaozh/act) commit `742c753c0d4a5d87076c8f69e5628c79a8cc5488`，保存在 `third_party/act/`，同时保留 ACT MIT 与 DETR Apache-2.0 许可。没有重写 Transformer、CVAE、位置编码或 ResNet backbone。

仅做了以下兼容改动：

| 官方文件 | 修改 | 原因 |
| --- | --- | --- |
| `detr/main.py` | 使用包内导入、`parse_args([])`、配置设备，并为无关 CLI 项提供默认值 | 从 NERO policy 直接构建模型，支持 CPU，且不读取外层程序参数 |
| `detr/models/detr_vae.py` | `state_dim/action_dim` 参数化，覆盖 ACT 与随附 CNNMLP 的输入/输出层 | 官方 ALOHA 的机器人相关 14D 不适用于 NERO 8D |
| `detr/models/backbone.py` | 直接接收 tensor，移除未 vendoring 的 `util.misc`，预训练权重可配置 | 保留网络结构，同时允许离线、单相机运行 |
| `detr/models/position_encoding.py` | 直接接收 tensor | 配合精简后的 backbone 输入，不依赖 DETR `NestedTensor` |
| `detr/models/transformer.py` | 仅移除调试用 IPython 导入 | 避免引入训练不需要的依赖，模型结构不变 |

逐项搜索后，vendored ACT 运行代码中已无字面量 `14`。原值只代表 ALOHA 双臂 14 维 qpos/action，不是网络结构常量；`num_queries` 独立对应 action chunk 长度，当前为 20，未被误改。

## 3. 新增文件

- `configs/act_nero.yaml`：单一训练/模型配置；
- `src/act/nero_dataset.py`：manifest 筛选、HDF5 读取、归一化、padding；
- `src/act/nero_policy.py`：官方 ACT 的薄封装、L1 与 KL loss；
- `src/act/train_smoke_test.py`：shape smoke test 和 1/3 episode overfit；
- `scripts/act_predict_debug.py`：checkpoint 离线推理与安全边界检查；
- `tests/test_act_nero_dataset.py`、`tests/test_act_nero_policy_shapes.py`：数据与模型回归测试；
- `third_party/act/`：固定版本的官方 ACT DETR/CVAE 必需文件。

## 4. Dataset 与归一化

加载器只读取 manifest 中 `status == "valid"` 的路径。每个样本输出：

```text
image:  [1, 3, 90, 160]
qpos:   [8]
action: [max_episode_length, 8]
is_pad: [max_episode_length]
```

batch 保留相机维，因此单相机输入是 `[B, 1, 3, H, W]`。不同 episode 可有不同长度；batch 前统一 pad，padding 值使用 action mean，使归一化后的 padding 为 0，并由 `is_pad` 从 loss 中排除。

统计量只从本次选择的训练 episode 计算。1 episode 测试的统计量为：

```text
qpos_mean   = [0.095342, 0.975470, -0.135614, 0.815914, 0.021587, -0.186131, 1.213925, 0.068653]
qpos_std    = [0.185938, 0.109056, 0.010000, 0.079943, 0.024557, 0.010000, 0.010000, 0.031764]
action_mean = [0.095342, 0.975470, -0.135614, 0.815914, 0.021587, -0.186131, 1.213925, 0.057004]
action_std  = [0.185938, 0.109056, 0.010000, 0.079943, 0.024557, 0.010000, 0.010000, 0.046472]
```

标准差下限为 `0.01`，避免近似静止关节除以极小值。测试覆盖 variable length、padding、offset、DataLoader batch 以及归一化/反归一化 round trip。

## 5. 模型输入输出与 loss

当前轻量离线配置使用 ResNet-18、128 hidden dim、2 层 encoder、2 层 decoder、20 queries，约 12.58M 参数。

```text
qpos input:       [B, 8]
image input:      [B, 1, 3, 90, 160]
training action:  [B, T, 8]
training output:  [B, 20, 8]
inference output: [B, 20, 8]
```

总损失为 `masked L1 + 10 * KL`。训练时 CVAE encoder 使用 action chunk 得到 latent posterior；推理时没有真值 action，按官方实现使用零 latent prior。因此训练 forward 的 `a_hat` 与独立 inference 输出不应被当成同一结果。

## 6. Smoke Test

```bash
source /home/user/nero/bin/activate
cd /home/user/projects/nero
python -m src.act.train_smoke_test --smoke-only --episode-count 1
python -m src.act.train_smoke_test --smoke-only --episode-count 3
```

实测 1 episode batch shape：

```text
image/qpos/action/is_pad = [1,1,3,90,160] / [1,8] / [1,273,8] / [1,273]
train/inference output   = [1,20,8] / [1,20,8]
```

3 episode batch 的 action/is_pad 为 `[2,356,8] / [2,356]`，证明不同长度可正常组成 batch。两组 forward 均为有限值，无 NaN/Inf。

## 7. Overfit Test 结果

执行命令：

```bash
python -m src.act.train_smoke_test --episode-count 1 --steps 300
python -m src.act.train_smoke_test --episode-count 3 --steps 100
```

| 数据 | steps | 前10步 loss | 后10步 loss | 前10步 L1 | 后10步 L1 | 固定样本推理 MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| episode 1 | 300 | 61.9057 | 2.4672 | 0.7300 | 0.2684 | 0.06353 → 0.04024 |
| episode 1–3 | 100 | 60.7153 | 4.0398 | 0.7617 | 0.5762 | 0.06151 → 0.05269 |

单 episode 的窗口 loss 下降约 96%，窗口 L1 下降约 63%，固定起点反归一化 MAE 下降约 37%，足以证明数据、loss、反向传播、checkpoint 和 inference 链路有效。3 episode 的 100 步只是运行性验证，训练步数不足以代表收敛或泛化。

## 8. Checkpoint、曲线与复现

本地生成：

```text
checkpoints/act_day4_overfit/policy_1ep_last.ckpt
checkpoints/act_day4_overfit/dataset_stats_1ep.pkl
checkpoints/act_day4_overfit/policy_3ep_last.ckpt
checkpoints/act_day4_overfit/dataset_stats_3ep.pkl
artifacts/act_day4/{1ep,3ep}/
```

每组 artifacts 包含 sample image、loss/L1/KL 曲线、prediction-vs-GT 曲线和 `metrics.json`。模型与统计量体积较大且可复现，已经由 `.gitignore` 排除，不进入 Git。

离线查看 checkpoint：

```bash
python scripts/act_predict_debug.py \
  checkpoints/act_day4_overfit/policy_1ep_last.ckpt \
  checkpoints/act_day4_overfit/dataset_stats_1ep.pkl \
  --episode-id 1 --start-ts 0
```

## 9. 反归一化与安全结论

1 episode checkpoint 能输出 `[20,8]` action chunk，反归一化各通道范围为：

```text
min = [-0.084301, 1.020458, -0.136953, 0.742682, 0.014595, -0.186084, 1.213309, 0.095929]
max = [-0.068133, 1.023819, -0.136580, 0.749530, 0.015148, -0.185858, 1.213675, 0.100195]
```

七个关节均在标称限位内，最大相邻关节步长为 `0.0143 rad`；但夹爪预测最大值 `0.100195 m` 略高于 `0.1 m`，离线 preflight 会明确拒绝。这也证明安全检查没有因 overfit 指标好看而被绕过。当前 checkpoint **严禁发送到机械臂**，正式推理层必须先做 action clamp、平滑、temporal aggregation 和独立真机安全门。

## 10. 测试与环境

运行：

```bash
python -m unittest discover -s tests
python -m compileall -q src scripts third_party/act tests
```

最初环境为 NVIDIA driver `535.288.01` 与 PyTorch `2.13.0+cu130`，驱动只能提供 CUDA Driver API 12.2，因此 `torch.cuda.is_available()` 为 false，本次已记录的 overfit 结果使用 CPU 和 4 个线程完成。随后将虚拟环境改为官方 PyTorch `2.5.1+cu121` 与 torchvision `0.20.1+cu121`；RTX 4090 已被正确识别，CUDA 矩阵运算和 ACT GPU forward smoke test 均通过。系统显示驱动未更换，也不需要重启。

为避免首次运行下载权重，Day 4 的 `pretrained_backbone` 仍为 false；这只适合链路验证，不是正式训练推荐配置。

最终结果：全仓库 79 项离线单元测试通过，`compileall` 与 `git diff --check` 通过；测试使用 mock，不连接 CAN 或真实机械臂。

## 11. 已知风险与 Day 5 Readiness

Day 4 的八个问题均已回答：HDF5 可读、shape 正确、机器人相关 14D 已参数化、单相机维保留、loss 正常下降、1 episode 明显 overfit、inference 输出 `[B,20,8]`、反归一化为关节/夹爪物理量且能被安全层审查。

代码与数据管线已经可以进入 Day 5，但模型尚不具备真机部署条件。Day 5 应优先：

1. 按 episode 划分 train/validation，避免同一轨迹帧泄漏；
2. 用全部人工确认成功的 valid demonstrations 正式训练，并启用预训练 backbone；
3. 离线评估未见 episode、夹爪事件和完整 action chunk；
4. 在任何真机策略推理之前实现 clamp、步长限制、temporal aggregation、急停与逐级 dry-run。

另外，Day 3 的 lag=0 只是采集 schema 对齐；真实机械臂响应仍约滞后 150–200 ms。正式闭环前必须用 replay/command-feedback 日志重新决定控制补偿，不能把训练 offset 随意改成 3–4 帧。
