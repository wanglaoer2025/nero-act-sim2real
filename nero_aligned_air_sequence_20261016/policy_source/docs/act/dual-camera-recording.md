# ACT 双相机录制与训练指南

## 是否值得使用双相机

双相机适合当前抓取任务：`camera_0`（ZED 左目）观察全局位置，`camera_1`（DICOTA）观察抓取接触区，可减少机械臂遮挡、目标在远景中过小造成的信息损失。它不会自动修复示教不一致、夹爪标签错误或相机移动；两台相机必须在录制和推理期间刚性固定，视角、方向和工作区保持一致。

当前实现是两台相机依次取帧，不是硬件同步。先用下列 dry-run 测量 20 Hz 是否达标；只有出现明显掉帧时才需要增加后台采集线程。

## 1. 固定与检查相机

- ZED 保持当前位置，作为 `camera_0`。
- DICOTA 放在无遮挡的侧前方，作为 `camera_1`；完整看到待抓物、目标、夹爪和两者之间的路径。
- 避免强反光、自动曝光剧烈变化以及示教者身体长期进入画面。
- 两个视角都必须看到任务关键阶段；录制中不得移动相机。

```bash
cd /home/user/projects/nero
source /home/user/nero/bin/activate

python -m src.act.record_episode \
  --dry-run \
  --output-dir data/act_dual_fixed \
  --episode-id 0 \
  --frequency 20 \
  --secondary-camera-device /dev/v4l/by-id/usb-DHZJ-251231-J_DICOTA_4K_01.00.00-video-index0 \
  --secondary-camera-rotation-deg 0
```

预览中左侧应为 `camera_0`，右侧应为 `camera_1`。如果右侧方向错误，把旋转参数改为 `90`、`180` 或 `270`。选定后，所有 episode 和实机推理都必须使用同一角度，并同步修改 `configs/act_nero_dual_day6.yaml` 的 `secondary_camera_rotation_degrees`。

## 2. 录制示教

双相机模型不能直接混用旧的单相机或任意起点 episode；请在 `data/act_dual_fixed` 重新录制一组完整数据。每条示教都要包含：接近、闭合夹爪、搬运、放置、打开夹爪和撤离。物体与目标位置应在安全工作区内做有控制的变化，不要每次完全相同。

```bash
python -m src.act.record_episode \
  --execute \
  --output-dir data/act_dual_fixed \
  --episode-id 0 \
  --position-group 1 \
  --task-name pick_place_dual \
  --frequency 20 \
  --secondary-camera-device /dev/v4l/by-id/usb-DHZJ-251231-J_DICOTA_4K_01.00.00-video-index0 \
  --secondary-camera-rotation-deg 0
```

操作键：`r` 开始；`o/c` 打开/闭合夹爪；空格或 `q` 完成并保存；`x/Esc` 取消；`e` 急停。先录 3 条并逐条检查，再批量录制，避免重复产生错误数据。

```bash
python -m src.act.episode_inspect data/act_dual_fixed/episode_0.hdf5
python -m src.act.validate_episode data/act_dual_fixed/*.hdf5 \
  --manifest data/act_dual_fixed/manifest.json
```

检查输出必须同时包含 `camera_0`、`camera_1`，夹爪事件通常至少有 2 次，并由操作者确认任务实际成功、两路画面无遮挡。建议至少录制 50 条成功示教；20 条只适合先验证链路和过拟合，不足以判断泛化能力。

### 使用 YOLO 自动生成连续示教

黄色杯子到蓝色卡片的重复成功流程可以直接作为教师策略生成 ACT episode。运行前把杯子和蓝卡放进已验证的工作区；输入一次 `AUTO_RECORD` 后，目标稳定锁定、接近、夹紧、搬运、放置、松爪和撤离均自动连续完成，中途不需要键盘控制。

```bash
python -m src.act.record_yolo_episode \
  --episode-id 0 \
  --position-group 1 \
  --weights /home/user/projects/yolo/best.pt \
  --source-offset-mm 0 -17 0 \
  --destination-offset-mm 0 0 0 \
  --transit-speed-percent 30 \
  --precision-speed-percent 5
```

每次成功后重新摆放杯子和蓝卡，并递增 `--episode-id`。`--position-group` 使用 `1`、`2`、`3` 标记三个杯子位置，每个编号必须保留相同数量的成功轨迹。脚本默认使用 `cup`（训练集中的黄色杯子）和 `blue`，新数据保存到 `data/act_dual_fixed`。运动失败、检测超时或录制异常时不会保留该条数据；即便脚本成功，仍需人工确认杯子确实落在蓝卡上，再运行 `episode_inspect` 和批量 `validate_episode`。

每条自动示教都会先通过 `src/arm/prepare.py` 低速移动到固定关节位姿，完成夹爪初始化后才启动录制，所以开头不再包含约 33 帧静止等待。原始 action 保持同周期；训练阶段只把七个关节前视 8 帧，夹爪开合事件不前移。

旧 `data/act_dual` 的前 33 帧可通过训练配置中的 `trim_start_steps: 33` 逻辑排除，原 HDF5 不会被覆盖。但旧轨迹起点不同、三个位置数量也不相等，不能作为这次最终模型的数据；应在 `data/act_dual_fixed` 重新采集三个位置等量轨迹。

### 固定起点重采与训练

建议每个位置先录 10 条成功轨迹，共 30 条；例如 episode 0–9 使用组 1，10–19 使用组 2，20–29 使用组 3。完成后执行：

```bash
python -m src.act.validate_episode data/act_dual_fixed/*.hdf5 \
  --manifest data/act_dual_fixed/manifest.json

python -m src.act.dataset_stats \
  --config configs/act_nero_dual_xyz_fixed_day5.yaml

python -m src.act.train \
  --config configs/act_nero_dual_xyz_fixed_day5.yaml \
  --device cuda

python -m src.act.eval_offline \
  --checkpoint checkpoints/act_dual_xyz_fixed_day5/policy_best_val.ckpt \
  --config configs/act_nero_dual_xyz_fixed_day5.yaml \
  --split val
```

训练会拒绝起始关节偏离固定姿态或三个位置轨迹数不相等的数据。每个 epoch 按 stride 4 使用所有完整时间窗，不再每条视频只抽一帧。

### 合并旧组与修复后的新组训练

合并训练使用 `data/act_dual` 的 episode 0–19 与 `data/act_dual_fixed` 的 episode
27–40，共 34 条；已确认包含错误放置高度的 episode 20–26 不进入 manifest。合并索引
不复制 HDF5。配置会对旧组裁掉开头 33 帧、新组不裁剪，并忽略两组原始 action 的
2/0 帧前视差异，统一从真实 qpos 生成 8 帧前视关节标签：

```bash
python -m src.act.dataset_stats \
  --config configs/act_nero_dual_xyz_merged_day5.yaml --force-split
python -m src.act.train \
  --config configs/act_nero_dual_xyz_merged_day5.yaml --device cuda
python -m src.act.eval_offline \
  --checkpoint checkpoints/act_dual_xyz_merged_day5/policy_best_val.ckpt \
  --config configs/act_nero_dual_xyz_merged_day5.yaml --split val --device cuda
```

两组的起点和运行速度仍不同；合并模型完成离线评估不代表实机任务成功，Day 6 必须重新
从 dry-run 和 10 个低速周期开始验收。

合并模型的实机配置为 `configs/act_nero_dual_xyz_merged_day6.yaml`。依次完成 50 周期
dry-run、单目标、10/50/150 周期无融合执行，确认策略持续推进任务后才启用 temporal
ensemble；每次扩展前检查上一次 `logs/act_dual_xyz_merged_day6` 的 summary。

新模型第一次实机验证默认关闭 temporal ensemble：

```bash
python -m src.act.run_act_real \
  --config configs/act_nero_dual_xyz_fixed_day6.yaml \
  --execute --max-steps 10 --speed-scale 0.5

python -m src.act.run_act_real \
  --config configs/act_nero_dual_xyz_fixed_day6.yaml \
  --execute --max-steps 50 --speed-scale 0.5 --extended
```

只有无融合时能持续接近、闭爪并推进任务，才在同一命令上增加 `--temporal-ensemble`。

## 3. 训练与离线验收

```bash
python -m src.act.dataset_stats --config configs/act_nero_dual_day5.yaml

python -m src.act.train_smoke_test \
  --config configs/act_nero_dual_day5.yaml \
  --smoke-only \
  --episode-count 1

python -m src.act.train \
  --config configs/act_nero_dual_day5.yaml \
  --device cuda

python -m src.act.eval_offline \
  --checkpoint checkpoints/act_dual_day5/policy_best_val.ckpt \
  --config configs/act_nero_dual_day5.yaml \
  --split val
```

双相机配置使用 `180×320` 输入，避免原 `90×160` 把小目标压缩得过小。显存不足时只把 `batch_size` 从 2 改为 1，不要改变相机顺序或图像尺寸后继续复用旧 checkpoint。

## 4. 实机推理

只有离线评估生成 `artifacts/act_dual_day5/safety_summary.json` 后，才运行双相机实机链路：

```bash
python -m src.act.run_act_real \
  --config configs/act_nero_dual_day6.yaml \
  --dry-run \
  --max-steps 50
```

确认日志、初始姿态和画面正常后，再依次进行 single-action、10 step 和扩展测试；不要直接跳到完整抓取。

## 2026-09-10 训练记录

- 数据：`episode_0`–`episode_29`，30/30 valid，均为 20 Hz、双相机、2 次夹爪事件。
- 固定划分：24 条 train，6 条 val（7、16、22、24、25、29）。
- 训练：300 epochs；best epoch 226；best val L1 0.10447。
- 离线评估：关节与夹爪全局 MAE 0.01965；夹爪二值准确率 98.58%；NaN/Inf 与关节越界均为 0。
- 输出：`checkpoints/act_dual_day5/policy_best_val.ckpt`、`artifacts/act_dual_day5/safety_summary.json`。
- 实机配置已通过 checkpoint 兼容性加载检查：双相机、`180×320`、chunk 20。

## 2026-09-11：加入显式目标坐标

旧模型只输入 7 个关节、夹爪状态和 RGB，容易在杯子位置变化时输出平均轨迹。新的独立配置把每条示教已保存的 `source_base_xyz_mm` 与 `destination_base_xyz_mm` 转为米，并追加到状态中：

```text
[J1..J7, gripper, source_x, source_y, source_z, destination_x, destination_y, destination_z]
```

动作仍为原来的 8 维关节与夹爪目标，安全过滤器不变；旧的 `act_dual_day5` checkpoint 不会被覆盖。训练和离线评估命令：

```bash
python -m src.act.train_smoke_test \
  --config configs/act_nero_dual_xyz_day5.yaml \
  --smoke-only --episode-count 1

python -m src.act.train \
  --config configs/act_nero_dual_xyz_day5.yaml \
  --device cuda

python -m src.act.eval_offline \
  --checkpoint checkpoints/act_dual_xyz_day5/policy_best_val.ckpt \
  --config configs/act_nero_dual_xyz_day5.yaml \
  --split val
```

实机入口会先复用 `src.demo.yolo_pick_place` 的稳定检测、深度过滤和 camera→base 变换，锁定杯子与蓝卡的 6 个坐标；释放 ZED 后再打开双相机 ACT 链路：

```bash
python -m src.act.run_act_real \
  --config configs/act_nero_dual_xyz_day6.yaml \
  --dry-run --max-steps 50
```

如果只需排查 ACT、暂时不运行 YOLO，可同时手工提供两组 base 坐标（单位 mm）：

```bash
python -m src.act.run_act_real \
  --config configs/act_nero_dual_xyz_day6.yaml \
  --dry-run --max-steps 50 \
  --source-base-xyz-mm -520 50 70 \
  --destination-base-xyz-mm -550 -150 0
```

手工坐标仅用于诊断；实机任务应使用自动 YOLO 定位。改变检测模型或修正量时，同时更新 `task_coordinates`，使其与自动示教时使用的参数一致。

本次 30 条数据训练结果：best epoch 281，validation L1 0.10156；真实单位全通道 MAE 0.01620，夹爪二值准确率 99.21%，没有 NaN/Inf 或关节限位越界。验证集中有 16/4553 个 observation 的预测 chunk 相邻步长超过 0.08 rad；实机安全过滤器仍会逐步限幅，不能因此放宽安全参数。离线 A/B 中 temporal decay `0.1` 的 MAE 最低，因此新 Day6 配置采用该值。

## 2026-09-15：连续位置新数据重训

- 数据实际为 `data/act_dual_fixed/episode_41`–`episode_61`，共 21 条，不混入旧相机域或错误放置高度的数据。
- 21/21 条均通过结构、20 Hz、双相机和夹爪事件检查；闭爪 TCP 到检测抓取点平均误差 2.07 mm，放置 TCP 到期望点平均误差 1.68 mm。
- `position_group` 被统一误标为 1，因此本轮不按该字段分层；固定划分为 17 条训练、4 条验证（45、49、54、58），验证坐标覆盖低/中/高区域。
- 训练 80 epochs；部署采用第 68 轮的关节 L1 最优权重。1,665 个验证窗口的真实单位全通道 MAE 为 0.03291，夹爪二值准确率为 95.66%，NaN/Inf 和七个关节硬限位越界均为 0。
- 固定 Home qpos 和任务坐标、只换三组真实双相机图像时，未来 20 步关节预测两两 MAE 为 0.070–0.134 rad，确认模型输入帧不同且策略会响应视觉变化。

复现实验与离线验收：

```bash
python -m src.act.dataset_stats \
  --config configs/act_nero_dual_xyz_latest_day5.yaml --force-split
python -m src.act.train \
  --config configs/act_nero_dual_xyz_latest_day5.yaml --device cuda
python -m src.act.eval_offline \
  --checkpoint checkpoints/act_dual_xyz_latest_day5/policy_best_l1.ckpt \
  --config configs/act_nero_dual_xyz_latest_day5.yaml --split val --device cuda
```

实机测试使用 `configs/act_nero_dual_xyz_latest_day6.yaml`：默认开启 temporal ensemble（decay `0.1`），并严格按 10、100、完整轨迹逐级增加周期。该配置同时启用 25 mm 抓取门禁：只有当估算 TCP 进入检测抓取点的球形容差时，OPEN→CLOSE 才会下发；进入搬运阶段后不会因离开抓取点而张开。CSV 的 `grasp_gate_blocked` / `grasp_tcp_distance_mm` 和 summary 的 `grasp_gate_blocks` / `grasp_gate_min_attempt_distance_mm` 用于验证门禁。每一级先检查 CSV 和 summary，不能直接开始完整高速抓取。

### 2026-09-15：抓取前窗口加权复训

针对 `act_day6_execute_temporal_20260915_193323.csv` 在杯子附近来回修正并触发进展
watchdog 的问题，复训仍只使用 episode 41–61，原 HDF5 不改写。配置
`configs/act_nero_dual_xyz_grasp_focus_day5.yaml` 在训练和验证中将松爪后的回 Home 尾段裁到
20 帧，并把首次闭爪前 100 帧的完整窗口重复为 3 倍；训练窗口由 1,483 增至 2,339。
80 epochs 完整训练的关节 L1 最优权重位于第 70 轮，validation L1 为 0.18606。

1,440 个裁剪后验证窗口中，真实单位全通道 MAE 为 0.03430，夹爪二值准确率 95.87%；
NaN/Inf、七关节硬限位越界和 safety 后越界均为 0。与旧模型相比，全通道 MAE 略高
（0.03291 → 0.03430），因此不能只看 validation 宣称问题解决。

使用失败 rollout 在 step 225/250/275/300/325 保存的真实模型输入做 A/B：新模型的预测关节
偏移连续变化均值从 0.227 降到 0.130 rad，最大值从 0.553 降到 0.240 rad，方向翻转由
6 次降到 2 次。固定 step 275 的 qpos、只换这五组双相机 tensor 时，首动作关节两两 MAE
均值从 0.0513 降到 0.0261 rad。复放命令不会连接相机或机械臂：

```bash
python -m src.act.eval_saved_rollout \
  --config configs/act_nero_dual_xyz_grasp_focus_day6.yaml \
  --csv logs/act_dual_xyz_latest_day6/act_day6_execute_temporal_20260915_193323.csv \
  --model-input-dir logs/act_dual_xyz_latest_day6/act_day6_execute_temporal_20260915_193323_model_inputs \
  --source-base-xyz-mm -574.547708 -16.127564 73.132202 \
  --destination-base-xyz-mm -599.514127 -196.374878 -9.015731 \
  --from-step 225 --to-step 325 --fixed-qpos-step 275 \
  --device cuda \
  --output artifacts/act_dual_xyz_grasp_focus_day5/candidate_saved_rollout_193323.json
```

该结果只通过离线稳定性门槛，不代表已经完成真机抓放。第一次真机只运行 100 steps、保持
`speed-scale 0.25`、temporal ensemble、25 mm 闭爪门禁和进展 watchdog；必须人工准备实体
急停，并在继续完整轨迹前检查新 CSV。

后续 `act_day6_execute_temporal_20260915_203716.csv` 表明模型会在 TCP 距抓取点仍有
86.42 mm 时请求闭爪，并在门禁保持张爪后继续输出抬升动作。当前 Day6 配置因此启用
`abort_on_grasp_gate_block: true`：第一次容差外闭爪请求会被记录但不会发送该周期的机械臂
或夹爪命令，程序随后按普通失败保持当前位置。该保护只完成自动测试，不能当成模型抓取
时序已经修复；它只是避免错误策略继续进入抬升阶段。

### 2026-09-16：compatible38 rollout 安全停止线

最新 910-step 实机日志显示策略在接近训练闭爪关节状态时仍未闭爪，并把 TCP 带到抓取点
下方约 53 mm。38 条有效示教在抓取点 150 mm 内、闭爪前的最低 Z 偏差为 `-19.67 mm`，
所以 `configs/act_nero_dual_xyz_compatible38_day6.yaml` 使用 `-25 mm` 作为高度停止线。
离线分析与关键 step 对照见
`artifacts/act_dual_xyz_compatible38_day5/rollout_diagnostics/execute_20260916_101344.json`。

下一次只验证安全停止，不把停止当成抓取成功：

```bash
python -m src.act.run_act_real \
  --config configs/act_nero_dual_xyz_compatible38_day6.yaml \
  --execute --extended --max-steps 900 --speed-scale 0.25 \
  --temporal-ensemble --device cuda --save-model-input-every 50
```

如果触发 `TCP 抓取高度过低`，先保存 CSV、summary 和模型输入，不继续增加步数。后续数据仍用
`src.act.record_yolo_episode` 的固定 Home、双相机和 20 Hz 协议采集，并保持当前侧向
`对准 → 接近 → 闭爪 → 抬升` 路径；训练时继续用 `grasp_focus_before_close_steps` 加权最后
100 帧。现有录制器没有“从杯旁开始”的恢复模式，不能通过手动把机械臂摆到杯旁冒充固定
Home 数据。

### 2026-09-17：固定蓝卡候选模型

`data/act_dual_fixed_destination` 共 17 条。只有 episode 0–8 的目标点接近同一区域；episode
9–16 的目标点相对参考位置偏离 17.7–201.3 mm，因此没有混入固定蓝卡候选。严格数据集为
7 条训练（0、1、3、4、5、6、8）和 2 条验证（2、7）。旧 compatible38 数据的轨迹节奏
与新数据不同，不直接拼接；它只通过 checkpoint 初始化和原统计量参与微调。

微调在第 20 轮取得最佳 validation L1 `0.09002`，连续 10 轮无改善后停止。完整验证集
2,286 个窗口的真实单位 MAE 为 `0.01517`，夹爪二值准确率为 `99.36%`，无 NaN/Inf、
关节越界或安全后越界。相同划分从头训练 80 轮的对照模型真实单位 MAE 为 `0.03454`，
因此不部署。部署配置为 `configs/act_nero_dual_fixed_destination_day6.yaml`，指向微调模型的
`policy_best_l1.ckpt`。

离线闭爪转场在 train/val 的 `close_idx-10` 到闭爪后均为 100% close 检出；但 9 条中有
6 条在闭爪前出现超过 15 mm 的 Z 向下探后回升，目标点 Y 总跨度为 17.6 mm。旧失败 rollout
的保存输入 A/B 中，新模型相邻预测变化均值从 `0.0993` 降至 `0.0764 rad`、方向翻转从 14
降至 12，但旧 rollout 状态仍未触发首动作闭爪。因此这版只允许按 README 的 dry-run 和
10-step 无融合测试逐级验收，不能直接运行完整 rollout，也不能把离线 close 准确率当作真机
抓取成功。
