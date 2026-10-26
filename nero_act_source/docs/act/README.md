# ACT 文档索引

本文件是 ACT 子系统的唯一入口。`act-day*-nero-codex-prompt.md` 是开发任务书，可能包含尚未实现或已经被后续实验推翻的建议；实际行为以 `src/act/`、当前 YAML 和 `act-day*-report.md` 为准。

## 离线源码补充包（2026-10-13）

导出的源码已将 `pyAgxArm` 导入延迟到 `create_nero_arm()` 调用时。仅加载
ACT 数据、计算归一化统计、训练或查看训练/推理入口帮助不需要机械臂 SDK，
仍需要相应的 NumPy、HDF5、PyTorch、torchvision、YAML 等软件依赖。
真实机械臂创建仍需 `pyAgxArm`，真实相机采集仍需 ZED SDK/`pyzed`。

从补充包根目录执行 `python -m unittest discover -s tests` 可复测。
`test_act_offline_imports.py` 在隔离子进程内禁止硬件 SDK 导入，并验证 HDF5
写入、八维 state/action 数据加载和训练模块导入。本次源电脑共通过 23 项测试；
仿真电脑必须单独执行，不将本机记录视为远端验证。

`src.act.run_act_real --dry-run` 仍会在确认后连接并使能机械臂，不能作为离线验证。
单相机八维配置使用 `act_nero_day5.yaml`/`act_nero_day6.yaml` 作为接口参考；
示范数据、归一化统计和训练权重需要针对仿真任务重新生成。

## 阅读顺序

| 阶段 | 实施报告 | 解决的问题 |
| --- | --- | --- |
| Day 1 | [act-day1-report.md](act-day1-report.md) | NERO 七关节/夹爪映射与 5/10/20 Hz 流式控制 |
| Day 2 | [act-day2-report.md](act-day2-report.md) | Leader 示教、双相机帧和 HDF5 schema |
| Day 3 | [act-day3-report.md](act-day3-report.md) | episode 验收、可视化、起点移动和安全 replay |
| Day 4 | [act-day4-report.md](act-day4-report.md) | 官方 ACT 最小适配、shape smoke test 和 overfit |
| Day 5 | [act-day5-report.md](act-day5-report.md) | 正式训练、split、归一化和离线评估 |
| Day 6 | [act-day6-report.md](act-day6-report.md) | 真实相机/反馈闭环、安全层和运行日志 |
| 双相机固定起点流程 | [dual-camera-recording.md](dual-camera-recording.md) | 当前推荐的数据采集、训练和实机验证步骤 |
| Day 7 | [act-day7-nero-codex-prompt.md](act-day7-nero-codex-prompt.md) | 尚未完成；正式 benchmark 的需求草案 |

要理解某个阶段为什么这样实现，再读取同编号的 `act-dayN-nero-codex-prompt.md`。不要直接照任务书中的命令执行实机。

## 当前结论（2026-09-11）

- 20 Hz 相机、推理、关节发送和反馈链路已通过；推理延迟不是策略停滞的核心原因。
- 旧单相机/双相机模型可复现运动，但完整自主抓取未通过任务验收。
- `act_dual_xyz_lead10` 模型能向杯子靠近，随后落入局部小幅摆动且不闭爪。离线 `val_l1` 或运行 `status=passed` 不能代表任务成功。
- 根因包括 episode 开头静止标签冲突、训练起点组合不一致、每轮抽样覆盖不足、三个物体位置不均衡，以及 temporal ensemble 对弱动作的抵消。
- 当前推荐路线是 `act_dual_xyz_fixed`：固定预备姿态、移除/裁剪静止开头、关节前视 8 帧而夹爪保持原时序、遍历全部有效时间窗、三位置等量。
- 新 fixed 模型必须先关闭 temporal ensemble 验证持续推进和夹爪事件；通过后才开启融合。完整根因证据见 [故障排查 11.12](../robot-troubleshooting.md#1112-关节前视对齐后仍在目标附近微摆)。

## 目录对应关系

| 类型 | 位置 | 是否提交 |
| --- | --- | --- |
| 实现 | `src/act/` | 是 |
| 实验配置 | `configs/act_*.yaml` | 是 |
| 自动测试 | `tests/test_act_*.py` | 是 |
| episode 与 manifest | `data/<experiment>/` | 否 |
| checkpoint/statistics | `checkpoints/<experiment>/` | 否 |
| 图表和离线评估 | `artifacts/<experiment>/` | 否 |
| 实机 CSV/summary | `logs/<experiment>/` | 否 |
| 官方 ACT 固定代码 | `third_party/act/` | 是，避免无关修改 |

同一实验的 `<experiment>` 名必须在 data、checkpoint、artifact、log 和训练/推理 YAML 中一致。不要把旧数据统计量或 checkpoint 接到新 action shape、相机列表或对齐配置上。

## 开始下一轮前

1. 先按 `dual-camera-recording.md` 检查三组位置数量相同、固定起点一致、没有录入静止前缀，且每条轨迹包含接近、闭爪、搬运、放置、开爪和撤离。
2. 运行 `validate_episode` 和配置对应的 `dataset_stats`，确认 rejected episode 不进入 split。
3. 训练后先做完整 validation 离线评估，不直接连接机械臂。
4. 实机按 dry-run、单动作、10/50/150 steps、完整 rollout 逐级增加，并同时观察业务阶段，不只看 summary 的 `passed`。
5. 只有完整抓取在固定 benchmark 上重复成功后，才进入 Day 7。

重训前使用 `src.act.grasp_transition_audit` 审查所有 episode 的采集协议、闭爪前
task-relative TCP、双相机关键帧和夹爪时序，并对当前 checkpoint 分别统计 train/val 的
close 检出与时间误差。它只做离线 FK 和模型推理，不连接硬件；详细命令见项目 README。
