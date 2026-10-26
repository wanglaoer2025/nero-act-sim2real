# NERO 短历史 ACT 对照

已批准的顺序实施已完成数据、接口、封存和A/B各1000次更新。六项物理检查两组均为2/6，`gate=stop`；历史尚未显示完整流程收益，未进入20轮正式训练、完整开发或新16项独立评估，未下发真机动作。完整抓放未通过。

- 当前RGB：320×180；物理状态8维，每帧策略15维；B使用最近5次有效数值观测。A用同一数据和运行器作单帧对照。
- 新数据：45,188个当前视图、118,895帧独立重渲染；37训练组/12验证组，48完整示教与18部分恢复分别记录。
- 原1792点失败已精确复现；命令保存精度修正后，两条真实失败续接回放误差均为0。
- 初始化实际输入核验最大关节差约3e-7 rad、夹爪差约7.5e-9 m。
- 原名义标定仍未取得真实TCP/相机精度资格；保存观测预测仅为离线名义数据检查。

[实施计划](../docs/superpowers/plans/2026-10-19-nero-act-short-history.md) · [设计](../docs/superpowers/specs/2026-10-19-nero-act-short-history-design.md) · [进度账本](evidence/progress.md) · [机器状态](evidence/pipeline_status.json)

## 数据与证据

[18条恢复批次](training_data/history_recoveries_v3/manifest.json)引用v2中16条已核验文件及v3两条新名义续接。v1/v2失败记录保留，拒绝的v2/2016未用于训练。

[五帧数据验收](training_data/history_complete_v1/data_acceptance.json) · [完整阶段覆盖核验](evidence/task5_data_final_audit.json) · [初始化等价](evidence/actual_initialization_equivalence.json)。

## 已实现的离线入口

从本目录使用`.venv/bin/python -m 模块名 --help`查看参数：

| 模块 | 用途 |
| --- | --- |
| `simulation.history_capture` | 冻结策略失败复现及完整控制快照 |
| `training.history_recoveries` | 生成带父谱系的部分恢复，失败批次不放行 |
| `training.history_data` | 从每个实际物理时刻重渲染五帧，不复用旧RGB凑数 |
| `training.history_seal` | 源码/资产/数据/数值16项与共同批次顺序封存 |
| `training.history` | A/B各1000更新；B通过后才允许两组各20轮 |
| `simulation.history_mechanism` | 三个20毫米抬升探针与三个验证剩余流程 |
| `simulation.visual_evaluation`、`simulation.visual_suite` | 原场景、旧16项开发、新16项独立评估 |
| `simulation.history_intervention` | 保存输入的历史遮蔽/顺序诊断，不进入合法观测流或任务验收 |
| `executor.history_shadow` | 保存的名义历史观测只读预测，绝不打开相机或发送动作 |

入口实现不等于训练/任务验收通过。正式训练要求已绑定的pilot放行文件；独立测试要求已绑定的完整开发16/16候选。20毫米机制结果不算全任务成功。

开发16项入口先要求同模型原场景成功；默认读取`results/nominal_A/summary.json`或`results/nominal_B/summary.json`，其他输出路径用`--nominal-summary PATH`指定。直接评估和suite汇总均校验16个唯一原场景ID与检查点哈希。pilot开发还须实际机制决策为`A_development_only`。

## 本轮最终结果

[训练、失败原因与复现命令](evidence/pilot_results.md) · [六项对照](evidence/pilot_comparison.json) · [机制停止决策](evidence/pilot_decision.json) · [最终完整性核验](evidence/final_verification.json)。

A/B均在验证源抬升和释放后退出通过；原失败场景仍出现闭爪不足/夹持削弱，放置保持仍越过3毫米边界。原场景400点未达20毫米目标；稳定夹持续接在A第4点/B第6点被保护停止。模型没有因此获得正式导出或真机资格。

373项测试通过、5项跳过、10个CLI核验通过；封存后的160个源/资产文件、170个数据输入及原214个来源文件均未改变。A/B保存的名义观测预测已完成，不等于当前真实相机shadow。
