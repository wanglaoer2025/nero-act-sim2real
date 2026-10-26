# 短历史ACT：实施、1000步对照及停止结论

2026-10-20（北京时间）。用户批准的顺序实施已执行到机制门槛：A、B各实际1000次优化，六项检查均为2/6。`gate=stop`，未开始两组20轮正式训练，未运行完整开发或新独立16项，未下发真机动作。本轮没有得到完整抓放合格模型，也没有证明增加历史解决了原闭爪/抬升问题。

## 完成内容与证据

- 48条完整示教＋18条通过物理续接验收的局部恢复；37训练源组/12验证源组。共45,188个当前视图、118,895帧重新渲染图像。A/B共用数据、统计、样本权重、批次ID、初始化和1000次更新预算。
- A输入为当前RGB＋当前15维状态；B增加最近5次有效15维数值观测的GRU128条件，当前RGB只编码一次。共同运行器保留真实曝光、逐帧因果命令收据、前缀完成屏障、确认后的五帧重新收集和时效保护。
- 原1792点失败的预测/提交已精确复现。原恢复批次v1缺少离桌变换追踪，予以保留并拒绝；v2两条名义续接受极小命令保存精度损失影响。只为新局部记录保留Float64命令，v3引用16条已验证旧文件并重建两条名义续接，回放误差均为0。
- 实际初始化输入最大输出变化约3e-7 rad、7.5e-9 m；未改变共同起始策略。B首步投影梯度0.012075、GRU梯度0；第二步GRU梯度0.00023265，满足零投影初始化后的梯度预期。CVAE后验、视觉条件层和ResNet layer4也有梯度，冻结的其余骨干无梯度。
- 独立审查的4个Important问题已先RED后GREEN：延迟到达的执行中曝光被误收、RGB缓存覆盖导致错误输入绑定、直接开发汇总不兼容候选入口、开发入口缺少前序门槛。修正后373项通过/5项跳过，10个CLI帮助入口通过。
- 优化前封存160个源码/资产文件和170个数据/证明输入；最终核验这些文件、两个模型完整性证明及原214个来源文件均未变。见[最终核验](final_verification.json)、[源码封存](pre_optimization_freeze.json)、[训练核验](paired_pilot_training_verified.json)。

## 实际训练

| 项目 | A：单帧 | B：五帧历史 |
| --- | --- | --- |
| 实际更新 | 1000 | 1000 |
| 正式轮数 | 0 | 0 |
| 固定检查点 | `training/pilot_A/checkpoints/policy_last.ckpt` | `training/pilot_B/checkpoints/policy_last.ckpt` |
| 零潜变量验证损失 | 0.0090638207 | 0.0091542739 |
| 优化及验证阶段耗时 | 43.29秒 | 45.95秒 |
| 机制及剩余流程检查 | 2/6 | 2/6 |

耗时不含数据构建、启动校验、证明写入和物理评估。损失值不等于任务成功率。两个训练都经过完整性校验，并使用完全相同的1000行批次计划；本轮未选择其他轮次或增加更新数。

## 六项物理检查

| 起始状态与检查 | A | B |
| --- | --- | --- |
| 原失败场景预闭爪，400个新动作点内抬升20毫米 | 失败：400点，未抬升 | 失败：400点，未抬升 |
| 原失败场景稳定夹持，400点内抬升20毫米 | 失败：4点，预测在非放置区域张爪 | 失败：6点，同一保护 |
| 验证源预闭爪，完成剩余抓放 | 失败：2876个新点耗尽原3900点总预算 | 失败：同样耗尽预算 |
| 验证源稳定夹持，抬升20毫米 | 通过：46点 | 通过：46点 |
| 验证源落桌待释放，完成剩余流程 | 失败：19点，离开3毫米放置保持范围 | 失败：17点，同一保护 |
| 验证源释放后，退出并稳定放置 | 通过：336点 | 通过：330点 |

两个通过项目是部分恢复检查，不是从原始起点完成抓取→搬运→放置的任务验收。三个抬升探针与三个剩余流程必须全部通过才放行；两组均未达到。原场景预闭爪“未抬升”也不能误报为控制器报错：它是400点机制预算内未达目标。逐项结果见[对照表](pilot_comparison.json)、[A结果](../results/mechanism_A/summary.json)、[B结果](../results/mechanism_B/summary.json)。

## 失败证据与解释边界

在保存的名义预闭爪样本中，实际开口95毫米，教师首两点继续闭合0.2/0.4毫米，A只预测0.022/0.197毫米，B约0/0.126毫米；实际闭环仍停留在未抬升状态。在名义稳定夹持样本中，教师首两点夹持差约3.692/3.892毫米，A约3.101/3.430毫米、B约3.132/3.452毫米，已接近3毫米下界。实际续接时两组进一步削弱夹持，触发非放置区域禁止张爪保护，保护未被放宽。见[预测与标签核对](pilot_prediction_label_probe.json)。这些证据证明此预算下策略未学稳接触过渡；尚不能把根因唯一归于历史长度、CVAE或某一个损失项。

在名义和验证源两个保持样本上，历史遮蔽对B输出有非零影响；翻转过去四帧的输出变化仅约1e-7 rad量级。保持阶段的相邻帧接近静止，不能据此断言网络完全忽略顺序，也不能据此宣称历史有效。干预只在离线张量分支执行，保留当前观测，不送入合法运行流或资格门槛。见[名义干预](../results/history_intervention_nominal/report.json)、[验证源干预](../results/history_intervention_heldout/report.json)。

[机制遥测](pilot_mechanism_telemetry.json)保存首两点原始预测/实际ACT提交、逐次夹持差、杯高、双侧接触时长、停止原因、仿真曝光到消费时延及墙钟推理时延，并绑定原始轨迹哈希。相机实际采集墙钟时延未测，字段明确为空；这些时延不能作为真实端到端时序验收。

## 后续边界

本计划的条件停止已触发。建议下一轮先针对真实失败状态逐点核对“教师命令—当前实际宽度—首1/2点输出—接触保护”的目标及误差，做小样本接触过渡拟合与闭环诊断，再决定是否调整接触监督或扩大训练；本轮不自行修改架构、增加轮数或更换检查点。新独立16项未被消耗。

A/B都完成保存的名义历史观测只读预测，[A](../results/saved_shadow_A/report.json)、[B](../results/saved_shadow_B/report.json)均无动作/夹爪发送。这不是当前真实相机shadow。旧名义TCP/相机版本仍未取得实测精度资格，真实语义前端与端到端时序、完整真机抓放均未验收。

## 复现命令

在`/home/user/wq/nero_act_history_feedback_20261019`下，设置`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nero_history_mpl`。训练和物理评估需宿主CUDA，评估另设`MUJOCO_GL=egl`；不需要机器人接口。

```bash
.venv/bin/python -m training.history --data-dir training_data/history_complete_v1 --initializer /home/user/wq/nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt --config config/history_experiment.json --variant A --stage pilot --output NEW_PILOT_A
.venv/bin/python -m training.history --data-dir training_data/history_complete_v1 --initializer /home/user/wq/nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt --config config/history_experiment.json --variant B --stage pilot --output NEW_PILOT_B
.venv/bin/python -m simulation.history_mechanism --training-run training/pilot_A --captures training_data/history_recoveries_v3 --output NEW_MECHANISM_A
.venv/bin/python -m simulation.history_mechanism --training-run training/pilot_B --captures training_data/history_recoveries_v3 --output NEW_MECHANISM_B
.venv/bin/python -m simulation.history_intervention --training-run training/pilot_B --capture evidence/saved_history_nominal_holding --output NEW_INTERVENTION
.venv/bin/python -m executor.history_shadow --model training/pilot_B --capture evidence/saved_history_nominal_holding --output NEW_SAVED_SHADOW
PYTHONPATH=. .venv/bin/python evidence/finalize_history_run.py
```

前两条仅说明完整复现参数，本轮已经执行过，未自动再训练；所有输出目录必须不存在。最终核验脚本只重读当前固定路径的工件、核验哈希并重建报告，不优化、不连接硬件。冻结源出现新缺陷时必须建立新隔离实验版本，不能改写当前封存源码与旧证明。
