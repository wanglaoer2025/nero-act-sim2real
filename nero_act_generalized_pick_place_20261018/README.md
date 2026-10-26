# NERO 在线 ACT 位置泛化新训练

2026年10月18日，用户批准位置泛化方案并要求开始新训练。本目录隔离保存新代码、仿真数据、训练、冻结评估和失败记录；原工程、权重、标定、真实记录及已终态授权不修改。

## 本轮训练内容

- 完整物理示教48条，36个训练场景、12个验证场景，来源分组及文件编号独立核验。
- 源杯XY各±30mm，蓝卡固定；高位TCP起点XYZ各±20mm、姿态各±3°，同肘型IK、关节限位及初始接触检查。
- 夹爪初始90～95mm，低力物理接触搬运，无物体绑定；0～100ms伺服延迟、颜色和光照变化。
- 每条通过完整任务、原动作合同和精确重放才纳入。拒绝候选保留，不计合格条数。
- 策略输入为单RGB、8维实测状态及上一条已提交夹爪目标差，共9维编码；输出20×8绝对目标。没有加入杯子真值/阶段标签/YOLO坐标作为策略输入。
- 使用run13作为训练初始化，保留其旧放置接触失败状态。新统计只取36个训练场景，输入投影及输出头作经过物理预测一致性测试的仿射换算。
- 先1轮CUDA联调，正式run14独立从原初始化开始20轮，以验证集选checkpoint；联调权重不用于正式初始化或模型选择。
- 正式训练成功且输入/源码完整性验证通过后才可导出，随后进行9杯位、4起点、3个历史现场匹配场景的16项在线闭环仿真。

## 真实数据的处理

已有完整真机教师记录虽然实物抓放验收成功，但旧HDF标签为零超前的实测关节位置，七轴action与同帧观测完全相同；夹爪为0/95mm事件目标，非连续低层提交目标。直接混入会教策略保持不动并引入标签跳变，因此本轮明确排除其动作监督，只留真实RGB和反馈作诊断。没有宣称已完成真实数据微调。审计见 [real_recording_audit.json](evidence/real_recording_audit.json)。

## 运行状态和产物

实时阶段以 [training_pipeline_status.json](evidence/training_pipeline_status.json) 为准；后台进程身份见 [training_pipeline_process.json](evidence/training_pipeline_process.json)。流水线等待48条全部验收后才启动优化，任一步失败停止，不自动重试。

| 内容 | 文件 |
| --- | --- |
| 已接受数据与失败候选 | [manifest.json](training_data/generalized_v1/manifest.json) |
| 分组 | [split.json](training_data/generalized_v1/split.json) |
| 仿真采集日志 | [collect_generalized_v1.log](evidence/collect_generalized_v1.log) |
| CUDA联调 | `evidence/training_smoke.log`，产生后可读 |
| 正式优化日志 | `evidence/run14_training.log`，进入优化后产生 |
| 逐轮指标 | `training/run14_position_generalization/training.csv` |
| 训练完整性终态 | `training/run14_position_generalization/artifacts/training_integrity.json` |
| 冻结模型 | `model/run14_position_generalization/` |
| 16项闭环结果 | `evaluation/run14_generalization16/summary.json` |
| 顺序执行与裁定 | [progress.md](evidence/progress.md) |

## 基线与验证边界

旧run13/v7在新目录复现了放置非预期接触失败；旧run11/v6完整名义仿真复跑成功，两者协议及状态分开记录。不能把旧run11成功当作新9维模型成功。

独立代码复核发现的三项重要问题均进入作者一次修正：采集中断恢复、filename/manifest/HDF编号一致性，以及优化前后输入/源码封存和导出终态约束。对应五项测试先失败后通过。首条新示教2772帧、278个前缀窗口，精确重放误差0；另显式通过13项教师测试，包含两项完整物理/局部退出集成测试。

数据、优化、完整仿真、真实shadow和真实抓放分别记录。训练与测试全程没有连接或控制机械臂，新模型硬件许可保持false；既有ACT_DEMO/ACT_REPLAY授权不用于本目录。

后续仍需新的真实输入核对与现场授权，才能检验真实完整抓放。历史现场场景测试使用保存的图像定位/关节记录，并假设先准备张开夹爪，不能作为当前硬件就绪证据。
