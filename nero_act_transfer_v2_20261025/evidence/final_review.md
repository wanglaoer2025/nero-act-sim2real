# ACT v2 整体代码复核

新上下文审查者：`transfer_v2_whole_review`，只读审查，无优化或硬件操作。原结论为“With fixes”，没有Critical，七项Important，一项Minor。实现者按一次修正流程添加回归，原七项测试全部观察到失败后修正通过；附加独立位置排除区测试也先失败再通过。

| 发现 | 修正及回归 |
|---|---|
| 仿真未回执动作仍能准入，初始previous只有无来源标量 | 仿真和真实均要求逐行布尔ACK；初始上下文须有较早的明确提交来源，或可核对的仿真t0伺服初始化。`test_simulation_nacks_and_unproven_initial_context_are_rejected` |
| 分组只核对名字数量，不核实际位置 | 精确组ID、HDF分组属性、base源位加协议偏移同时核对；重复路径规范化。`test_spatial_admission_checks_actual_scene_and_hdf_metadata` |
| 第二个夹爪点没有相对同一previous检查 | 每个查询的两个可用点同时检查，报告查询帧和点号。`test_second_prefix_point_uses_same_previous_and_reports_location` |
| 开发验收只覆盖3中心和4邻近 | 保留7项准备门槛，加入全部30条开发条件及历史7场景，每候选44条完整物理开发运行。`test_development_suite_keeps_all_thirty_and_seven_historical_cases` |
| 选模误差包含未执行夹爪18点，可能丢掉可适配候选 | 部署误差只计arm20/gripper2；保留适配资格，再按完整成功、阶段事件、误差选择。`test_validation_ignores_unexecuted_gripper_and_selection_keeps_eligibility` |
| 部分恢复/教师介入可被完整成功判据接收 | 运行器增加明确fresh-start来源，验收要求teacher_free和fresh-start，拒绝partial。`test_complete_predicate_requires_fresh_teacher_free_provenance` |
| 日志线程阻塞且队列满时close无限等待 | 关闭用事件和有界join，不向满队列阻塞插sentinel；失败后关闭锁存。`test_trace_close_is_bounded_when_queue_full_and_writer_stalled` |

恢复历史开发场景后，发现原先未使用的测试预留坐标中有位置距历史开发点不足5毫米。第一次优化之前另封存[最终位置协议](../final_position_protocol.json)，只替代尚未使用的test坐标，训练/开发采集位置和已生成资料不变；新20位置与全部训练/开发/历史位置的最小距离约5.099毫米。`test_final_positions_exclude_every_selected_development_case`覆盖该约束。

修正证据：[RED](final_review_red.log)、[附加排除区RED](final_position_exclusion_red.log)、[GREEN](final_review_green.log)、[完整测试](full_suite_after_review.log)。代码准入不代表任务性能或训练数据合格。

Minor延期：仿真视觉适配器内部登记记录使用调用序号与合成时间，尚未透传真实曝光ID。外层时效保护仍使用实际仿真曝光。此项不能作为真实硬件时间合同已完成的证据，真机只读接入前须补齐；本轮无硬件入口。

审查者未评判的范围：真实几何及硬件时序/接触安全；未训练策略的学习性能；150条数据完成情况；真实适配、40次最终验收及真机执行；未修改vendor的全部历史行为。实现者裁定这些仍按各自独立门槛阻止后续阶段，不从本次代码复核推断通过。

独立阻断：采集27条合格、2条失败、4条未完成后停止。两条失败发生在当前杯体颜色连通域歧义；未启动任何优化。详见[停止后数据审计](data_after_stop.json)、[失败帧重现](visibility_failure_diagnosis/result.json)。

## 后续观察器修正验证

在同一次修正流程中，补充了失败图像的窄缝连接回归、图像VDS来源变化回归，以及版本化数据清单入口，均观察RED后GREEN。完整测试扩展至53/53。[3002帧修正后审计](dense_visibility_fragment_repair/result.json)通过；原失败的`train_04_09`、`train_06_03`在修正版本各复核一次，均完成整段物理教师任务，见[新版本清单](../data/v2_repaired_01/manifest.json)。它们不是ACT策略任务成绩，旧版本两次失败保留。

`rebuild_data.py`属于实现者在此修正流程补充的工具，没有第二次独立审查；通过实际源文件的逐帧重观测及53项软件回归验证。既有成功物理轨迹只重算视觉特征，不重采原始RGB/动作；新版本再有失败就取消未开始工作，不自动重试。主数据集150条及优化预算保持。
