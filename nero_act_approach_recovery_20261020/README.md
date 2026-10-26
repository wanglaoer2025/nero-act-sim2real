# ACT下降恢复数据修复（2026-10-20）

**当前：A机制6/6、B为5/6；A原高位完整抓放0/1，仍未完全修复。** 已修正监督/恢复入口缺陷，补充实际失败恢复并完成五轮隔离对照；当前完整开发在2574点关节限位拒绝，未GRASP。

[完整失败原因、修正及各轮结果](evidence/repair_results.md)；[最终核验](evidence/final_verification.json)。最新393项测试通过、5跳过；同批次A/B各1000步，68条数据、37/12组，原67条保持。未执行20轮正式训练、新独立16项或真机。

下一轮训练范围调整待用户选择，状态见[evidence/pipeline_status.json](evidence/pipeline_status.json)。不将低位恢复、教师成功或6/6机制记作完整高位流程通过。
