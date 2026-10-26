# ACT命令增量分支：失败分析与离线验证

当前：正式1000更新完成，原三个失败输入3/3通过，原六项由2/6提升至5/6。完整高位流程尚未运行，真机未部署。剩余名义已夹持起抬被原桌面保护拦截。

- [完整分析、已执行方案与下一阶段](evidence/results.md)
- [增量表示设计](design.md)
- [执行记录](progress.md)
- [输入及原六项原始结果](results/qualification/summary.json)
- [开发模型](training/pilot_A/policy_last.ckpt)

模型仅用于离线开发，不能视作完成真机迁移或泛化验收。保留全部历史及失败日志，入口拒绝覆盖已有训练/评估结果。复现实验须在新隔离目录进行，评估前先创建`results/mechanism_original`和`results/nominal_after`两个父目录。当前目录用于留档，不重复追加更新。
