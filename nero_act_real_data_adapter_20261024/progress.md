# SDD ledger — plan: /home/user/wq/docs/superpowers/plans/2026-10-24-nero-real-data-adapter.md

用户要求基于现有资料完成适配，沿用顺序实施；不重复请求方案批准。

工程选择：8维未来状态辅助视图，默认2帧；缺少确认历史和视觉几何时不冒充15维。风险：辅助视图的训练效果尚未验证，不能直接部署。

工程选择：复用旧37/10 episode划分，32条reserve。风险：不是位置独立泛化评估；不据此扩大能力声明。

工作区无Git，以独立目录和来源哈希替代worktree/commit；原始项目只读。

Pre-flight: Task 1提供转换、读取和train统计；Task 2使用相同split和8维/20×8合同。Task 3消费真实结果，不能把8维接口通过写成15维部署通过。未发现接口冲突。

Task 1: complete — 10项测试先因未实现而FAIL，新增转换、VDS源检查、8维/20点读取及train统计后，完整pytest 10/10 PASS。修改归一化边界测试为仅传val，避免重复来源检查掩盖非train拒绝。

Task 2: complete — 新增split四项及独立核对一项测试，分别观察4/4和1/1未实现FAIL，实施后全套15/15 PASS。实数据79份导出56,261帧，37 train/10 val/32 reserve；79/79数值及时序精确匹配，158对VDS图像一致。真实batch经既有ACT新初始化网络CPU前向输出[2,20,8]、loss有限，无optimizer.step；原15维审计仍拒绝。prepare/preflight两个CLI帮助通过。

Task 3: complete — README和主报告24.6节已更新；独立整体复核无Critical/Important/Minor，独立重算train统计及来源分组匹配，证据见final-review.md。交付前再次运行完整测试并更新来源索引、HTML及integrity.json；不启动训练或真机。
