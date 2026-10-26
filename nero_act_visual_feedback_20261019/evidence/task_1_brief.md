## Task 1 隔离工程和可重复的旧协议基线

**Files:** 创建`README.md`、`evidence/source_inventory.json`；移植必要`executor/health_trace.py`及其调用；测试`tests/test_isolated_baseline.py`、既有因果历史/停止测试。

**Interfaces:** 输入为上述只读来源树；输出`source_inventory.json`，包含相对文件名、来源绝对路径、SHA256、用途和未移植差异。后续任务只修改新目录。

- [ ] 检查目标目录不存在后建立隔离源码树和现有`.venv`链接，记录旧包765个依赖封存核验及所用初始化checkpoint。不存在有效源时明确失败，不猜测其他模型。
- [ ] 写测试`test_online_baseline_preserves_previous_submitted_gripper`和`test_baseline_uses_compact_complete_health_trace`：拒绝漏传历史、记录内容缺失、JSON导出变化；先验证测试暴露未移植差异。
- [ ] 断言固定请求中`previous_gripper_target_m`等于最后提交目标；健康记录导出`assert decoded_trace == original_samples`，不删嵌套状态以换性能。
- [ ] 移植已验证健康记录方式及必要门控/停止修正，保留在线因果历史；运行上述测试及原`test_stop_latch.py`、`test_gate_health_freshness.py`、`test_health_trace.py`、`test_retained_command_context.py`，不存在同名测试时复制对应来源测试并登记。
- [ ] 阶段记录：测试全通过、既有8/9维模型行为保持；任何遗留失败先裁定，不以新训练覆盖问题。

