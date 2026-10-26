## Task 9 冻结完整泛化与实时视觉作用验收

**Files:** 修改`simulation/{evaluate_generalized,supervised_rollout}.py`、`executor/evaluation_integrity.py`；创建`tests/test_visual_evaluation.py`；核验Task8已冻结`evaluation/protocol.json`。

**Interfaces:** 复用`rollout(config, policy, stats, scenario, output)`，新增观测路径与短前缀记录；summary明确`passed,total,complete,finalized,calibration_sha256,observation_schema`和工件来源。单次完整通过要求下降无意外接触、夹持连续≥0.5秒、抬升/搬运达原任务目标、承托释放并退出后稳定至少5秒。

- [ ] 写失败测试`test_wrong_visual_condition_is_not_silently_ignored`、`test_partial_grasp_is_not_complete`、`test_clocks_and_confirmation_wait_require_fresh_queries`、`test_changed_geometry_invalidates_evaluation`。同状态下不同杯位应改变实际抓取落点；错配图像/位置条件在无硬件对照中必须可观察和归因，不以第一块差异代替成功。
- [ ] 只到GRASP的测试断言`assert result['success'] is False`；全套summary断言`assert summary['passed'] == summary['total'] == 16`才可作为区域通过，9/14输入明确拒绝。
- [ ] 冻结模型及全部依赖后运行9杯位、4有效起点和3现场匹配/误差延迟场景，共16项；同Task3观测、Task4预算与Task10短前缀规则，接触许可等待后重新取帧。验证16项场景ID、种子及协议哈希与优化前记录相同。
- [ ] 每项保存新图像/预测/提交目标/反馈/接触/失败/遮挡/视觉时延；执行点预算只计实际执行，保持原3900点上限。候选失败时记录原因，不自动调大预算或重训再挑同一测试实例。
- [ ] 全16项通过才发布该区域版本；另报视觉时效是否满足10～20Hz目标。若碰撞检查或视觉处理超时，停止并报告，不以关保护换吞吐；后续修正后用新的独立实例评估。

