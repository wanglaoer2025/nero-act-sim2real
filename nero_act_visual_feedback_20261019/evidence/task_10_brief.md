## Task 10 在线短前缀闭环和可撤销的执行入口

**Files:** 修改`executor/{runtime,policy_worker,ros_execution,execution_guards,permission,shadow}.py`；创建`executor/visual_run.py`、`tests/test_visual_runtime.py`、`tests/test_visual_permission.py`。

**Interfaces:** `LiveWorker.predict()`保留`(raw20x8, details)`，details新增同帧观测、配对状态、请求ID、因果历史、标定哈希及各单调时间戳；`select_prefix(details: dict, phase: str) -> int`，近杯1～2点、其他阶段最多2点作为首版统一上限。`python -m executor.visual_run --model ... --profile ... --output ...`默认只读，显式`--execute --receipt ...`才可能运动。

- [ ] 写失败测试`test_send_two_advances_only_two_history_targets`、`test_grasp_confirmation_discards_previous_chunk`、`test_camera_stall_or_occlusion_latches_stop`、`test_no_execution_path_calls_full_rollout_or_ik`、`test_invalid_feedback_100_06mm_stops_before_close`、`test_cleanup_restores_readonly_without_retry`。
- [ ] 两点提交后`np.testing.assert_array_equal(history, submitted_rows[1])`，不能等于未提交第10点；过期/取消/遮挡后`assert sends_after_stop == 0 and automatic_retry is False`。回放生成与IK函数在测试中替换为调用即抛异常，正常在线流程仍可完成合成整段。
- [ ] 持久相机/观测/ACT工作进程每个请求用同一RGB观测几何并返回完整15维输入，反馈环保持150ms监督不被图像处理阻塞；每段执行后用新观测，连续性不通过不平移/重锚原始ACT预测。
- [ ] `validate_prefix`可先校验原10点，但只提交选定前缀，历史和点预算仅推进实际提交点；45ms ACK、250/750ms时效保留。GRASP等等待后丢弃旧块、重测杯位及夹指关系。接触保护姿态保持与ACT任务目标分别记来源。
- [ ] 用Task2/4实测几何建立并更新复合杯体碰撞对象、持物与释放状态；区域profile核验当前杯位的不确定范围，而非锁定旧源位±5毫米。检测到杯子被推移仍停止，不能把它当新的合法初始杯位。
- [ ] 权限绑定新schema、工作区、模型、统计、标定、观测代码和16项冻结报告；任何旧9维receipt、9/14旧报告、修改后几何或未通过时延均拒绝。默认/只读入口测试断言零动作客户端、零SDK发送，并可在无硬件环境运行。
- [ ] 此任务分两次完成：训练前先实现全部代码和无硬件测试，使用合成策略/报告夹具检查协议，夹具不可冒充部署证据；Task9通过后再运行隔离通信/取消/门控测试及真实只读shadow，记录新鲜度、健康间隔、唯一发送者和CAN计数，独立整体审查新改动。模拟通过不下发真机；由用户当前现场授权后才执行一次完整流程，保留四次确认、失败停止和终态只读证据。
