## Task 7 观测一致的数据集与真实标签审计

**Files:** 创建`training/visual_dataset.py`、`tests/test_visual_dataset.py`；修改`simulation/{collect_generalized,supervised_rollout}.py`及`training/integrity.py`。

**Interfaces:** `VisualReferenceDataset(...)`保留既有返回`(images, observation[15], actions[20,8], padding, start)`；`visual_statistics(paths: list[Path]) -> dict`统计训练有效观测；HDF5新增`observations/visual_relative_m[N,6]`、`visual_valid[N]`、`frame_id[N]`、`visual_error_bound_m[N]`及相机/反馈/实际提交时间，原8维物理数据不变。

- [ ] 写失败测试`test_sim_and_real_use_identical_observation_function`、`test_no_truth_fallback_on_occlusion`、`test_previous_command_is_last_submitted`、`test_real_same_frame_state_is_not_action_label`、`test_scene_groups_and_retries_cannot_leak`。
- [ ] 同一RGB/标定/状态/历史断言`np.testing.assert_array_equal(training_observation, inference_observation)`；`assert train_groups.isdisjoint(val_groups)`；完全遮挡但真值存在仍不得产生有效窗口。
- [ ] 每个训练/推理时刻的新6维经Task3观测器从对应渲染RGB估计；仿真真值只存审计字段。无效观测不生成有效部署窗口，也不让仿真继续依赖真值走通；在闭环评估中同样停止并记录失败。
- [ ] RGB与6维几何若做增强，按增强类型同步变换标定/像素；纯颜色变化不改变坐标，裁剪缩放必须配套K，禁止只挪图不改动作。恢复数据继承实际上一提交目标，不使用快照后的未来动作。
- [ ] 没有可审计真实提交标签时不混入旧HDF动作监督；为后续新示教定义同schema记录，时间配对测试通过后才能纳入训练。用固定小样本验证数据集输出、统计隔离和延迟样本处理。

