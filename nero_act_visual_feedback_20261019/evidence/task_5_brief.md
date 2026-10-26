## Task 5 15维ACT模型、初始化与工件合同

**Files:** 修改`executor/{model,contract,bundle,evaluation_integrity,policy_worker}.py`、`training/{train,command_context,rebase_initializer}.py`；创建`tests/test_visual_policy.py`。

**Interfaces:** 扩展`make_policy_observation(state, stats, config, previous_gripper_target_m=None, *, geometry_features_m=None) -> ndarray`；`FrozenAct.predict(rgb, state, previous_gripper_target_m=None, *, geometry_features_m=None) -> ndarray[20,8]`。新profile为`visual_relative_base_v1`，`state_dim=physical_state_dim=action_dim=8,policy_input_dim=15,chunk_size=20`；新增`geometry_mean/geometry_std`长度6。

- [ ] 写失败测试`test_old_8_9_and_new_15_profiles_are_explicit`、`test_readout_uses_only_first_eight_channels`、`test_init_9_to_15_preserves_same_image_prediction`、`test_cvae_encoder_and_prior_accept_15`、`test_geometry_changes_gradient_without_changing_output_shape`。缺少位置输入不回退9维；错误统计维数/NaN拒绝。
- [ ] 断言`assert observation.shape == (15,)`、`assert raw.shape == (20,8)`；只修改后6维时物理锚定状态不变；新增投影列初始化为0时`np.testing.assert_allclose(new_raw, old_raw, atol=1e-6, rtol=1e-5)`。
- [ ] 测试`FullTaskPolicy.forward`实际送入底层ACT的训练/验证输入均15维，修正原噪声分支只拼回前9维的问题；首版显式`training_observation_noise_normalized=[0]*8`，不引入与6维FK关系不一致的额外关节噪声。观测扰动通过Task3/7的成对观测路径生成。
- [ ] 只在明确新profile允许坐标条件；扩展`input_proj_robot_state`与`encoder_joint_proj`为15输入，新增6列初值0；初始化标记`initialization_only=true`，禁止当完成训练模型导出。保持前8维残差锚定和物理读出，不把位置通道混入8维状态统计。
- [ ] 新统计只取新训练分组；旧9维初始化按已有仿射重配方法换算前8维及读出统计，用同RGB/同物理状态验证零新增列时物理预测一致。单独比较160×90与320×180，不把分辨率变化混入等价测试。
- [ ] 保存配置schema、标定/观测代码/统计哈希；旧8/9维旧包仍严格验证原合同。运行专项及原模型/合同测试，正式训练前验证后验非零梯度、有效KL和零潜变量部署路径。

