## Task 8 正式训练、分辨率比较与导出

**Files:** 创建`training/visual.py`、`config/visual_training.json`、`evaluation/protocol.json`、`tests/test_visual_training.py`；修改`training/export.py`及已有完整任务损失调用。

**Interfaces:** `python -m training.visual --data-dir ... --initializer ... --config ... --output ...`；输出完整声明轮次、逐轮验证、权重/统计/代码/标定哈希与训练终态。初始候选使用源run27，不把回放派生配置当新训练证明。

- [ ] 写失败测试`test_selected_parameters_and_cvae_receive_gradients`、`test_early_exit_cannot_export`、`test_new_statistics_and_normalization_are_bound`。新增投影、动作Transformer/CVAE及输出、ResNet的layer4参与训练，其余视觉层冻结；沿用已有非线性夹爪头。不得套用固定场景heads-only模式。
- [ ] 训练步断言`assert posterior_grad_norm > 0 and geometry_projection_grad_norm > 0 and layer4_grad_norm > 0`；冻结层`assert all(p.grad is None for p in frozen_parameters)`；训练终态不完整时导出必须抛`ValueError`。
- [ ] 固定种子42、batch32、20轮；Transformer/输出/新增投影学习率1e-4、layer4为1e-5、夹爪头3e-4；`cvae_training=true,kl_weight=.001,prior_gripper_loss_weight=.005,pose_loss_weight=.05,prefix_margin_loss_weight=0`，原前10点L1权重5。显式关闭heads-only/adapter-only，不继承run27的冻结后验适配模式；选模按固定零潜变量验证损失，记录夹爪/七轴/接触过渡分项。
- [ ] 先各用160×90和320×180同一小训练/开发场景做1轮吞吐、延迟及整段可推进诊断，权重不用于正式初始化。正式两种分辨率使用同分组/种子/轮次；在开发集完整任务结果和时效合同同时合格的候选间按预声明验证指标选择。不得用16项最终测试反复选模型。
- [ ] 训练前须Task2独立标定、Task6数据、Task7观测及Task10无硬件执行核心通过并封存源码；缺真实标定时仅允许合成单元/梯度测试，不宣称完成迁移训练。导出保留`hardware_deployment_allowed=false`，验证模型15维输入与20×8输出。
- [ ] 任何优化前冻结16项独立场景ID、种子和`evaluation/protocol.json`哈希；它们不用于分辨率、训练超参或checkpoint选择。开发对比与最终独立验收分别存放。

