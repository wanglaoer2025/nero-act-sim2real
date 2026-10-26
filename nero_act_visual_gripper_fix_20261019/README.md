# ACT视觉反馈：保留七轴模型的夹持力修复

第二次修复对照，离线训练与物理仿真，不连接真机。原15维视觉ACT的图像编码、CVAE、位置输入投影和七轴分支均保留；仅微调ACT内部已有的非线性夹爪分支，未添加手工闭爪或阶段真值输入。

## 为什么采用此对照

旧模型名义16项5/16成功，原场景停在闭爪间隙不足。第一轮修正了训练采样缺口并重训全部分支，原场景却在闭爪前意外接触停止（0/1、732动作点、夹爪95毫米、无GRASP）。[第一次完整失败诊断](../nero_act_visual_contact_fix_20261019/evidence/attempt1_failure_diagnosis.json)已封存。

本次保留已完成20轮的原320×180视觉15维模型（最佳第17轮），统计和全部非夹爪权重逐值保留；使用修正后的力建立输入并合并重复静止等待。保留CVAE已训练权重，本阶段不优化后验，夹爪适配使用实际部署的零潜变量路径。仍按验证组损失选取模型，不靠完整测试选检查点。

## 数据与保护

48个完整物理示教来源，36训练/12验证；20721有效图像池为上一次修复生成的只读共享数据。训练选择13833个不同输入、21925加权窗口，夹持间隙≥3毫米的close输入540个，旧采样为30个。配套统计沿用相同36训练组的原统计，保证相同输入下初始化及七轴读出不变。

仅夹爪优化器，非夹爪梯度必须为空，每轮比较所有非夹爪参数和缓冲区。四次确认、碰撞、接触、滑移、稳定夹持、点预算和速度/时效阈值均不变。旧16项为开发回归，另16项数值场景在优化前冻结，尚未执行。旧名义标定不构成真机精度验收。

## 验证

[独立审查](evidence/adapter_independent_review.md)：无Critical/Important发现。新增测试先失败后通过，证明夹爪一次更新后七轴输出及非夹爪权重逐值不变。完整回归287通过、5跳过。

训练及闭环结论将在实际运行后补充。训练：[training/gripper_force_building_v1/status.json](training/gripper_force_building_v1/status.json)。

## 复现入口

在本目录运行，CUDA/EGL需要可访问主机GPU。命令全部为离线；输出目录必须为不存在的新目录。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nero_visual_mpl .venv/bin/python -u -m training.visual_gripper_repair \
  --data-dir /home/user/wq/nero_act_visual_contact_fix_20261019/training_data/contact_complete_v1 \
  --initializer /home/user/wq/nero_act_visual_feedback_20261019/training/visual_retained_v1_320/checkpoints/policy_best_val.ckpt \
  --output training/new_gripper_repair

MUJOCO_GL=egl MPLCONFIGDIR=/tmp/nero_visual_mpl .venv/bin/python -u -m simulation.visual_evaluation \
  --training-run training/new_gripper_repair --nominal-only --output results/new_nominal

MUJOCO_GL=egl MPLCONFIGDIR=/tmp/nero_visual_mpl .venv/bin/python -u -m simulation.visual_suite \
  --training-run training/new_gripper_repair --development-regression --workers 4 --output results/new_development16
```

独立16项入口为同一visual_suite命令去掉development-regression，记录为独立数值工况；只能在候选与执行源码冻结后评估，不用该结果挑选检查点。当前入口均不具备下发真机功能。

## 第二次修复实际结果

20/20轮完成，按验证选择第3轮；208个非夹爪张量逐轮逐值不变，统计文件字节一致。[核验](evidence/training_completion_verified.json)。

原场景 **0/1**，1777点时停止：双侧接触、目标开度差3.04/3.45毫米，稳定夹持已成立，零意外接触；尚未离桌，触发持杯模型穿桌保护。[结果](results/nominal_gripper_force_building_v1/summary.json)。

进一步数值复核表明，预测TCP其实向上约31微米，而保护器模型用了初始源位及过早刚性假设；最初“向下”解释已更正。后续不重训模型，在[第三次保护器修复目录](../nero_act_visual_attachment_fix_20261019/README.md)修正当前持杯几何及首次离桌锁存。保持本目录训练源码与权重封存。
