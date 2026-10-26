# ACT事件结束示教微调实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans顺序执行。

**Goal:** 消除固定等待导致的近似标签冲突，保持新相机接近行为并验证完整抓放。

**Architecture:** 独立离线专家副本调用稳定事件门；复用collect、RGB转换、ACT损失与保护，新增当前相机接近回放。

**Tech Stack:** Python、MuJoCo、PyTorch、HDF5、现有RTX4090。

**Spec:** docs/superpowers/specs/2026-10-24-nero-act-event-finetune-design.md

## Global Constraints

- 当前1a28初始化、15维/20×8、配对统计和参数范围保持；最多1000更新，固定最终模型。
- 只读原数据；不接入未通过的RGB-D；所有真实资格为false；不访问ROS/CAN/机器人。
- close/release连续稳定0.5秒、原8秒上限；条件缺失或超时拒绝。
- 数据完整回放、合同及监督消歧通过才训练；原相机成功接近保持转为当前相机回放。

## Review Focus

- 夹持断续、非有限读数、杯子失稳及释放时未承托不能通过事件门。
- 专家模块切换只在数据进程局部生效，不改变正常运行时。
- 接近保持输入必须匹配原查询曝光、前一夹爪命令及RGB；目标来自实际初始化模型。
- 训练/验证组隔离，固定最后模型，不用失败ACT作为专家标签。
- 所有失败保留，模拟自动确认不能算实机验收。

## Task1: 事件示教与数据门

Files: event_gate.py、teacher_event.py、prepare.py、test_event_gate.py、data/manifest.json。
Interface: StableEvent(stage,control_hz).update(measurement,previous_gripper_target_m)->bool。

- [ ] 写失败测试：连续稳定、失稳清零、缺失/非有限值拒绝、未落桌拒绝、超时拒绝。
- [ ] 最小实现事件门；复制专家并仅替换close/release固定等待。
- [ ] 生成5个完整物理专家与RGB；回放/前缀合同验证，近似冲突审计；不通过不训练。

## Task2: 新相机保持和微调

Files: preservation.py、train.py、test_preservation.py、training/。
Interface: 当前相机96条观测/RGB/20×8基线参考，assess输出首两点物理误差。

- [ ] 测试保持损失只比较执行首两点、参考无梯度；验证形状与来源绑定。
- [ ] 复现源查询并保存保持数据，初始化误差阈值1e-6 rad/m；失败停止训练。
- [ ] 数据门通过后单组1000步，保存采样顺序、冻结核对、梯度比例和最终模型。

## Task3: 完整回归及交付

Files: evaluate.py、evaluation_protocol.json、results/、README.md。

- [ ] 同运行时评估三个开发场景与两项新位置，保存前后结果及全部保护停止原因。
- [ ] 运行相关测试，一次最终独立只读复核；重要问题单次RED→GREEN修正。
- [ ] 记录实际更新数、完整成功率、接近保持及未解决问题，真实部署维持false。
