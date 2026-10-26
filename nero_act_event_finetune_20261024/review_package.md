# 一次独立只读最终复核

Review scope: 本目录全部新增代码、对应spec/plan、训练/数据/保持证据；评估仍在运行，未完成的结果不得计成功。

No git repository. Baseline: 原1a28固定模型及`../nero_act_grouped_gripper_20261021/simulation/teacher.py`；比较`teacher_event.diff`与独立新增文件。不要改写任何文件或连接硬件，不启动额外评估/训练；可用CPU运行当前7项新测试。

Spec: `/home/user/wq/docs/superpowers/specs/2026-10-24-nero-act-event-finetune-design.md`
Plan: `/home/user/wq/docs/superpowers/plans/2026-10-24-nero-act-event-finetune.md`
Ledger: `progress.md`。既有封存来源不可修改；如需修正，将保留本次训练/结果原始记录。

Review Focus（原文）：
- 夹持断续、非有限读数、杯子失稳及释放时未承托不能通过事件门。
- 专家模块切换只在数据进程局部生效，不改变正常运行时。
- 接近保持输入必须匹配原查询曝光、前一夹爪命令及RGB；目标来自实际初始化模型。
- 训练/验证组隔离，固定最后模型，不用失败ACT作为专家标签。
- 所有失败保留，模拟自动确认不能算实机验收。

请按Critical/Important/Minor给出实际影响和文件行号，列出考虑后未判断事项。不要派子代理。本轮训练已完成1000步，新模型SHA见README；只评其真实证据，不推断真实部署就绪。
