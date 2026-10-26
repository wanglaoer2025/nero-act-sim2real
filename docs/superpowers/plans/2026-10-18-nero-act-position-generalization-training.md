# NERO ACT 位置泛化新训练实施清单

用户已批准2026-10-18设计并明确要求开始新训练，沿用顺序实施。本清单细化已授权工作，不重新请求同范围批准。当前范围为离线数据、训练与评估，不包含硬件操作。

Spec: [位置泛化设计](../specs/2026-10-18-nero-act-position-generalization-design.md)。

工作目录：`/home/user/wq/nero_act_generalized_pick_place_20261018`。旧目录只读借用。无Git仓库，以初始/最终哈希及evidence/progress.md记录实现、测试和裁定。

## Task 1 隔离与故障基线

- 复制现有任务代码和ACT源码，不复制旧训练运行与数据；复用现有虚拟环境。
- 复制run13和run11冻结包为初始化/诊断来源，保留其失败或未部署状态。
- 复跑run13名义监督仿真，记录放置非预期接触；复核成功run11/v6作为完整流程物理基线。不同版本协议明确区分，不能冒称run13已完整通过。
- 审计已有真机episode的采样、相机时刻、动作事件和时间配对；不能通过审计则不纳入监督。

## Task 2 分布生成与完整示教

Files: `simulation/generalization.py`、`simulation/collect_generalized.py`、`tests/test_generalization.py`。

Interfaces: `episode_config(center, index, split, attempt=0) -> dict`，`validate_initial_pose(model, config, center) -> dict`，`collect_dataset(output, count=48) -> manifest`。

- 先写失败测试：源杯覆盖±30mm、蓝卡固定、种子/场景分组不泄漏；初始TCP±20mm及姿态±3°同肘型、夹爪90～95mm，非法索引/越限拒绝。
- 复用物理教师、IK、低力伺服、same-cycle保存、合同检查和精确重放。
- 48条完整示教，36train/12val；按场景独立分组。失败独立保存，不计成功，不改力/碰撞/合同限制。每个场景最多三次不同的、记录清楚的安全候选；不足48不得发布完整训练清单。
- 首条及边界样本先验证、测速，再完成全体。颜色/光照及0～100ms延迟记录在每条scene_config，不平移图像伪造标签。

## Task 3 新数据统计与上下文训练

Files: `training/generalized.py`、`training/rebase_initializer.py`、`tests/test_generalized_training.py`。

Interfaces: `validate_manifest(manifest, split) -> paths`，`rebase_initializer(checkpoint, old_stats, new_stats, config) -> state_dict`。

- 先写失败测试：任意合格48条完整清单和36/12划分；拒绝未验证、重复/泄漏及角色错误数据；统计只取train。
- 现有训练入口硬编码12条且拒绝新统计微调，新增专用训练适配器；复用现有采样、损失和模型，避免修改旧入口。
- 新统计与9维因果上下文配对；旧初始化权重的输入投影及输出头作严格仿射换算，以物理预测不变测试证明，不混用旧统计。仅作initializer，不能当新训练模型导出。
- 首轮20轮、batch32、验证集选checkpoint，完整20点监督、前10权重5，继续低力/连续性原合同。先小批次联调再正式训练；记录数据、源码、初始化与统计哈希。
- 因果历史来自已执行action[t-1]，不使用未来动作；数据增强时同步修正上下文。

## Task 4 导出与冻结闭环评估

Files: `simulation/evaluate_generalized.py`、`tests/test_generalized_evaluation.py`，复用既有export和supervised_rollout。

- 只导出完成声明轮次的验证集最佳checkpoint，硬件许可保持false。
- 冻结后测试9个杯位、4个高位起点、3个现场匹配随机工况，共16项；测试实例不进入训练选择。
- 每项保存原始预测、施加目标、反馈、物理接触、阶段确认与失败。完整任务及合同同时通过才记成功，模拟确认不是硬件授权。
- 训练完成与16项评估通过分开报告；失败不重写成训练成功即任务成功。需要纠正时先记录原因，不无限追加训练。

## Review Focus

- 位置扰动改变动作与渲染场景，不能只改画面或只改标签。
- 新统计换算必须保持原模型物理读出一致，尤其七轴残差锚定与夹爪均值项。
- 36/12分组以独立场景为单位，失败重试不能泄漏来源。
- 真实记录若时间配对不可靠，拒绝混入；单条真实数据不能同时作为独立泛化验收。
- 保留所有失败与旧包；本轮训练不使能、不发CAN、不进行真机shadow或运动。

执行记录和新测试输出写入新目录evidence。训练与评估完成后按executing-plans做一次独立整体代码复核，并只针对重要问题修正验证。
