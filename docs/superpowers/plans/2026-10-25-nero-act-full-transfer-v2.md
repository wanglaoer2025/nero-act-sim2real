# NERO ACT 完整迁移 v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. 用户已要求按新方案执行，沿用当前会话顺序实施，不再询问执行方式。步骤用复选框记录，持久进度以新工程 `progress.md` 为准。

**Goal:** 建立15维实时视觉条件ACT完整抓放的新训练链，先完成统一观测、数据审计与无硬件运行验证，再按已批准预算训练和验收。

**Architecture:** 隔离工程 `nero_act_transfer_v2_20261025` 固定源码及环境。复用官方ACT核心、现有教师与物理判据，提供同一份因果15维合同、可微CVAE/增量夹爪训练读出、当前图像约束的目标登记、过期停止和完整任务评估。历史记录只读。

**Tech Stack:** 已安装Python3.12、PyTorch2.5.1、torchvision、NumPy、OpenCV、SciPy、HDF5、MuJoCo、pytest，不新增硬件依赖。

**Spec:** [2026-10-25 新方案](../specs/2026-10-25-nero-act-full-transfer-v2-design.md)，用户已批准执行。以下接口和任务细化不扩大方案训练预算或真机授权。

## Global Constraints

- 输入15维、物理状态8维、输出20×8；七轴rad，开度及几何m；只执行前2点。
- 上一夹爪命令取曝光前已实际提交的命令；不填补零、未来、同帧或未提交目标。
- 夹爪每点相对查询previous±1毫米，无cumsum；监督前2点；反馈开度±9.5毫米、物理0～95毫米边界保留。
- 不继承1a28动作权重。128/512/2/2模型、视觉末层、CVAE与动作主干及增量头按新方案训练；其余视觉预训练冻结。
- 现有标定优先复用；全遮挡/观测退化停止，背景稳定不能证明卡稳定。
- 预检最多200更新；正式单组20K；真实适配两组各最多3K；120train/30dev完整仿真示教；40独立试验只供最终一个候选。
- 正式训练需要完整同观察器数据与CUDA就绪。CPU可验证工具/模型梯度，不冒充正式GPU训练；不得据此跳过数据门槛。
- 本地初始检查不是Git仓库；按已批准方案使用独立目录和SHA，不初始化或变更历史仓库。
- 目前仅离线实现和数据处理，不打开ROS/CAN、不创建动作客户端、不发送使能/复位/夹爪/运动。

## Review Focus

1. 时间相等/乱序/未来或只预测未提交的命令不能进入previous；合同及数据集共同检查。
2. 当前卡体完全遮挡、单边退化、相机移动或旧帧不能通过静态登记继续；观察器及整条教师可见性审计覆盖。
3. 可微训练不能因旧Frozen wrapper截断梯度；后验、视觉layer4、decoder和增量头分别验证。
4. 日志阻塞、队列满、写入失败或执行时延不得使过期预测进入发送；纯内存发送器故障注入检查。
5. 代理真实标签、恢复片段、开发位置及动作阈值事件不能冒充完整主任务/独立位置验收。

## Task 1 隔离、资源与来源

**Files:** 创建`README.md`、`progress.md`、`source_manifest.json`、`evidence/resource_check.json`；固定`vendor/`必要ACT核心和教师依赖。

**Interfaces:** 来源清单记录source/destination/SHA；真实数据不复制大图，引用原路径并检查变化。

- [ ] 建立新目录，复用已安装环境，记录Git探测、CUDA实际分配和渲染后端；不得修装系统驱动。
- [ ] 固定必要源码，读取其许可证；列明未创建任何硬件发送者。
- [ ] 验证来源SHA与历史关键36份证据均未变化；资源失败作为后续依赖，不阻止独立工具实现。

## Task 2 因果合同和无硬件实时链

**Files:** `contract.py`、`runtime.py`、`tests/test_contract.py`、`tests/test_runtime.py`。

**Interfaces:** `CommandHistory.submit(target8,stamp,ack=True)`、`before(exposure)->target8`；`make_observation(state8,geometry6,previous,stats)->float32[15]`；`PendingPrediction`保存exposure/predicted/feedback时间、frame_id、20×8目标；`PrefixRunner.submit(prediction,now)`只有前2点提交且逐点重查时效；`AsyncTrace`有界异步写，写入故障拒绝后续发送。

- [ ] 先写测试，观看未来/同帧命令拒绝、归一化一致、反馈过期、日志阻塞后过期拒绝、停止锁存和队列满失败。
- [ ] 实现最小共享合同和可注入时钟/内存发送器；无SDK/ROS导入。
- [ ] 运行专项及全套pytest，保存RED/GREEN日志；测试不能把内存发送器成功当作硬件时序实测。

## Task 3 统一观察器与完整历史可见性核查

**Files:** `observer.py`、`tests/test_observer.py`、`observation_protocol.json`、`evidence/visibility_audit.json`。

**Interfaces:** `RegisteredCardObserver`多帧登记卡边界；`measure(rgb,stamp,frame_id)->GeometryEstimate`用当前卡体约束位姿并独立验证背景；`observe_episode`与运行调用同函数。估计值含source、首次登记时间、当前证据、不确定度和valid，不以历史值冒充当前测量。

- [ ] 先测试部分遮挡保留几何、单边沿边滑动、完全遮挡、背景移动、卡移动、重复帧；再实现。
- [ ] 复用已有相机投影与杯体估计，训练/运行不再使用不同卡域；必要拷贝依赖固定在vendor。
- [ ] 读取至少一条完整教师所有阶段图像；区分原未渲染帧、真实无效帧和新算法拒绝；不把稀疏成功样本说成完整可观测。
- [ ] 若已有图像不足，先对完整物理轨迹重渲染同一曝光图像并审计，不立即生成150条或训练。

## Task 4 数据审计、位置分组和教师采集

**Files:** `data_audit.py`、`protocol.py`、`collect_sim.py`、`tests/test_data.py`；输出`data/audit.json`和固定场景协议。

**Interfaces:** `audit_episode(path)->dict`检查观测、真实目标/回执、时间、各阶段和标定；`plan_positions()->dict`生成12train/6dev/20test组，组间至少5mm；`qualify_real_episode`缺字段即拒绝主15维，不修改旧RGB8视图。

- [ ] 先测试未来历史、未提交标签、半段恢复、位置泄漏、VDS源丢失与padding；再实现审计与分组。
- [ ] 重新检查68份已有仿真记录和79份旧真实记录，报告可直接用/只能诊断/需补项及字段来源；图像不无意义全量复制。
- [ ] 新观察器整段合格后，封存12/6/20位置协议，生成120/30完整物理示教；失败记录保留，不改变测试分组以迎合结果。
- [ ] 不能完成可见性或物理教师时，停止依赖采集/训练，继续独立可做工具，并报告具体未满足条件。

## Task 5 新策略、数据集和训练预检

**Files:** `policy.py`、`dataset.py`、`train.py`、`tests/test_policy.py`、`tests/test_dataset.py`。

**Interfaces:** `ACTPolicy.forward(obs15,rgb,actions=None,padding=None)`输出20×8；训练显式后验，无标签推理z0；损失arm20 masked L1＋grip2 increment MSE＋KL。训练与推理同一物理读出；数据集只使用同观察器有效窗口及已提交标签。

- [ ] 先验证输出、padding无损失、无cumsum、超范围标签拒绝、CVAE/decoder/layer4/增量头梯度与冻结层不变，观看RED再实现。
- [ ] 拷贝官方核心按来源绑定，动作层新初始化，ImageNet权重单独绑定；无隐式加载1a28。
- [ ] 训练入口默认检查完整数据、分组、配置、可见性与CUDA；缺项输出具体阻断，不开始正式优化。
- [ ] 仅在前提就绪时运行最多200更新预检，重放连续闭爪、保持、释放，记录净进展；CPU梯度单元测试不等同于这次数据预检完成。

## Task 6 固定预算正式训练和完整评估

**Files:** `evaluate.py`、`tests/test_evaluate.py`、`train_protocol.json`、`evaluation_protocol.json`。

**Interfaces:** 复用MuJoCo完整任务判据，默认dry-run，明确complete/full_task/position_scope；只有GRASP不可通过；部署仅首2点，不根据教师补任务。

- [ ] 先测试不完整任务、强制附着和预算超限不能通过、测试位置不用于选模。
- [ ] 前提全合格后正式20K，2K/5K/10K/20K开发选模；中心3次及邻近4/5允许适配。
- [ ] 有12条合格真实记录才执行sim-only/mix两组3K；否则列缺项，保留仿真成果而不宣称适配完成。
- [ ] 最终一个候选独占40次；来源/权重/统计/观察器均封存，完整成功至少32/40才报该范围通过。

## Task 7 整体复核和现场交付

**Files:** `evidence/final_review.md`、`results.json`、`README.md`、`progress.md`。

- [ ] 运行隔离工程全部测试，更新每项实际状态和来源哈希；历史证据保持。
- [ ] 使用一位新上下文审查者整体审查已完成代码和未完成依赖。Critical/Important单次RED→GREEN修复，记录未完成任务，不把被阻断训练标完成。
- [ ] 如前提缺失，具体报告当前产物和下一依赖；不删除非Git工程的持久台账。若全部仿真/适配/只读门槛就绪，准备一次固定布置完整试验的可核查包；现场运动仍按新方案明确范围单独执行。

## 验证命令

在新工程根运行：`.venv/bin/python -m pytest -q`。每任务先运行对应`tests/test_*.py`记录失败，再实现并运行全套。CLI只在存在后验证`--help`；具体采集/训练命令及真实输出登记README，不能事先宣称已运行。
