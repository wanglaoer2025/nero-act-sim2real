# NERO 完整抓放 ACT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 生成并训练覆盖接近、抓取、抬升、搬运、放置、退出的 ACT，在无教师仿真闭环通过后接入真实图像和反馈完成完整幅度真机抓放。

**Architecture:** 所有新增代码与产物位于 `/home/user/wq/nero_act_full_pick_place_20261017`，原包只读。复用官方 ACT 适配和 ROS 独占 CAN 链路，独立实现完整任务物理教师、评估及接触执行合同。运行时只执行 ACT 生成且通过保护的动作，不以几何教师或回放接管。

**Tech Stack:** MuJoCo 3.15.0，Python，NumPy，h5py，Torch 与 torchvision，ROS 2 Jazzy，MoveIt，SocketCAN 接收观察器。

**Spec:** [完整方案](../specs/2026-10-17-nero-act-full-pick-place-design.md)，用户已批准。

## Global Constraints

- 20 步八维绝对目标，执行前 10 点；七关节 rad，夹爪总开口 m。
- RGB 单相机，倒装图像旋转 180° 一次，模型输入 90×160。
- 每 50 ms 七轴目标变化不超过 0.002 rad，夹爪变化不超过 0.5 mm；不得用整体缩幅代替完整任务。
- 保留发送前 250 ms、块内 750 ms 时效，以及原物理限位、健康、碰撞和控制独占要求。
- 先实现完整物理示教与冻结策略仿真评估，再验证真机 shadow；未通过时保持失败和禁止下发状态。
- 桌高 0 mm 与 TCP `[152,0,−23.5] mm` 仍为已接受近似值，接触阶段保留现场观察确认。
- 不改旧包、旧权重、旧标定或既有授权；不自行提交、清理或回退文件。
- 当前目录不是 Git 仓库，使用新目录隔离，保存哈希与每次结果，无 commit 步骤。

## Review Focus

1. 杯子原已在蓝卡上：必须有非零搬运和释放后独立承托，不能仅按终点距离判断任务成功。
2. 起点和相机基准不一致：先核对 link7 与 TCP，再比较同关节状态 FK，禁止删相机内参属性绕过版本。
3. 仿真接触夹力高于真机：低力条件失败时禁止部署，不能以自由宽度跟随要求替代持物健康判据。
4. 人工等待使预测过期：恢复时重新采集和推理，不能消费旧动作块。
5. 仿真任务通过但实际执行合同违规：两项都必须通过，不能只汇报物体成功。

## Task 1 兼容环境与场景一致性

**Files:** `simulation/scene.py`，`scene/config.json`，`tests/test_scene.py`，`evidence/environment.json`。

**Interfaces:** `build_scene(config: dict) -> mujoco.MjModel`，`tcp_in_link7(model) -> (offset_xyz, rotation)`；消费原场景 XML 和当前真实配对状态，输出独立可渲染模型。

- [ ] 写 `test_scene.py`，要求模型载入、相机存在、关节限位保留，桌面上表面为 0，源和目标 XY 不同，TCP 变换与选定名义参数对应；拒绝非有限值与反射矩阵。
- [ ] 在隔离环境运行测试，确认缺少实现时失败；不修改旧环境。用 `/home/user/nero/bin/python -m venv --system-site-packages .venv` 创建新目录环境，再在其中安装指定兼容 MuJoCo；保存实际版本和依赖检查结果。
- [ ] 实现 `build_scene`，重用原网格和刚体链，保留标定相机，更新本轮独立桌面和源目标配置；只允许初始化重设物体状态。
- [ ] 运行场景测试和 EGL 渲染，生成实拍与渲染对照；使用只读 MoveIt FK 对相同关节状态复核机器人坐标。失败则定位差异，不进入示教生成。

## Task 2 完整物理教师与数据

**Files:** `simulation/teacher.py`，`simulation/collect.py`，`tests/test_teacher.py`，`training_data/manifest.json`，`training_data/split.json`。

**Interfaces:** `run_teacher(config: dict, record: callable | None) -> dict`；回调收到控制前 `qpos/qvel`、同周期 `target[8]`、`stage` 和 RGB；`collect_episode(config, output: Path) -> dict` 创建新 HDF5，不覆盖。

- [ ] 写失败测试：阶段必须包含 carry、place、release、retreat；源与目标相同拒绝；目标无效和不足开口拒绝；终点在源位不得通过。写真实物理集成测试断言抬升、非零转运、释放后无双指接触及 5 秒稳定。
- [ ] 运行测试确认失败；实现完整教师，复用现有数值 IK 和 NeroServo，按高位接近、下降、闭爪、抬升、搬运、下降、释放、退出执行，无杯子绑定和中途重置。
- [ ] 同周期保存并实际执行 float32 动作，保持 0.002 rad 与 0.5 mm 单点限制；低力接触参数明确定义，加入杯体和非预期接触检查。
- [ ] 验证一条真实物理全流程，然后生成至少 12 条示教，8 train、2 val、2 开发诊断，保留全部失败。固定种子与划分，重新执行保存动作核对物理重放一致性。
- [ ] 运行全部场景与教师测试；检查所有成功数据均具有完整放置和退出，不把旧 pickup/hold 数据算入完整示教数量。

## Task 3 完整任务 ACT 训练

**Files:** `training/train.py`，`training/config.json`，`tests/test_training_contract.py`，`model/`。

**Interfaces:** 复用旧包 `NeroActPolicy(config)` 和数据预处理；模型输入 `(state[8], RGB[90,160])`，输出绝对目标 `[20,8]`；`model/runtime_metadata.json` 绑定训练分组、权重与统计哈希。

- [ ] 写训练合同测试：统计只含 train；release 与 retreat 末尾窗口可采样；20 步均监督；反归一化保持绝对目标；无发布者和 SDK 导入。
- [ ] 确认测试失败，适配现有训练实现，加入完整阶段采样与 padding，保存可复现配置；不复制第二套 ACT 网络。
- [ ] 先做小批次训练与保存加载一致性验证，测量实际吞吐；再训练并按验证集选择 checkpoint。记录真实轮次和损失，冻结后才做独立测试。
- [ ] 运行训练合同和全部既有本任务测试，保存完整模型 schema、checkpoint 和统计摘要，硬件许可默认 false。

## Task 4 无教师闭环评估

**Files:** `simulation/evaluate.py`，`simulation/outcome.py`，`tests/test_outcome.py`，`evaluation/protocol.json`，`evaluation/summary.json`。

**Interfaces:** `evaluate(config, policy, scenario) -> dict`；`OutcomeMonitor.update(measurement: dict) -> None`，`OutcomeMonitor.result() -> dict`；只读取杯子真值作为监控，不生成策略动作。

- [ ] 写结果测试：原杯在目标但未搬运失败、抬升后滑落失败、夹住不释放失败、释放后退出碰杯失败、5 秒窗口截断失败；任务成功但任一执行契约失败也判部署失败。
- [ ] 确认失败后实现结果监控和 ACT 仿真循环，保存每次原始预测、施加目标、反馈、接触、碰撞和时刻，不调用教师。
- [ ] 在评估前固定基准 1、独立种子 8、时延 3、外力 2 的方案，绑定冻结权重哈希；所有组保存结果，诊断组分开。
- [ ] 运行完整评估，分别汇报物理任务、执行合同、总体分数和每项失败；任何部署前置条件不满足则不开放真机发送。

## Task 5 完整任务合同与实时 shadow

**Files:** `executor/contract.py`，`executor/policy_worker.py`，`executor/shadow.py`，`tests/test_contract.py`，`tests/test_worker.py`。

**Interfaces:** `FullTaskContract.validate_prefix(raw[20,8], observation, history) -> targets[10,8]`，拒绝不连续或过期目标；策略进程持久加载权重，输出实际相机、状态和推理时刻。

- [ ] 写失败测试：错误任务和窗口拒绝、绝对目标不重基、0.002 rad 与 0.5 mm 限速、超 10 mm 夹爪前缀拒绝、过期预测拒绝、暂停后必须重新推理、所有截断记录 raw/sent 差值。
- [ ] 实现新任务合同，不改旧 air profile；保留物理限位和时效，完整路径幅度不缩小。
- [ ] 复用相机和策略进程设计，完成单独的无发送 shadow；验证无动作客户端调用、无控制发布者、无 SDK 发送。
- [ ] 用新权重对新鲜真实 RGB 和状态推理，核对起点、场景、完整预览和合同；保存参数与模型哈希，仍保持许可默认关闭。

## Task 6 接触执行与实体验收

**Files:** `executor/ros_execution.py`，`executor/contact.py`，`run.py`，`tests/test_contact.py`，`tests/test_execution_guards.py`，`README.md`，`package_sha256.json`。

**Interfaces:** `ContactMonitor.update(health, observation) -> contact_state`，`execute_task(policy, contract, session, confirm) -> dict`；arm 与 gripper 通过已有 ROS 控制门执行 ACT 目标，SDK 仅接收。

- [ ] 写失败测试：接触与自由开口跟随分开；夹力异常、滑移、杯底未承托时禁止释放；真实反馈静止不能判运动成功；取消先尝试关门且失败仍取消轨迹；旧授权、多个发送者和过期场景拒绝。
- [ ] 复用 ROS 轨迹链、MoveIt 状态检查和杯体扫掠；仅检查 ACT 路径，不调用教师规划接管。默认无动作，显式执行参数加本轮现场口令才进入控制。
- [ ] 在只读模式完成服务与当前真实反馈联调，核对 CAN 独占、模型与场景哈希、节点唯一性和全部测试。
- [ ] 仿真及 shadow 全部通过后，现场重置杯子源位并复拍；确认实体急停、空杯和净空，执行一次完整 ACT 抓放。接触观察确认时继续采样，恢复后重新推理。
- [ ] 核对真实抓取、抬升、搬运、落桌、释放与退出；保存逐块预测和实际反馈，不仅看控制器 action success。
- [ ] 关闭发送并恢复只读，复拍最终结果，写外部验收记录且保留原始数据审核状态。全部交付做哈希与文档命令复核；未完成项如实列明。

## 执行选择

建议 Native，由当前会话按任务顺序实现。任务共享场景、状态和接触接口，顺序推进可减少不同实现间的不一致。计划涉及新增代码与训练，按 writing-plans 要求，用户审阅本计划并选择执行方式后开始。
