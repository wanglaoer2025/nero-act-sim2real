# NERO 实时视觉条件 ACT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 用户此前已选择由当前会话顺序实现，沿用此方式，不再要求选择执行方式。

**Goal:** 建立经过标定的15维实时视觉条件ACT，使杯位变化后仍可在线完成完整抓放，现场不逐次生成完整仿真回放。

**Architecture:** 原始RGB、时间对齐的8维实测状态、1维因果夹爪历史、6维目标相对位置形成统一观测，ACT生成20×8目标，仅执行满足时效的短前缀后重新观测。标定、观测、数据训练、执行共用版本化几何与观测约定；独立健康门控、碰撞和停止链继续使用既有实现。

**Tech Stack:** Python3.12、NumPy/SciPy、OpenCV、PyTorch、HDF5、MuJoCo、ZED SDK、ROS2/MoveIt。复用现有虚拟环境和依赖，新增依赖须有实际缺口证据。

**Spec:** [已批准设计](../specs/2026-10-19-nero-act-live-visual-feedback-design.md)。

**当前状态：** 用户已审阅并批准由当前会话顺序实现，随后确认尚无新标定、先完成离线工具。实际完成项、测试和前置依赖见[隔离工程](../../../nero_act_visual_feedback_20261019/README.md)及其持久账本；本计划不标记为全部完成。

## Global Constraints

- 七轴单位rad，开度及位置单位m；人工报告mm/deg。物理状态8维、数值策略输入15维、动作20×8。
- 输入顺序固定为`state_normalized[8] + previous_gripper_delta[1] + relative_positions_normalized[6]`；相对位置按base坐标表达，杯底在前、蓝卡在后。
- 上一夹爪目标取保护后实际提交历史，历史差除0.01m；不取本次预测、未来标签或整块未发送点。
- 初期训练源杯XY各±30毫米、独立验收各±20毫米，蓝卡固定；高位同肘型起点、已测杯把朝向。未验收区域不自动开放。
- 保留5%速度、夹爪≤1N、原5毫米最低名义间隙；误差预算要求更大时提高间隙。TCP≤1毫米、视觉位置≤2毫米为独立验证目标，非现有能力。
- 保留2mrad/0.5mm逐点限制、30mrad/10mm原始前缀连续性、250/750ms观测时效、150ms完整反馈/门控时效、45ms ACK及原3900有效点预算。新增图像与反馈对齐最大差50ms；不得用它放宽原时效。
- 期望10～20Hz完成新观测、推理与短前缀闭环，需实测；近杯每次1～2个50ms点。模型仍预测20点，完整幅度不缩小。
- 真值只用于训练标签审计/任务验收，不作为新增6维运行观测；新观测失败不可静默回退真值、旧帧或零向量。
- 四次关键确认GRASP、离桌后NEXT、承托后RELEASE、退出前NEXT。异常锁存停止，不自动重试/回位；恢复只读。
- 默认离线/只读，无动作客户端和SDK发送；现场采样运动、示教和完整真机试验分别需要当时授权，旧授权不复用。
- 当前`/home/user/wq`不是Git仓库。使用新隔离目录和初始/阶段/最终哈希记录，不为满足提交步骤初始化或改动历史仓库。

## Review Focus

1. 旧外参、错误光学坐标、180°旋转或单位混用会把稳定检测变成错误空间位置；Task2/3拒绝不匹配参数，并用独立3D控制点验证。
2. 新图像配旧状态、重复帧、完全遮挡、杯把被当杯身时不能继续下发；Task3/4/10分别验证观测拒绝与运行停止。
3. 将15维输入当15维物理状态，会污染残差读出、CVAE或统计；Task5验证仅前8维参与物理读出，输出始终20×8。
4. 短前缀只发2点，却用第10点推进历史，或等待确认后继续旧预测，会改变闭爪行为；Task10用真实已提交目标和确认后全新帧验证。
5. 名义仿真成功、训练损失下降或9/14旧成绩不能变成新部署许可；Task8/9/10验证证据绑定、独立测试与失败不可授权。

## 文件与依赖约定

新目录`/home/user/wq/nero_act_visual_feedback_20261019`，下文文件均相对此目录；文档仍登记在现有全流程文档。本计划不要求现在创建执行工程。

基础源为`nero_act_current_cup_trial_20261019`的`executor/`、`simulation/`、`training/`、`policy_source/`、`tests/`，保留完整9维因果上下文和CVAE/非线性夹爪实现。仅复制源码、所需网格及来源配置，不复制旧运行、训练数据、授权或模型为已部署产物。最新健康日志/门控/停止改动与`nero_act_replay_context_retest_20261019`逐项比较、移植并复测，不整目录覆盖在线runtime。该回放包runtime未传因果历史，不能直接充当新在线基线。

新模块职责：`calibration/solve.py`离线求解/独立验证，`calibration/bundle.py`工件和误差预算，`vision/observation.py`同帧几何观测，`vision/synchronization.py`帧状态配对，`executor/visual_observation.py`策略特征，`training/visual_dataset.py`训练同schema读取；其余复用已有文件。默认从新根目录执行`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest ...`；先检查pytest可用，再记录完整命令和结果。

阶段产物写`evidence/progress.md`和`evidence/pipeline_status.json`，分别记录工具完成、实测标定、数据、训练、完整仿真、shadow、真机结果，不能合并成单一success。每个任务完成后更新哈希；不修改旧封存包。

## Task 1 隔离工程和可重复的旧协议基线

**Files:** 创建`README.md`、`evidence/source_inventory.json`；移植必要`executor/health_trace.py`及其调用；测试`tests/test_isolated_baseline.py`、既有因果历史/停止测试。

**Interfaces:** 输入为上述只读来源树；输出`source_inventory.json`，包含相对文件名、来源绝对路径、SHA256、用途和未移植差异。后续任务只修改新目录。

- [ ] 检查目标目录不存在后建立隔离源码树和现有`.venv`链接，记录旧包765个依赖封存核验及所用初始化checkpoint。不存在有效源时明确失败，不猜测其他模型。
- [ ] 写测试`test_online_baseline_preserves_previous_submitted_gripper`和`test_baseline_uses_compact_complete_health_trace`：拒绝漏传历史、记录内容缺失、JSON导出变化；先验证测试暴露未移植差异。
- [ ] 断言固定请求中`previous_gripper_target_m`等于最后提交目标；健康记录导出`assert decoded_trace == original_samples`，不删嵌套状态以换性能。
- [ ] 移植已验证健康记录方式及必要门控/停止修正，保留在线因果历史；运行上述测试及原`test_stop_latch.py`、`test_gate_health_freshness.py`、`test_health_trace.py`、`test_retained_command_context.py`，不存在同名测试时复制对应来源测试并登记。
- [ ] 阶段记录：测试全通过、既有8/9维模型行为保持；任何遗留失败先裁定，不以新训练覆盖问题。

## Task 2 标定求解与独立误差验收

**Files:** 创建`calibration/{__init__,solve,bundle}.py`、`tests/test_calibration.py`、`config/calibration_samples.example.json`。

**Interfaces:** `solve_tcp(samples: list[dict]) -> dict`；`solve_camera(base_points_m: ndarray, pixels_rectified: ndarray, K: ndarray) -> dict`；`validate_calibration(solution: dict, holdout: dict) -> dict`；`load_calibration(path: Path, *, require_verified: bool) -> dict`。TCP样本固定为`sample_id,base_from_flange_4x4`，求解输出`tcp_offset_flange_xyz_m,fixed_point_base_xyz_m,rank,residual_m`。报告定义`schema=nero_visual_calibration_v1`、具名坐标变换、相机序列号、图像空间/尺寸、TCP与指面几何、杯口/杯把/桌面、训练与独立样本ID、误差界限、哈希、`verified`及`accuracy_target_met`。

- [ ] 写并运行失败测试：`test_tcp_recovery_and_rank_deficiency`用已知TCP和固定参考点验证毫米精度、拒绝同姿态秩亏；`test_camera_holdout_is_disjoint_and_metric`拒绝样本重用、米毫米错配和错误图像空间；`test_approximate_bundle_cannot_authorize`拒绝135毫米旧粗测、152毫米名义参数冒充实测。
- [ ] 合成无噪姿态断言`np.testing.assert_allclose(solution['tcp_offset_flange_xyz_m'], true_tcp, atol=1e-8)`；`assert report['synthetic'] is True and report['verified'] is False`。
- [ ] TCP使用`R_i*t+p_i=c`最小二乘，至少6组覆盖姿态且矩阵满秩；固定标定夹具/参考点与夹持中心关系须实测。相机使用已知base控制点的PnP和独立点检验，至少6个非共面控制点，检查正深度和刚体变换。求解残差不能代替独立验收。
- [ ] 独立TCP至少3个未参与求解姿态，相机至少9个覆盖源/目标与接近高度的控制点；报告最大误差、RMSE及旋转在指尖处的影响。`accuracy_target_met`仅在独立最大误差满足1/2毫米目标时为真；`verified`另表示真实独立采样、仪器不确定性、外形/桌面和实际误差界限已审核。未达精度目标时记录失败；仅可在按实测误差重新验证的较小工作区及更大间隙中使用，不自动加补偿或宣布达标。
- [ ] 实现CLI`python -m calibration.solve --samples ... --output ...`，默认不连接相机/机器人；示例与合成数据只证明求解器，输出标记`synthetic=true`且不可部署。运行专项测试与`--help`，记录通过。

## Task 3 相机与状态的实时只读观测

**Files:** 创建`vision/{__init__,observation,synchronization}.py`、`tests/test_visual_observation.py`、`tests/test_visual_sync.py`；修改`executor/camera.py`。

**Interfaces:** `CameraFrame`含`frame_id,rgb,exposure_monotonic_s,received_monotonic_s,camera_metadata_hash`；`GeometryObservation`含`frame_id,cup_bottom_base_m,target_base_m,cup_handle_yaw_rad,quality,error_bound_m,calibration_sha256,valid,invalid_reason`。`VisualObserver.measure(frame: CameraFrame) -> GeometryObservation`；`upright_to_optical(pixel_xy: tuple, image_size_wh: tuple) -> tuple`放在`vision/observation.py`；`pair_feedback(frame: CameraFrame, feedback: list[dict], *, max_skew_s: float=.05) -> dict`返回该帧配对的8维状态和时间误差。

- [ ] 写失败测试`test_unrotation_and_rectified_intrinsics_are_paired`、`test_cup_handle_and_occlusion_are_not_valid_centres`、`test_pair_rejects_51ms_skew_repeated_future_frame`；包括黄杯把、夹指遮挡、蓝色手机干扰、多目标和无目标，不能按最高置信度猜目标。
- [ ] 像素映射断言`assert upright_to_optical((0,0),(640,360)) == (639,359)`；同一映射不得做两次。遮挡样本断言`assert observation.valid is False`，51ms配对须抛`ValueError`。
- [ ] 图像旋转180°只在显示/模型预处理做一次，几何像素明确映回rectified光学图；每次相机启动保存实际K、尺寸、图像空间和时间映射。SDK曝光时间戳若不能可靠映射单调时钟，标记未满足在线时序，不伪造为接收时间。
- [ ] 从现有YOLO黄杯检测、颜色轮廓和蓝卡检测改造观测入口：分别拟合杯身/杯口并识别杯把，误差界限来自独立标定；不靠固定75/95毫米轮廓吃掉杯把。不能区分遮挡/杯身时`valid=false`。深度作为单独比较，不默认开启或替代未测高度。
- [ ] 用保存帧及合成像素验证上述测试，再实现`python -m vision.observation --readonly --calibration ... --output ...`，无动作客户端/SDK发送。相机设备采集仅在可用时只读运行，前后记录CAN TX；没有现场可见性样本时记录该项未验收。

## Task 4 统一15维特征与几何误差预算

**Files:** 创建`executor/visual_observation.py`、`tests/test_visual_features.py`；修改`calibration/bundle.py`、`executor/execution_guards.py`。

**Interfaces:** `relative_positions(state: ndarray, geometry: GeometryObservation, robot_geometry: RobotGeometry) -> ndarray[6]`；`require_visual_observation(frame, geometry, paired_feedback, *, now_s: float, calibration: dict) -> None`；`required_clearance_m(bounds: dict, *, speed_m_s: float, detection_s: float, stopping_s: float) -> float`。误差预算为TCP、视觉、外形、跟踪保守界限之和，再加上速度×检测/停车时长及2毫米额外余量，下限5毫米；这里“速度×时长”使用实测的最大扫掠速度界限，非仅TCP平移速度。

- [ ] 写并运行失败测试`test_relative_positions_are_base_metric_and_current_tcp`、`test_geometry_stats_use_train_only`、`test_stale_invalid_or_wrong_calibration_is_rejected`、`test_clearance_adds_errors_and_stop_distance`。相机/状态任一未来时间、过期、误标坐标、完全遮挡或缺失停车距离证据均拒绝。
- [ ] 测试使用具名bounds`tcp_m,vision_m,shape_m,tracking_m`，断言`assert required_clearance_m(bounds, speed_m_s=.01, detection_s=.1, stopping_s=.1) == pytest.approx(.009)`，各误差依次1/2/1/1毫米。
- [ ] 计算`[cup_bottom-current_tcp,target-current_tcp]`，不输入真值或手写动作目标。杯把朝向在已测范围外时拒绝；当前夹爪几何由开度及实测模型决定，FK不能使用旧TCP。
- [ ] 误差预算中未实测项不填零、无穷大不变成默认通过；测试1毫米TCP+2毫米视觉+1毫米外形+1毫米跟踪、速度0.01m/s和总停车0.2s时要求9毫米。不同位姿旋转影响折算到包络后计入。
- [ ] 专项测试通过，给Task5/7/10固定schema和质量检查入口；旧回放的固定5毫米规则不变。

## Task 5 15维ACT模型、初始化与工件合同

**Files:** 修改`executor/{model,contract,bundle,evaluation_integrity,policy_worker}.py`、`training/{train,command_context,rebase_initializer}.py`；创建`tests/test_visual_policy.py`。

**Interfaces:** 扩展`make_policy_observation(state, stats, config, previous_gripper_target_m=None, *, geometry_features_m=None) -> ndarray`；`FrozenAct.predict(rgb, state, previous_gripper_target_m=None, *, geometry_features_m=None) -> ndarray[20,8]`。新profile为`visual_relative_base_v1`，`state_dim=physical_state_dim=action_dim=8,policy_input_dim=15,chunk_size=20`；新增`geometry_mean/geometry_std`长度6。

- [ ] 写失败测试`test_old_8_9_and_new_15_profiles_are_explicit`、`test_readout_uses_only_first_eight_channels`、`test_init_9_to_15_preserves_same_image_prediction`、`test_cvae_encoder_and_prior_accept_15`、`test_geometry_changes_gradient_without_changing_output_shape`。缺少位置输入不回退9维；错误统计维数/NaN拒绝。
- [ ] 断言`assert observation.shape == (15,)`、`assert raw.shape == (20,8)`；只修改后6维时物理锚定状态不变；新增投影列初始化为0时`np.testing.assert_allclose(new_raw, old_raw, atol=1e-6, rtol=1e-5)`。
- [ ] 测试`FullTaskPolicy.forward`实际送入底层ACT的训练/验证输入均15维，修正原噪声分支只拼回前9维的问题；首版显式`training_observation_noise_normalized=[0]*8`，不引入与6维FK关系不一致的额外关节噪声。观测扰动通过Task3/7的成对观测路径生成。
- [ ] 只在明确新profile允许坐标条件；扩展`input_proj_robot_state`与`encoder_joint_proj`为15输入，新增6列初值0；初始化标记`initialization_only=true`，禁止当完成训练模型导出。保持前8维残差锚定和物理读出，不把位置通道混入8维状态统计。
- [ ] 新统计只取新训练分组；旧9维初始化按已有仿射重配方法换算前8维及读出统计，用同RGB/同物理状态验证零新增列时物理预测一致。单独比较160×90与320×180，不把分辨率变化混入等价测试。
- [ ] 保存配置schema、标定/观测代码/统计哈希；旧8/9维旧包仍严格验证原合同。运行专项及原模型/合同测试，正式训练前验证后验非零梯度、有效KL和零潜变量部署路径。

## Task 6 真实外形场景与完整示教

**Files:** 修改`simulation/{scene,teacher,collect_generalized,refresh_observations}.py`、`simulation/scene_template.xml`、`training/pose_objective.py`；修改`executor/ros_execution.py`的`RobotGeometry`与`cup_object`；创建`tests/test_measured_scene.py`。

**Interfaces:** `scene_from_calibration(calibration: dict, scenario: dict) -> dict`位于`simulation/scene.py`；`cup_collision_parts(calibration: dict) -> list[dict]`位于`calibration/bundle.py`，输出以杯底坐标表示的杯身、杯口及杯把碰撞体，两侧MuJoCo/MoveIt共用。旧模型只在legacy profile保留固定75/95毫米断言；新profile要求已验证实测包。

- [ ] 写失败测试`test_handle_collision_rejected_before_body_contact`、`test_real_dimensions_match_mujoco_and_moveit`、`test_nominal_tcp_cannot_generate_deployment_data`：复现旧圆柱有空隙但新杯把/杯口相交的案例，整臂与夹爪基座也不得漏检。
- [ ] 固定几何夹具断言`assert old_body_gap_m > .005 and measured_handle_gap_m < 0`，新间隙检查必须拒绝；两侧几何尺寸/局部位姿用`np.testing.assert_allclose(..., atol=1e-9)`比较。
- [ ] 由标定包生成一致外形和TCP，保持低力物理接触、无粘附/物体绑定，质量/摩擦未测量时保留假设并纳入评估变化。新几何不改动旧封存场景。
- [ ] `fk_spec_from_scene`及姿态损失使用同一实测场景生成的TCP/指面坐标；若实测TCP包含旋转，训练FK与执行FK均表达它，不沿用“TCP site必须无旋转”的旧限制。比较训练FK、MuJoCo和执行FK的独立姿态样本，禁止使用源配置中的旧外部XML路径计算新姿态损失。
- [ ] 在48个既有场景定义上重新采集36训练/12验证完整示教，记录几何、视觉有效性、20Hz实际目标与反馈；保留失败、不计入48，不通过则停止数据发布。杯把朝向按已测范围采样并记录。
- [ ] 全部通过精确重放、全动作合同、误差预算间隙及完整抓放，再发布新manifest。局部下降接触、闭爪、确认后恢复及释放退出数据单独登记来源，不能冒充完整示教。

## Task 7 观测一致的数据集与真实标签审计

**Files:** 创建`training/visual_dataset.py`、`tests/test_visual_dataset.py`；修改`simulation/{collect_generalized,supervised_rollout}.py`及`training/integrity.py`。

**Interfaces:** `VisualReferenceDataset(...)`保留既有返回`(images, observation[15], actions[20,8], padding, start)`；`visual_statistics(paths: list[Path]) -> dict`统计训练有效观测；HDF5新增`observations/visual_relative_m[N,6]`、`visual_valid[N]`、`frame_id[N]`、`visual_error_bound_m[N]`及相机/反馈/实际提交时间，原8维物理数据不变。

- [ ] 写失败测试`test_sim_and_real_use_identical_observation_function`、`test_no_truth_fallback_on_occlusion`、`test_previous_command_is_last_submitted`、`test_real_same_frame_state_is_not_action_label`、`test_scene_groups_and_retries_cannot_leak`。
- [ ] 同一RGB/标定/状态/历史断言`np.testing.assert_array_equal(training_observation, inference_observation)`；`assert train_groups.isdisjoint(val_groups)`；完全遮挡但真值存在仍不得产生有效窗口。
- [ ] 每个训练/推理时刻的新6维经Task3观测器从对应渲染RGB估计；仿真真值只存审计字段。无效观测不生成有效部署窗口，也不让仿真继续依赖真值走通；在闭环评估中同样停止并记录失败。
- [ ] RGB与6维几何若做增强，按增强类型同步变换标定/像素；纯颜色变化不改变坐标，裁剪缩放必须配套K，禁止只挪图不改动作。恢复数据继承实际上一提交目标，不使用快照后的未来动作。
- [ ] 没有可审计真实提交标签时不混入旧HDF动作监督；为后续新示教定义同schema记录，时间配对测试通过后才能纳入训练。用固定小样本验证数据集输出、统计隔离和延迟样本处理。

## Task 8 正式训练、分辨率比较与导出

**Files:** 创建`training/visual.py`、`config/visual_training.json`、`evaluation/protocol.json`、`tests/test_visual_training.py`；修改`training/export.py`及已有完整任务损失调用。

**Interfaces:** `python -m training.visual --data-dir ... --initializer ... --config ... --output ...`；输出完整声明轮次、逐轮验证、权重/统计/代码/标定哈希与训练终态。初始候选使用源run27，不把回放派生配置当新训练证明。

- [ ] 写失败测试`test_selected_parameters_and_cvae_receive_gradients`、`test_early_exit_cannot_export`、`test_new_statistics_and_normalization_are_bound`。新增投影、动作Transformer/CVAE及输出、ResNet的layer4参与训练，其余视觉层冻结；沿用已有非线性夹爪头。不得套用固定场景heads-only模式。
- [ ] 训练步断言`assert posterior_grad_norm > 0 and geometry_projection_grad_norm > 0 and layer4_grad_norm > 0`；冻结层`assert all(p.grad is None for p in frozen_parameters)`；训练终态不完整时导出必须抛`ValueError`。
- [ ] 固定种子42、batch32、20轮；Transformer/输出/新增投影学习率1e-4、layer4为1e-5、夹爪头3e-4；`cvae_training=true,kl_weight=.001,prior_gripper_loss_weight=.005,pose_loss_weight=.05,prefix_margin_loss_weight=0`，原前10点L1权重5。显式关闭heads-only/adapter-only，不继承run27的冻结后验适配模式；选模按固定零潜变量验证损失，记录夹爪/七轴/接触过渡分项。
- [ ] 先各用160×90和320×180同一小训练/开发场景做1轮吞吐、延迟及整段可推进诊断，权重不用于正式初始化。正式两种分辨率使用同分组/种子/轮次；在开发集完整任务结果和时效合同同时合格的候选间按预声明验证指标选择。不得用16项最终测试反复选模型。
- [ ] 训练前须Task2独立标定、Task6数据、Task7观测及Task10无硬件执行核心通过并封存源码；缺真实标定时仅允许合成单元/梯度测试，不宣称完成迁移训练。导出保留`hardware_deployment_allowed=false`，验证模型15维输入与20×8输出。
- [ ] 任何优化前冻结16项独立场景ID、种子和`evaluation/protocol.json`哈希；它们不用于分辨率、训练超参或checkpoint选择。开发对比与最终独立验收分别存放。

## Task 9 冻结完整泛化与实时视觉作用验收

**Files:** 修改`simulation/{evaluate_generalized,supervised_rollout}.py`、`executor/evaluation_integrity.py`；创建`tests/test_visual_evaluation.py`；核验Task8已冻结`evaluation/protocol.json`。

**Interfaces:** 复用`rollout(config, policy, stats, scenario, output)`，新增观测路径与短前缀记录；summary明确`passed,total,complete,finalized,calibration_sha256,observation_schema`和工件来源。单次完整通过要求下降无意外接触、夹持连续≥0.5秒、抬升/搬运达原任务目标、承托释放并退出后稳定至少5秒。

- [ ] 写失败测试`test_wrong_visual_condition_is_not_silently_ignored`、`test_partial_grasp_is_not_complete`、`test_clocks_and_confirmation_wait_require_fresh_queries`、`test_changed_geometry_invalidates_evaluation`。同状态下不同杯位应改变实际抓取落点；错配图像/位置条件在无硬件对照中必须可观察和归因，不以第一块差异代替成功。
- [ ] 只到GRASP的测试断言`assert result['success'] is False`；全套summary断言`assert summary['passed'] == summary['total'] == 16`才可作为区域通过，9/14输入明确拒绝。
- [ ] 冻结模型及全部依赖后运行9杯位、4有效起点和3现场匹配/误差延迟场景，共16项；同Task3观测、Task4预算与Task10短前缀规则，接触许可等待后重新取帧。验证16项场景ID、种子及协议哈希与优化前记录相同。
- [ ] 每项保存新图像/预测/提交目标/反馈/接触/失败/遮挡/视觉时延；执行点预算只计实际执行，保持原3900点上限。候选失败时记录原因，不自动调大预算或重训再挑同一测试实例。
- [ ] 全16项通过才发布该区域版本；另报视觉时效是否满足10～20Hz目标。若碰撞检查或视觉处理超时，停止并报告，不以关保护换吞吐；后续修正后用新的独立实例评估。

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

## 执行顺序与现场依赖

任务编号区分交付边界，实际依赖顺序为：Task1→Task2→Task3→Task4→Task5→Task6→Task7→Task10代码和无硬件测试→Task8训练→Task9冻结评估→Task10隔离/只读及现场阶段。这样训练、仿真与真机绑定同一已实现的短前缀规则，不在评估完成后才改执行源码。

Task1、Task2求解器、Task3/4只读工具、Task5模型接口可先用合成样本完成，不需要现场运动。Task2真实采样若尚未具备，可继续完成Task5及Task7/10的无硬件协议测试；不得用合成结果解除真实标定前置条件。正式Task6/8/9及真机阶段按各前置依次推进，不把未具备数据写成已完成。

每个任务先运行新增测试看见预期失败，再做最小实现，运行专项与受影响旧测试，保存阶段产物/哈希。Task9完整仿真与Task10隔离/只读验证完成后，再做一次独立整体代码复核。依照用户已批准方式由当前会话顺序实施；本计划不包含立即真机运动授权。

各任务的新增测试命令如下，从新工程根目录执行；首次应因明确缺失接口或行为断言失败，不能把依赖安装/硬件不可用当作预期红灯。实现后要求返回0且所有对应测试通过，再跑该任务列出的旧回归。命令描述的是待执行验证，当前没有运行这些尚未实现的测试。

| Task | 命令 |
| --- | --- |
| 1 | `.venv/bin/python -m pytest tests/test_isolated_baseline.py -q` |
| 2 | `.venv/bin/python -m pytest tests/test_calibration.py -q` |
| 3 | `.venv/bin/python -m pytest tests/test_visual_observation.py tests/test_visual_sync.py -q` |
| 4 | `.venv/bin/python -m pytest tests/test_visual_features.py -q` |
| 5 | `.venv/bin/python -m pytest tests/test_visual_policy.py -q` |
| 6 | `.venv/bin/python -m pytest tests/test_measured_scene.py -q` |
| 7 | `.venv/bin/python -m pytest tests/test_visual_dataset.py -q` |
| 8 | `.venv/bin/python -m pytest tests/test_visual_training.py -q` |
| 9 | `.venv/bin/python -m pytest tests/test_visual_evaluation.py -q` |
| 10 | `.venv/bin/python -m pytest tests/test_visual_runtime.py tests/test_visual_permission.py -q` |
