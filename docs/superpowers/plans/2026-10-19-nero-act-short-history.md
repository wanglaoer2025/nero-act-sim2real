# NERO Short-History ACT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 用户已选择由当前会话顺序实施，不再选择执行方式；本计划待审阅后开始实现。

**Goal:** 对比相同闭环恢复数据下的单帧A与短历史B，使在线ACT完成完整幅度抓取、抬升、搬运、放置、释放和退出，并如实检验历史是否有效。

**Architecture:** 共用因果历史合同、完整离线恢复快照、物理验收数据及运行器。B在当前RGB和15维输入之外，用5次15维观测及时间元数据形成零初始化GRU残差条件；A保留单帧条件。先各1000次更新验证机制，再依放行规则进行正式20轮对照和冻结完整任务评估。

**Tech Stack:** 现有Python3.12、PyTorch、NumPy、HDF5、OpenCV、MuJoCo及`.venv`；不新增依赖。离线工具不导入ROS/机械臂SDK，不使能或发送真机动作。

**Spec:** [已书面批准的具体设计](../specs/2026-10-19-nero-act-short-history-design.md)。用户回复“设计符合，生成实施计划”；本计划不是实现或训练完成记录。

## Global Constraints

- 物理状态8维，每帧策略数值15维，当前RGB为320×180；历史`[B,5,15]`，时间与有效掩码各`[B,5]`，输出绝对目标`[B,20,8]`。七轴rad、宽度和位置m。
- 历史旧到新、末项等于当前15维；五项全部有效才训练/预测。允许50/100毫秒实际间隔，相邻不超过150毫秒，总跨度不超过400毫秒；不插值、不重复旧帧、不取未来标签。
- 每项使用其曝光前已接受的最后命令；不使用当前命令覆盖过去，不采集尚有待执行前缀时的历史。最新帧仍满足配对≤50毫秒及250/750毫秒时效。
- 初始/确认后清空并采集5次新观测；A/B流程相同。新帧等待最多100毫秒，期间不推理或复用旧块，健康检查继续；运行中失效锁存停止，不自动回退单帧。
- GRU单层、隐藏128，输入15维加`age_s/0.1`时间元数据，128→128输出投影零初始化；作用于ACT与CVAE当前状态条件。物理锚点仍只用当前8维。
- 共同初始化SHA256为`b252dd578b4bf5626566ec5b60a47be0bdcfee30e9eab990cde05ac122bdea71`；初始化物理误差七轴≤1e-6rad、夹爪≤1e-7m。零潜变量验证/部署，历史不接触未来动作。
- A/B联合训练ACT/CVAE、状态投影、七轴、已有夹爪和ResNet layer4；更早视觉层冻结。学习率ACT/layer4均1e-5、夹爪3e-4、B历史1e-4；种子42、批32、相同样本与顺序。损失：前10点L1×5、KL 0.001、夹爪辅助0.005、姿态0.05、梯度裁剪0.1。
- 原48完整示教36/12分组；16条角色恢复12训练/4验证继承父分组，另2条名义诊断恢复同属额外训练组；整体37训练组/12验证组。恢复不是完整示教，失败ACT动作只作过去上下文。
- 小对照每组恰好1000次更新、只评估末步；抬升机制最多400个新增有效点、至少20毫米，不能计为完整任务。只有B通过规定机制/验证恢复才进行正式双组各20轮；仅A通过则停止扩大B，A进入完整开发。
- 全任务仍要求抬升≥180毫米、搬运>100毫米、目标承托释放、退出后100个50毫秒稳定帧、四事件齐全、零意外接触/执行合同失败、有效点≤3900。
- 最多2个50毫秒点，近杯原单点规则，5%速度、夹爪≤1N、2mrad/0.5毫米逐点限制、30mrad/10毫米原始前缀、150毫秒健康、45毫秒ACK、−1微米持杯桌面保护及离桌永久锁存保持。
- 名义标定仅限离线，哈希`33c9ca07d85a5ae3b4f33099a4d887308dc95e18cc5085c418a0131fd4d90be8`；物体真值仅用于教师、仿真安全门控/验收，不进入历史或ACT。
- `/home/user/wq`不是Git仓库。新工程隔离并以SHA256阶段账本代替提交；旧模型、源码、数据、结果及现场进程保持。当前审阅不启动真机。

## Review Focus

1. 过去观测误配现在的命令、确认后混入旧帧或共享数组被改：应拒绝预测，Task1/4测试绑定与不可变快照。
2. 零输出历史分支第一步内部无梯度、第二步仍无梯度，或只传入后验未传入零潜变量路径：Task3分别测试两次反向和先验/后验调用。
3. 相邻恢复跨train/val、稀疏图像重复充历史、B筛掉困难样本：Task5按父谱系分组并检验真实帧、共同样本交集及标签不变。
4. 恢复时丢失扰动计数、接触计时、确认、监测器或首次离桌锁存：Task2测试中断前后下一步及连续执行一致，禁止把真实停机授权恢复。
5. 1000步pilot被当20轮完成，或旧worker无历史预热绕过新profile：Task3/6/9验证加载与用途边界；未完成的训练不导出为完整候选。

## 文件职责与执行约定

新工程根为`/home/user/wq/nero_act_history_feedback_20261019`，所有后续源码路径相对此根；当前仅编写本计划，不创建该工程。基础源为`nero_act_visual_attachment_fix_20261019`，它包含第三次已验证的安全几何修正。模型来源为`nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1`，不重新命名为已通过模型。

只读原始48条数据源为`/home/user/wq/nero_act_generalized_pick_place_20261018/training_data/generalized_v1`；已有图像采样/权重参照为`/home/user/wq/nero_act_visual_contact_fix_20261019/training_data/contact_complete_v1`。新历史数据输出到新根的`training_data/history_complete_v1`，新恢复输出到`training_data/history_recoveries_v1`，不得覆盖原池。

| 文件 | 职责 |
| --- | --- |
| `executor/observation_history.py` | 共用历史合同、曝光前命令查找、不可变历史包 |
| `simulation/runtime_checkpoint.py`、`simulation/history_capture.py` | 完整离线恢复与冻结策略失败记录 |
| `policy_source/src/act/history_encoder.py`及现有ACT调用链 | 历史条件与原先验/后验、物理读出兼容 |
| 现有`executor/visual_inference.py`及`simulation/supervised_rollout.py` | 共同采集/预热/等待、绑定和短前缀闭环 |
| `training/history_recoveries.py`、`training/history_data.py`、`training/history_dataset.py` | 父来源恢复、图像历史派生及共同数据读取 |
| `training/history.py`、现有`training/integrity.py` | 配对初始化/优化、两类预算与完整性 |
| `simulation/history_mechanism.py`及现有完整评估模块 | 小机制试验和完整开发/独立评估 |
| `executor/history_shadow.py`及现有工件模块 | 保存观测的离线预测及历史版工件校验 |

每项遵循RED→最小实现→GREEN，完成后更新`evidence/progress.md`、`evidence/pipeline_status.json`及源码/工件哈希。测试命令从新根运行，环境统一`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nero_history_mpl`；物理图像生成/评估另加`MUJOCO_GL=egl`。CUDA/EGL若仅主机可见，按既有授权使用主机执行离线命令，核实实际优化步，不把启动信息当结果。

以下新增CLI均为**待实现接口**，实现并通过`--help`后才运行；其中PATH/NEW_DIR为需替换的工件路径，方括号表示可选参数、竖线表示互斥选择，实际调用使用`.venv/bin/python`；不把本计划中的命令记为当前可用能力。

## Task 1 隔离与共享因果历史合同

**Files:** 创建`executor/observation_history.py`、`tests/test_history_contract.py`、`README.md`、`evidence/source_inventory.json`。

**Interfaces:** `lookup_receipt(receipts: list[dict], exposure_s: float) -> dict`取`accepted=true`且`ack_s<exposure_s`的最后收据；收据含`receipt_id,rows,sent_s,ack_s,kind,scope`，只接受实际提交的1/2点。`make_history_item(packet: dict, receipts: list[dict], stats: dict, config: dict, *, epoch: int) -> dict`构造/校验该帧15维。`HistoryBuffer.clear(*,epoch:int,after_s:float)`、`append(item:dict)`、`snapshot(current:ndarray) -> HistoryBatch`；`HistoryBatch`为不可变记录，含`observations,age_s,valid,frame_ids,receipt_ids,epoch,context_sha256`及版本标识。采集时准入证明、状态/位置及各自时间也进入哈希。

- [x] 核实目标不存在；复制基础源的源码、资产、配置、测试、`pytest.ini`、`run.py`、小型`model/`和`initializers/`配置夹具，以及`evidence/{run18_pipeline.py,probe_fixed_heads.py,site_snapshot.json,retained_calibration_sources.json,baseline_development_protocol.json}`；链接现有`.venv`。不复制旧运行日志、收据或大训练输出。旧数据只读引用，独立协议保留源文件哈希。
- [x] 先运行继承测试，预期原基线292通过/5跳过；不同则先定位。新增`test_past_item_uses_its_own_receipt`、`test_history_order_span_and_confirmation`、`test_snapshot_cannot_be_mutated`、`test_invalid_admission_stops_history`。用五帧间隔0.1秒验证：
  ```python
  assert batch.observations.shape == (5,15)
  np.testing.assert_allclose(batch.age_s,[.4,.3,.2,.1,0])
  assert batch.receipt_ids[0] != batch.receipt_ids[-1]
  assert np.array_equal(batch.observations[-1],current)
  ```
- [x] Run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_history_contract.py`，先确认缺接口失败；实现上述合同。4帧、跨度0.401秒、相邻0.151秒、重复/乱序/未来帧、曝光前未ACK、跨标定/相机/会话/确认epoch、包被修改均拒绝。数组复制并设只读，不能只冻结dataclass外壳；无效历史不能变成单帧。
- [x] 重跑专项为PASS，登记复制来源、初始化模型哈希和阶段哈希；源码清单应确认所有旧源未变。

## Task 2 完整监督器快照与原失败重现

**Files:** 创建`simulation/runtime_checkpoint.py`、`simulation/history_capture.py`、`tests/test_runtime_checkpoint.py`；修改`simulation/supervised_rollout.py`的可选记录/恢复接口，不改默认旧动作。

**Interfaces:** `capture_runtime(model,data,servo,previous,*,control_state:dict,observation_state:dict,bindings:dict) -> dict`；`restore_runtime(model,data,servo,record:dict,*,expected_bindings:dict) -> tuple[ndarray,dict,dict]`。`control_state`须包含contact、phases、monitor、attachment、support_pose、active_points、全局物理点偏移、计时和随机数；`observation_state`含曝光历史队列、提交/确认边界、Task1历史和当时当前图像/哈希。`rollout(...,*,resume_context=None,capture_control=False,runtime_profile='legacy_trace_v1')`保留旧调用默认语义。

- [x] 新增`test_resume_preserves_next_prediction_and_guard`、`test_resume_preserves_disturbance_and_liftoff_latch`、`test_incomplete_or_live_checkpoint_is_rejected`。在两点前缀后、确认后、首次离桌后各中断，验证：
  ```python
  np.testing.assert_allclose(resumed_qpos,continuous_qpos,atol=1e-7,rtol=0)
  assert resumed_guard_decision == continuous_guard_decision
  assert resumed_active_points == continuous_active_points
  assert resumed_attachment_airborne == continuous_attachment_airborne
  ```
- [x] Run `.venv/bin/python -m pytest -q tests/test_runtime_checkpoint.py`观察RED；复用现有`simulation.snapshot.capture/restore`保存积分/伺服状态，显式白名单序列化其余状态。恢复后的扰动索引用全局物理点偏移，不能因新日志从零开始而重放错误扰动；验证所有源、场景、模型及历史绑定。
- [x] 将记录接入原循环，未指定新profile时不加预热/保持、不改预测；实现离线CLI `python -m simulation.history_capture --source-run PATH --output NEW_DIR`。只在仿真内部重建模拟许可；缺字段旧快照拒绝，不能恢复真实停机权限。
- [x] 专项GREEN后，用冻结模型重现原名义开发失败并采集完整上下文，核对原1792点/保护原因/发送目标及预测容差；若额外记录改变轨迹，停止后续并解释。保存`evidence/original_failure_control_capture.json`，不使用新16项。

## Task 3 ACT历史条件、严格初始化与调用链

**Files:** 创建`policy_source/src/act/history_encoder.py`、`tests/test_history_policy.py`；修改`policy_source/src/act/nero_policy.py`、`policy_source/third_party/act/detr/models/detr_vae.py`、`executor/model.py`、`executor/policy_worker.py`、`training/train.py`及相应配置/工件校验；在新副本`policy_source/docs/README.md`登记接口。

**Interfaces:** `NumericHistoryEncoder.forward(observations:Tensor,age_s:Tensor,valid:Tensor)->Tensor[B,128]`；所有`FullTaskPolicy/BoundedActPolicy/NeroActPolicy.forward`统一新增关键字`history:dict[str,Tensor]|None=None`，字典固定键`observations,age_s,valid`；DETRVAE接收`history_embedding:Tensor|None=None`。`FrozenAct.predict(...,geometry_features_m=None,history:HistoryBatch|None=None)->ndarray[20,8]`；新profile名`causal_numeric_gru5_v1`，A为显式`none`。

- [x] 新增`test_zero_history_projection_preserves_physical_policy`、`test_second_step_trains_gru`、`test_prior_and_posterior_both_receive_history`、`test_legacy_and_history_profile_load_strictly`。断言：
  ```python
  assert raw.shape == (20,8)
  assert np.max(abs(raw[:,:7]-source_raw[:,:7])) <= 1e-6
  assert np.max(abs(raw[:,7]-source_raw[:,7])) <= 1e-7
  assert first_projection_grad > 0 and second_gru_grad > 0
  ```
- [x] Run `.venv/bin/python -m pytest -q tests/test_history_policy.py`先RED；实现单层GRU(16输入、128隐藏)、时间比例0.1、零输出投影，并在先验和后验的当前状态条件各加入同一历史残差。全链路显式传参，不用临时forward hook或模型全局缓存传历史。
- [x] 只从当前`qpos[:,:8]`做物理读出，当前RGB编码一次；新初始化仅允许精确新增历史模块键，其余严格加载。B缺历史应报错；A与旧调用保持兼容。worker内部可用明确的合成五帧张量预热计算，但不把它登记为相机观测或放入历史；旧stream入口若无历史必须在打开相机前拒绝B。
- [x] GREEN并运行原`test_visual_policy.py`、`test_visual_training.py`及动作合同测试；记录正常CVAE/几何/七轴/夹爪梯度、两步历史梯度、视觉冻结范围与初始化物理等价证据。

## Task 4 共同历史推理、确认重置与新帧等待

**Files:** 修改`executor/visual_inference.py`、`executor/visual_control.py`、`executor/visual_dispatch.py`、`simulation/supervised_rollout.py`；创建`tests/test_history_inference.py`、`tests/test_history_runtime.py`。

**Interfaces:** `await_fresh_observation(capture,*,after_s:float,deadline_s:float,clock,health,spin)->dict`仅取得合格观测；worker持有Task1 buffer及命令收据日志，`confirmation_received(now_s)`清空历史并推进epoch。新packet在原输入哈希中增加历史数据、各帧/收据标识和profile；dispatcher在碰撞检查前后重新验证绑定。仿真`runtime_profile='history_common_v1'`给A/B相同预热/等待，B额外读取历史张量。

- [x] 新增`test_wait_100ms_checks_health_and_never_reuses_prediction`、`test_confirmation_discards_old_history_and_pending_prefix`、`test_packet_detects_mutated_past_frame`、`test_A_B_share_warmup_and_submission_receipts`、`test_delayed_exposure_is_actual_old_physics`。使用假时钟断言：
  ```python
  assert elapsed_s <= .100 and health_calls > 0
  assert newest_exposure > max(last_submission_s,confirmation_s)
  assert model_calls_before_five_valid_frames == 0
  assert submitted_rows.shape[0] in (1,2)
  ```
- [x] Run `.venv/bin/python -m pytest -q tests/test_history_inference.py tests/test_history_runtime.py`观察RED；接入Task1/3，不更改原ACK/尾部/最新帧门控。只在前缀完成或无待执行点的保持期纳入观测；确认后采集5项新观测。等待过程零预测、零旧块复用，超过100毫秒停止；无效/乱序不是可无限等待的“预热”。
- [x] 仿真仍按真实曝光快照配对旧物理状态，不能只调年龄；接触/持杯监测在保持/等待期间照常更新并检查。Task2旧重现profile不受共同新流程影响。保存原1/2点、四模拟确认与新鲜度回归结果。
- [x] 专项及原`test_visual_{dispatch,inference,evaluation}.py`、`test_supported_attachment.py`全部GREEN；两组共同运行配置在训练前冻结。

## Task 5 带父谱系的恢复与五帧图像数据

**Files:** 创建`training/history_recoveries.py`、`training/history_data.py`、`training/history_dataset.py`、`tests/test_history_data.py`；复用`training/contact_sampling.py`、`training/visual_dataset.py`、`simulation/collect.py`，旧`training/recoveries.py`规则保留。

**Interfaces:** `validate_history_manifest(manifest:dict,split:dict)->dict`；`prepare_history_data(source:Path,recoveries:Path,output:Path)->dict`；`HistoryReferenceDataset.__getitem__(index)->dict`返回`image,current,history,actions,padding,sample_id`，其中history为Task3张量字典，sample_id为`(episode_id,start_ts,view_id)`。manifest保留父场景、split、来源类型、完整/部分角色、历史索引、标签/图像/快照/收据哈希；样本索引在A/B之间完全相同。

- [x] 新增`test_sparse_images_are_rerendered_not_repeated`、`test_parent_recovery_cannot_cross_split`、`test_failed_actions_are_context_not_labels`、`test_A_B_use_identical_current_samples`、`test_statistics_use_only_37_training_groups`。明确：
  ```python
  assert train_parent_groups.isdisjoint(val_parent_groups)
  assert len(train_groups)==37 and len(val_groups)==12
  assert A_sample_ids == B_sample_ids
  np.testing.assert_array_equal(derived_base_actions,original_base_actions)
  ```
- [x] Run `.venv/bin/python -m pytest -q tests/test_history_data.py`先RED；通过Task1构造50/100毫秒真实观测视图，补渲染每个所需历史时刻，不拿同一旧RGB复制五份。明确保存收据与曝光关系，取同样有效交集；无效中心不得只从B排除。复用原去重采样，不能因为历史需要静止帧就把全部等待帧当监督中心。
- [x] 实现离线CLI `python -m training.history_recoveries --base-data PATH --captures PATH --output NEW_DIR`：四角色各3训练/1验证，另两个名义开发训练起点；优先ACT失败，无相应失败则标明教师中断来源。教师从Task2精确物理状态生成剩余流程，原失败动作只进入历史上下文；核验物理剩余全流程、`verify_replay`误差≤1e-7及原合同。保留每次生成失败，不以换组名绕过谱系校验。
- [x] GREEN后运行恢复生成及`python -m training.history_data --source PATH --recoveries PATH --output NEW_DIR`；完整48条和18条部分恢复分开计数，必须全部核验通过才开始训练。输出`data_acceptance.json`、冻结共同样本清单/归一化和批次顺序清单；单独报告图像数，不能沿用旧20721数冒充新五帧覆盖。

## Task 6 配对训练、两类预算与严格工件

**Files:** 创建`training/history.py`、`config/history_experiment.json`、`evaluation/history_protocol.json`、`tests/test_history_training.py`；修改`training/integrity.py`、`executor/bundle.py`、`training/export.py`的明确新schema分支。

**Interfaces:** `train_history(data_dir:Path,initializer:Path,config:dict,output:Path)->dict`；CLI `python -m training.history --data-dir PATH --initializer PATH --config PATH --variant A|B --stage pilot|formal --output NEW_DIR`。配置预算为`{kind:updates,limit:1000}`或`{kind:epochs,limit:20}`，pilot不伪填20轮。`require_finalized(checkpoint,config,*,allowed_budgets=('epochs',))->Path`默认仍拒绝pilot，只有离线机制调用显式允许`updates`。

在`training/history.py`提供`freeze_history_protocol(source:Path,runtime_contract:dict,output:Path)->dict`，保留全部数值场景、原源哈希和共同新运行合同。Task8实现只消费封存输出的接口，实际评估在Task11。

- [x] 新增`test_budget_999_cannot_finalize_1000`、`test_pilot_is_not_twenty_epochs_or_deployable`、`test_same_batches_and_initial_physical_function`、`test_full_ACT_groups_have_gradients`、`test_validation_and_test_data_do_not_select_pilot_checkpoint`。断言：
  ```python
  assert A_batch_ids == B_batch_ids and actual_updates == 1000
  assert pilot_summary['budget_kind']=='updates'
  assert pilot_summary['hardware_deployment_allowed'] is False
  assert A_stats_sha256 == B_stats_sha256
  ```
- [x] Run `.venv/bin/python -m pytest -q tests/test_history_training.py`先RED；从冻结15维检查点及其原物理/几何统计直接重配到共同新统计，不走旧9→15扩展。按Global Constraints建立优化器，历史初始化隔离RNG、DataLoader使用独立固定顺序；后验、先验、七轴及夹爪都接收同一history，早期视觉层保持冻结。
- [x] 记录真实更新计数、样本数、批次ID、梯度、权重和归一化哈希。仅完整预算且所有输入未变才finalize；pilot只保存预定1000步末检查点，formal每轮保存并按零潜变量验证损失选最佳。原epochs完整性分支保持，不允许新分支让历史未完成训练导出。
- [x] 新增并运行`test_protocol_frozen_before_optimizer_includes_history_runtime_and_assets`，先确认缺少封存会拒绝优化，再实现封存检查并GREEN。在Task9、任何pilot优化前，将基础源`evaluation/protocol.json`解析到的未见16项协议复制为独立新版本，保存源哈希；完整保留几何、全部数值扰动、query_stride/曝光延迟，仅增加共同预热/等待合同及全部运行依赖。协议不执行，作为A/B训练受保护输入；缺少或变更时拒绝开始/完成训练。
- [x] 专项和旧`test_visual_training.py`、完整性相关回归GREEN；正式与pilot都使用新目录、异常保留失败状态且不出成功终态。此项完成仅表示训练器可用，不运行20轮。

## Task 7 小对照判定器

**Files:** 创建`simulation/history_mechanism.py`、`tests/test_history_mechanism.py`；后续运行产物在`training/pilot_A`、`training/pilot_B`、`results/mechanism_{A,B}`。

**Interfaces:** `mechanism_passed(result:dict,role:str)->bool`；CLI `python -m simulation.history_mechanism --training-run PATH --captures PATH --output NEW_DIR`显式只允许已完成updates预算，复用Task2/4运行器恢复，不生成教师替代动作。`pilot_gate(A:dict,B:dict)->str`仅返回`formal_pair`、`A_development_only`或`stop`。

- [x] 新增`test_20mm_is_mechanism_not_full_task`、`test_all_heldout_roles_required_for_gate`、`test_B_failure_never_launches_formal_pair`。断言：
  ```python
  assert mechanism_passed(lift_20mm_in_400_points,'hold_to_lift')
  assert not full_task_passed(lift_20mm_in_400_points)
  assert pilot_gate(A_pass,B_fail)=='A_development_only'
  assert pilot_gate(A_fail,B_fail)=='stop'
  ```
- [x] Run `.venv/bin/python -m pytest -q tests/test_history_mechanism.py`先RED；按设计第7节实现门槛：两个已知开发起点与独立验证抬升恢复均需稳定夹持并在新增≤400点内离桌≥20毫米，原3900绝对预算仍适用；其余三验证恢复角色需完成各自剩余流程。拒绝假接触、错误阶段及只拟合标签的“成功”。
- [x] 专项GREEN，冻结小对照判定器；本项不优化，待Task9源码封存后执行Task10。

## Task 8 完整任务评估器与候选选择

**Files:** 修改`simulation/visual_evaluation.py`、`simulation/visual_suite.py`、`executor/evaluation_integrity.py`；创建`tests/test_history_evaluation.py`；消费Task9将封存的`evaluation/history_protocol.json`，现有旧协议只作只读来源。

**Interfaces:** `select_independent_candidate(records:list[dict])->str|None`：仅完整开发16/16通过者入围，验证损失更低者优先，差≤1e-8选A。评估与suite均新增`--protocol PATH`参数（默认保留各自旧路径），suite向每个子进程转发同一路径并核验内容哈希。正式评估为`python -m simulation.visual_evaluation --training-run PATH --protocol evaluation/history_protocol.json --output NEW_DIR [--nominal-only|--development-regression]`；独立套件为`python -m simulation.visual_suite --training-run PATH --protocol evaluation/history_protocol.json --output NEW_DIR --workers 1`，最多允许4进程。开发归档沿用原数值场景及明确development角色，不用新16项替代旧开发。

- [x] 新增`test_candidate_selection_never_uses_independent_results`、`test_explicit_protocol_is_forwarded_and_hash_bound`、`test_only_16_unique_finalized_children_can_aggregate`、`test_pilot_20mm_and_partial_events_cannot_pass_full`：
  ```python
  assert select_independent_candidate(no_16_of_16_candidate) is None
  assert not full_task_passed(partial_grasp_or_20mm_result)
  assert all(child_dependencies==parent_dependencies for child_dependencies in children)
  ```
- [x] Run `.venv/bin/python -m pytest -q tests/test_history_evaluation.py`先RED；把历史、控制快照及所有新模块、XML/网格、统计/权重和协议纳入共同依赖，父子清单必须一致。只允许formal预算进入完整候选加载；小对照A单独开发使用明确pilot开发角色，不能冒充完整正式验收或触及独立套件。
- [x] 专项GREEN，保留完整任务判据与旧调用回归；本项不训练或运行独立16项。

## Task 9 离线历史推理与优化前封存

**Files:** 创建`executor/history_shadow.py`、`tests/test_history_shadow.py`；登记新工程`README.md`和`policy_source/docs/README.md`接口。

**Interfaces:** `python -m executor.history_shadow --model PATH --capture PATH --output NEW_DIR`只读取已保存合格历史观测并预测，绝不构造相机/ROS/SDK发送对象。输出`raw20x8`及当前/历史/收据/模型/标定/源哈希，`motion_sent=false`。导出沿用Task6的预算和用途校验，pilot与formal元数据不得混同。

- [x] 新增`test_shadow_never_imports_robot_or_opens_camera`、`test_history_bundle_warmup_does_not_seed_live_history`、`test_pilot_export_cannot_claim_formal_qualification`。Run `.venv/bin/python -m pytest -q tests/test_history_shadow.py`先RED，接通Task1/3/4冻结加载链；旧模型按旧schema、新模型按新schema严格加载。
- [x] 专项GREEN后核验Task1至9所有运行源码、协议、资产及初始化哈希，实际执行Task6的协议封存，输出`evidence/pre_optimization_freeze.json`。源文件此后不再修改；发现需修正时创建新隔离版本并重做受影响实验，不覆盖已封存源或改写旧完整性证明。此时才允许Task10优化。

## Task 10 执行1000步小对照与机制放行

**Files:** 仅新增`training/pilot_{A,B}`、`results/mechanism_{A,B}`和`evidence/pilot_decision.json`，不改冻结源码。

**Interfaces:** 使用Task6训练CLI与预算核验、Task7机制CLI与`pilot_gate`，读取Task9封存清单。

- [x] GREEN后顺序执行Task6 CLI的pilot A/B，各1000更新并核验完整性/批次一致，再执行机制CLI。保存首2点、原始预测、实际提交、夹持差值、杯高、接触时长、停止原因和推理/采集时延。B的历史遮蔽/顺序干预仅在隔离实验分支记录，不能送入真实合格观测流。
- [x] 保存`evidence/pilot_decision.json`，严格按gate继续：`stop`则不加训练预算，报告失败并讨论；仅A通过时停止扩大B，A进入原场景完整开发且不得宣称历史有效；B通过才允许Task11正式双组。新独立16项仍不得调用。

## Task 11 正式训练、完整开发与独立16项

**Files:** 仅新增`training/formal_{A,B}`、`results/nominal_{A,B}`、`results/development_{A,B}`、`results/independent_selected`及`evidence/candidate_selection.json`，不改冻结源码。

**Interfaces:** 使用Task6正式训练、Task8评估CLI及`select_independent_candidate`，消费Task10实际gate；Task7的小机制结果不能替代完整任务。

- [ ] `formal_pair`时从原共同初始化重新训练A/B各20轮，不继承pilot。完整性通过后，逐组评估原场景；通过者才运行旧16项开发。每组只使用预定最佳验证轮次，不因开发失败试其他轮。若仅A pilot通过，只运行A的原场景开发并报告，未另批准正式训练前不扩预算。
- [ ] 完整开发完成后按预定选择函数冻结一个候选，再运行新独立16项；无开发16/16候选则不运行。采用全任务原判据，不用动画、GRASP事件、教师恢复或低损失冒充成功；独立结果若用于诊断，后续明确改作开发。并行仿真最多4个进程，模型训练仍顺序，异常不自动重跑。

## Task 12 最终验证与过程文档

**Files:** 只更新新工程`README.md`、现有全过程/问题文档和`evidence/final_verification.json`；不改封存接口文档或运行源码。

**Interfaces:** 核对Task9源封存、Task10机制gate及Task11按条件生成的训练/评估工件；条件未满足的阶段明确标为未执行。

- [x] GREEN后执行全量`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nero_history_mpl .venv/bin/python -m pytest -q -p no:cacheprovider`及所有新增CLI `--help`。只有新变更或新失败才重复已通过检查；生成`evidence/final_verification.json`核对所有输入/输出、旧源未变和实际预算。
- [x] 以训练/机制/正式开发/独立评估/只读数据验证/真实shadow/真机分别记录状态。真实shadow、实测几何及真机未做就保持未做；本轮名义仿真不能令`measured_geometry/live_frontend_qualified`变true。记录所有失败，独立报告历史是否比A有收益。
- [x] 按executing-plans要求完成一次最终独立代码审查；如需源码修正，另建新隔离版本并运行针对性回归及受影响实验，不修改本版封存源；不启用并行实现、不让审查运行训练或真机。更新过程文档中的实际模型/结果路径，给出完整可复现离线命令和当前限制。

## 执行前审阅与停止条件

顺序依赖：Task1→2→3→4→5→6→7→8→9→10→11→12。Task6实现封存与入口约束，Task9在所有源码完成后实际封存，Task10才开始优化；不允许训练后再修改冻结评估器。

本计划等待用户确认是否符合目标，然后沿用顺序实施。未完成数据/恢复/合同校验不训练；机制失败不堆20轮；完整开发未通过不消耗新16项；不因仿真结果跳过真实几何和观测验收。


## 2026-10-20 实际执行结果

Task1–10、12已完成；Task11条件不成立，未执行。A/B各1000更新、六项机制均2/6，`gate=stop`。未进入正式20轮、完整开发、新独立16项或真机。373项测试通过/5项跳过，源码封存保持。详见[实验总结](../../../nero_act_history_feedback_20261019/evidence/pilot_results.md)及[核验](../../../nero_act_history_feedback_20261019/evidence/final_verification.json)。
