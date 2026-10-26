# NERO ACT 完整迁移 v2 执行记录

2026-10-25续行：原200步ACT机制预检已完成，固定反馈下闭爪/保持/释放3项通过；它不是完整物理抓放验收。**150/150条**合格完整示教现已齐备（训练120、开发30；12×10/6×5位置分组），当前版本0失败，合同/空间/保存的完整物理结果已复核，见[完整数据核验](evidence/main_dataset_completion_verified.json)。当前正在执行训练前的全量来源哈希与零更新机制复验，正式更新仍为0；通过后自动进入原定20K。实际阶段以[流程状态](evidence/pipeline_state.json)为准。

两个确切失败条件分别完成2951帧整段物理教师回归；全套软件测试74/74通过。仍不能据此宣称ACT已完成全任务，或全部位置均可观测。正式20K当前为0；12条合格真实示教、真实几何验收和真机动作仍未完成。

后续流程已启动：150条全部合格→原200步checkpoint的完整12位置零更新复验（36项z=0重复查询＋全新初始化、全训练统计的96个位置/阶段梯度样本）→复验全部通过才从头进行原20K正式训练及4个checkpoint各44项开发评估。原有限pilot完成记录保留，不追加200步，不加载pilot作为正式初始化，任一阶段失败即停止。见[阶段说明及续行裁定](../docs/superpowers/plans/2026-10-25-nero-act-mechanism-pilot-admission.md)。

本轮按当前RGB中同一名义杯体的投影关联分裂区域；外接框最优解受遮挡偏置时，使用可见轮廓共同约束。原2.5像素门限、额外目标/裁切/漏检拒绝及蓝卡几何核验保留。旧成功域返回不变，120条来源哈希/合同已重核；新增10条旧版本成功示教同样核对，见[等价证明](evidence/contour_observer_equivalence.json)。这不是实机毫米精度验收：原名义bbox位置读出仍未获得真实误差包络。

此前版本保留：[最初27合格/2失败](data/v2_complete/manifest.json)、[修补后120合格/9失败](data/v2_repaired_01/manifest.json)、[凸包版本0合格/10失败](data/v2_enclosed_missing_01/manifest.json)、[投影版本10合格/1失败](data/v2_projected_missing_01/manifest.json)。失败没有删除或算作成功。

[设计](../docs/superpowers/specs/2026-10-25-nero-act-full-transfer-v2-design.md) · [实施计划](../docs/superpowers/plans/2026-10-25-nero-act-full-transfer-v2.md) · [持久台账](progress.md)

## 已验证内容

- 因果15维观测、前2点提交、停止锁存、发送前时效复查、日志异常和ACK延迟，以及碎片、VDS来源变化、限定pilot与正式模式隔离等74项自动测试通过，记录见[测试日志](evidence/full_suite_contour_final.log)。软件测试不代表物理实测或完整任务性能。
- 同一观察器完整重渲染历史教师3002帧，修正后的新审计3002/3002有效，见[完整可见性结果](evidence/dense_visibility_9000_fixed/result.json)。它证明该仿真轨迹可观测，不证明真实几何误差小于某个值。原失败审计保留。
- 新ACT后验、两层decoder、视觉layer4和增量夹爪头具有实际梯度；其余视觉层冻结。未加载旧ACT动作权重，ImageNet初始化单独绑定。
- 新策略与受保护物理评估入口完成七场景接口运行。未训练策略七场景均因初始关节目标不连续被拒绝；这是接口检查，不能当作完整抓放成绩。见[原始结果](evidence/fresh_model_runtime_smoke_directory_fixed/result.json)。
- RTX4090、CUDA及batch32梯度/显存检查通过。早期CUDA失败来自沙箱设备隔离；无需重装驱动。见[主机资源证据](evidence/resource_host.json)。
- [来源清单](source_manifest.json)保留原始文件及派生副本哈希；65份历史源码和36份关键证据未变化，见[复查](evidence/historical_integrity_recheck.json)。

## 数据和训练规则

[采集位置协议](position_protocol.json)划分12个训练位置、6个开发位置。恢复历史开发回归后，在第一次优化前重新封存[最终20个测试位置](final_position_protocol.json)，排除所有新旧开发位置5毫米邻域；原先未使用的test预留坐标作废，train/dev未变。采集最多4进程并行，每次失败保留，禁止自动重试或悄悄替换位置。每条示教必须同时通过完整物理任务判据和统一观察器合同。

78份旧仿真文件视图中，58份具有可复用的完整物理标签；79份旧真实资料目前没有一份满足新主15维合同，主要缺实际提交动作、回执、曝光配对和标定绑定。[审计结果](evidence/data_audit_corrected.json)保留逐文件原因。文件视图数量不代表独立任务数量。

本次将机制预检和正式数据准入分开：仅显式`mechanism_pilot_data`＋`pilot_only=true`可按90train/30dev、9×10/6×5进行一次200更新检查。正式模式仍要求150条、12/6分组，并拒绝机制专用完成记录。当前120条不会被标成完整150条，也不自动追加预检更新。

正式20K从声明初始化重新开始，保存2K/5K/10K/20K四个候选；每候选评估全部30条开发条件、历史7场景及7条准备门槛，共44条完整物理运行。预检权重不作正式初始化。完整数据及新的逐位置数值复验均通过后才准入。

首次版本阻断：抬升过程中，13529像素杯体旁出现20像素细碎颜色区域，严格连通域规则将其判为多个目标。已在同一第1299帧复现并保存[诊断](evidence/visibility_failure_diagnosis/result.json)和[原图](evidence/visibility_failure_diagnosis/failure_rgb.png)。

3×3连接修正通过了[中心3002帧审计](evidence/dense_visibility_fragment_repair/result.json)及原两个失败条件，但扩展到`[-30,+30]mm`边缘位仍失败。之后内部凸包包围规则通过局部帧，却只把整段失败延后到1455～1495帧。不能用局部回归或中心成功推断全工作区可观测。

此前已复现内部22～28像素区域与杯体不连通；“连通域数量≠物体数量”的问题已进入当前几何关联修正。外接框偏置与轮廓约束诊断见[实际失败帧](evidence/cup_association_second_diagnosis/result.json)。未放大形态学核或放宽拟合门限；接触及真实运动保护保留。

旧成功返回语句和其余观测计算保持原样，并对120条文件进行源哈希、合同及位置核查，记录[等价证据](evidence/observer_domain_extension_equivalence.json)。因此可用只读硬链接复用这120条，不改写RGB、动作或已得到的观测；该结论不覆盖失败输入。

`rebuild_data.py`使用VDS引用原始图像，保存来源哈希；数据集同时监控源文件变化。主集仍为150条，补齐后做不新增优化的完整复验，再按原20K预算推进。新版本出现失败即取消未开始任务，不进行同版本自动重试。

12条合格真实示教尚缺；真实混合适配、40次独立位置验收和真机操作均未完成。现有名义几何仍未获得真实误差包络。本轮新入口均为离线，未创建或启动硬件发送者，也未打开ROS/CAN或发送使能、复位、夹爪及运动指令。vendor内的历史执行器只作为几何与保护依赖保留，不能把本次工具测试当作其硬件认证。

## 查看与复现

在本目录运行自动测试：

```bash
.venv/bin/python -m pytest -q
```

默认训练入口仅检查数据与环境，不执行优化；输出目录必须不存在：

```bash
.venv/bin/python train.py --manifest data/v2_contour_01/manifest.json --output evidence/manual_preflight
```

进行GPU训练或EGL渲染时须使用已验证可见GPU的执行上下文。原自动链的失败记录保留。当前补采日志见[采集输出](evidence/contour_remaining_collection.log)，后续零更新复验/20K条件流程见[阶段日志](evidence/contour_stage_pipeline.log)。原200步启动、输出及完成记录仍在`training/pilot/`与`evidence/mechanism_pilot_*`。

`pipeline.py`必须先有代码审查及修正验证记录，才会等待完整数据，核对独立零更新复验及其来源绑定，再条件执行正式训练；任何数据失败、预检失败、实现哈希变化或旧未完成目录都阻止继续，不自动重试。数据清单中的未训练字段表示其生成时刻，后续训练进度以`training/pilot/status.json`、`training/formal/status.json`及动态流程状态为准。
