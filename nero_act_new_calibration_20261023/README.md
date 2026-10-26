# 2026-10-23 新手眼标定接入 ACT 的核对记录

已将今天的新手眼标定导入本目录的 ACT 离线验证配置。没有覆盖外部项目的正式配置，没有启动 ROS/CAN，没有下发机械臂或夹爪动作。本次任务仍为黄色空杯抓取、抬升、搬运到蓝卡、放置、松爪及退出。

**最新进展：用户已放回杯子和蓝卡，20:31 复拍已确认，已按新标定完成位置估计和两组 ACT 离线核对；完整抓放仍未通过。** 18:57 目标缺失的记录保留为历史，当前不再要求用户重新摆放或测量。

## 使用的标定

- 原文件：`/home/user/projects/nero/artifacts/methylene_blue_recalibration_20261023.KRmPgj/camera_to_base.candidate.json`。
- 本地副本：[inputs/candidate_camera_to_base.json](inputs/candidate_camera_to_base.json)，SHA256 为 `b9874ae76258d6e804d40a3e6eca947c63ea0d33dacdd48335817b74ccbc4cb6`。
- 原始样本：`/home/user/projects/nero/calibration/auto_handeye_20261023_161905.json`，本地副本为 [inputs/handeye_samples.json](inputs/handeye_samples.json)。
- 相机序列号：39602891；标志黑框边长：95 毫米。
- 20 组原始样本，15 组有效样本；来源工具排除第 14、15、16、17、19 组（从 1 编号）。
- 按来源工具的“各样本反算相机位姿一致性”定义复算：平移 RMSE **4.585845 毫米**，旋转 RMSE **0.672862°**。这不是独立抓取位置误差上界。

矩阵方向是 `T_base_from_camera`，使用原始 ZED 左目光学坐标系。接入时只将平移从毫米换成米；图像仍旋转 180°，没有再旋转或反向使用外参。适配代码和真实样本回归测试分别为 [calibration_adapter.py](calibration_adapter.py) 和 [test_calibration_adapter.py](test_calibration_adapter.py)。

新旧估计的相机原点相差约 72.93 毫米，旋转相差约 6.36°，因此不能直接沿用旧杯位坐标。这只是两版估计的差异，不能据此断定相机实际移动了多少。法兰到标志板的变换也不等于两指夹持中心 TCP；本次没有将旧名义 TCP 改报为已验证。

## 实际完成的验证

1. 4 项适配回归测试通过：单位和方向、非法矩阵拒绝、原始样本误差复现，以及旧场景/TCP/内参不被意外修改。
2. 固定原杯位、起点、TCP、内参、模型权重及统计量，仅换新外参，运行完整流程评估入口。结果在第 0 个动作前拒绝：`Cup clipped by image boundary`。这说明该组合不能直接运行，不能算完整 ACT 流程通过。详情见 [results/summary.json](results/summary.json)。
3. 18:57:47（北京时间）只读获取当前 ZED 图像，确认相机序列号与原始标定样本一致。采集实际分辨率为 1280×720，使用本帧相机内参进行定位核对，见 [current_camera/report.json](current_camera/report.json)。
4. [当前正向画面](current_camera/upright.png)中，操作桌面上未见黄色任务杯和蓝卡。现有定位器也拒绝输入：`Unique visible cup and target required`。没有用旧目标坐标补齐输入，见 [current_camera/localization_check.json](current_camera/localization_check.json)。

第 2 项是固定世界坐标的外参敏感性试验，保留旧内参作为对照；第 3、4 项则使用当前帧的真实内参。这两种测试的输入和结论不能混用。

## 用户放回目标后的核对

新的 [现场图像](scene_capture_after_return/upright.png) 和 [当前只读反馈](scene_capture_after_return/feedback.json) 已保存。控制器为 `arm_status=0`、`err_code=0`；八次关节反馈稳定。相机与 SDK 反馈分别采集，没有声称硬件同步。

全图颜色定位受到显示器蓝色区域干扰。按原教师工作区 XY（X −650～0 毫米、Y −300～300 毫米）和名义桌面 Z=0，用新标定进行射线投影筛选后，两个显示器蓝色区域均在工作区外，桌上的蓝卡完整保留。已有 YOLO 杯身份检测、黄色轮廓修正及唯一蓝色凸四边形检测估算：

| 项目 | 基座坐标，毫米 |
| --- | --- |
| 杯底中心 | [−516.77, 54.60, 0] |
| 蓝卡目标 | [−524.65, −131.81, 0] |
| 源到目标距离 | 186.57 |

这是采用旧名义杯尺寸和桌面高度的**离线几何诊断**，不代表实时定位工具或独立精度验收通过。筛选只影响几何诊断，未遮罩 ACT 的 RGB 输入，也未改写 ACT 动作。详情见 [定位记录](scene_capture_after_return/workspace_sensor_localization.json)。

当前实测夹爪开度为 99.42 毫米；ACT 的合法目标上限为 95 毫米，因而不能直接把 99.42 毫米当作上一条合法指令初始化。保存帧的编码被拒绝，没有生成 ACT 预测；`structural_contract_error=null` 不表示通过，因为结构检查没有执行。

[本轮完整离线结果](after_return_validation/summary.json) 包含两组：

| 起点 | 结果 |
| --- | --- |
| 完全保留实测状态，包括 99.42 毫米开度 | 在历史回执初始化处拒绝，0 查询、0 动作点 |
| 仅假设夹爪已准备至 95 毫米，其他场景及七轴不变 | 1950 次查询、3900 个 ACT 动作点；无抓取事件、抬升为 0，任务失败 |

第二组只是假设，没有在真机准备夹爪。其模拟控制窗口为 195 秒，最近杯中心的 XY 误差仍有 56.54 毫米，之后偏离目标，详见 [轨迹诊断](after_return_validation/trajectory_diagnostic.json)。`error=null` 表示没有抛出异常，不能算抓放成功；`complete=true` 仅表示评估已结束。

本轮 5 项软件回归测试通过。没有扩大训练预算，没有更新权重，没有启动 ROS/CAN，也没有执行夹爪准备或真机动作。原始实测状态未被静默裁剪到 95 毫米。

另外，在同一新标定、当前位置和 95 毫米假设起点下，已有物理教师完成了接近、下降、闭爪、抬升、搬运、放置、释放及退出：抬升约 194.4 毫米，水平搬运约 186.6 毫米，3002 帧，精确重放最大关节误差为 0，301 个动作窗口通过原动作检查。见 [物理教师诊断](after_return_teacher_diagnostic_egl/report.json)。这表明这套**名义仿真场景**存在完整抓放路径；它不是 ACT 成功或真机精度证明，也没有用作运动替代。生成的 HDF5 标记为 diagnostic，本轮未进入训练。

## 当前边界和后续步骤

新标定在来源文件中仍标为 `candidate_saved_pending_independent_validation`。已接入离线验证，尚未启用真机部署。最新受限修正模型也仍未通过完整 ACT 抓放，详见 [上一轮结果](../nero_act_preservation_scope_20261023/results.json)；更换外参不会自动证明策略能够闭爪、抬升和释放。

目标已放回，当前不需要重复测量或重新摆放。夹爪启动衔接和 ACT 的接近行为仍需修正：95 毫米假设对照已证明，仅处理开度不足以完成任务。后续应针对新标定下的位置条件验证恢复数据和接近行为，再进行完整路径及只读核对。当前没有排队的运动或自动重试任务。

原始复制来源与哈希见 [inputs/provenance.json](inputs/provenance.json)，数值审计见 [integration_report.json](integration_report.json)。
