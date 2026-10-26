# 离线阶段独立代码审查

审查范围：本目录实现的离线标定、观测与同步、15维模型/数据接口、优化器探针及纯短前缀协议。审查员只读操作，未接触相机或机器人；独立执行相关41项测试通过。整体计划尚未完成，现场缺少新实测标定。

## 发现与修正

| 级别 | 发现 | 修正与证据 |
| --- | --- | --- |
| Important | `joint_target_lead_steps=0`绕过真值检查，使父数据集把真实提交标签替换为同帧qpos | 拒绝任意非None lead参数；`test_zero_joint_lead_cannot_replace_submitted_labels_with_qpos`先红后绿 |
| Important | 单独数组有序，但标签时间早于观测也能通过 | 要求该行实际目标在观测之后250ms内提交；`test_past_action_labels_cannot_be_used_as_future_supervision`先红后绿 |
| Important | 帧ID增加仍可能使用上一提交之前的曝光 | 记录上一提交时间，新曝光和反馈均必须晚于它；`test_new_frame_id_cannot_hide_exposure_before_previous_submission`先红后绿 |
| Important | 离线观测器只核对尺寸，错误相机元数据仍被标为valid | 观测器也核对相机元数据哈希；`test_observer_refuses_wrong_camera_or_intrinsics_hash`先红后绿 |

修正时一并加入内参K与元数据一致性检查及哈希绑定，`test_camera_metadata_binds_current_intrinsics`先红后绿。五个失败用例记录于`review_fixes_red.log`；数据7项、协议7项、相机/标定/特征15项分别通过，完整回归见`offline_suite_final.log`。无Critical或Minor审查发现。

## 审查未判断项及实施裁定

- 真实计量精度、工作区覆盖、指面/杯把实测几何：尚无原始样本，继续作为实测前置，不能宣布标定达标。
- 在线语义检测、曝光时钟与遮挡性能：前端尚未合格，缺少前端时返回无效观测；不发布实时纠偏能力。
- 正式训练、新数据、冻结完整评估与部署授权：均未启动，不以探针梯度或初始化等价替代。
- ROS传输、实际前缀完成、取消和只读恢复：仍由待实现的适配器负责；纯协议不接触硬件，执行入口明确禁用。
- 完整预测来源绑定：留在后续同帧在线worker/ROS适配中实现，当前纯协议不能独立授权任务动作；本次已修正其自身可复现的帧序缺口。
