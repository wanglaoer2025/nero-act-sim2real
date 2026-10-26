# 当前反馈来源排查结果

当前已完成隔离虚拟同步并恢复只接收，真实SDK、ROS与虚拟桥一致；CAN TX保持240992。详见末节。

以下保留首次排查记录；本轮没有向实体下发机械臂、夹爪或使能操作。

| 来源 | 用途 | 结论 |
| --- | --- | --- |
| `/feedback/joint_states` | 真实驱动反馈 | 发布者只有`agx_arm_ctrl_single_node`；10次与独立SDK交叉核对，最大关节差约0.005° |
| 独立SDK观察器 | 从CAN接收完整健康反馈 | 底层send、单消息、批消息发送均封锁；与真实ROS反馈一致 |
| `/control/joint_states` | GenericSystem虚拟控制器状态 | 仍是旧建模姿态，不能当作当前真机反馈 |

| 关节 | 真实反馈(度) | 虚拟桥(度) |
| --- | ---: | ---: |
| J1 | 10.497 | 22.452 |
| J2 | -7.249 | 34.276 |
| J3 | 5.729 | 0.582 |
| J4 | 90.000 | 86.750 |
| J5 | -1.367 | 12.033 |
| J6 | 16.140 | 19.671 |
| J7 | 7.508 | -33.185 |

CAN TX：240992 → 240992。gated命令发布者：0。这些计数只说明本次采样期间的状态；此前240977→240992的累计15帧差异仍未归因。

结论：当前真反馈来源清楚，与独立接收交叉核验一致；旧虚拟状态没有同步。历史姿态为什么变化仍缺少证据，不能认定为用户手调或其他程序运动，也不能把旧姿态写回真实反馈。

后续以当前真机起点重新建模、训练和评估。只有在发送完全隔离的条件下才同步虚拟控制桥；不向真机发送“追赶旧虚拟姿态”的轨迹。

原始证据：[feedback_source_recheck.json](feedback_source_recheck.json)。诊断入口：[recheck_feedback_source.py](recheck_feedback_source.py)。

再次只读复测见[feedback_source_recheck_later.json](feedback_source_recheck_later.json)：SDK/ROS最大差仍约0.005°，与首次采样姿态最大差约0.004°；CAN TX再次240992→240992，gated发布者0。两次采样之间未观察到实机关节姿态实质变化，不能据此解释更早的历史差异。

当前ROS端点也已核对：`/feedback/joint_states`的唯一发布者仍是真驱动，订阅者为`robot_state_publisher`及`move_group_private_104605484623440`；`/control/joint_states`唯一发布者为`joint_state_broadcaster`，当前订阅者0。详见[真实端点](real_feedback_endpoints_current.txt)、[虚拟端点](virtual_control_endpoints_current.txt)。MoveIt状态监控目前确实读取真反馈。

实际安装的`move_group.launch.py`链接到工作区源码；`follow=true`只将MoveIt的joint_states订阅选择为feedback_topic。`agx_arm.ros2_control.xacro`另行声明GenericSystem及initial_value，没有将真实反馈写入mock内部状态的同步路径。因此follow模式并不自动消除真实与虚拟控制器的差异。此结论解释当前来源分歧，不能归因更早的真实姿态变化。

历史来源边界：旧`scene/config.json`的起点标为`current_capture_first_paired_state_20261016`，与当前虚拟控制值相近；它不是本轮最新真反馈。数值相近不证明旧建模过程一定读错了话题，也不证明两次记录之间实体移动过。当前能确认的是两套话题表示不同对象、真实SDK/ROS一致且稳定；历史采集来源及更早状态变化仍未归因。

## 虚拟控制器已完成隔离同步

已验证旧只接收驱动PID922297及三重发送封锁并停止；同步时ROS真驱动0、gate0、gated发布/订阅0、其他raw控制订阅0，唯一硬件插件为GenericSystem。只把模拟关节同步到独立SDK实测姿态。同步前后实体反馈最大差约0.005°，CAN TX均240992，没有向实体发送指令。

随后恢复只接收驱动PID1384575，三个SDK发送入口继续封锁、auto_enable=false、control_enabled=false、速度参数0。再次10次交叉核对后，真实SDK、真实ROS与虚拟桥最大差约0.005°。`control_enabled`读回False；启动日志的“control is now enabled”在源代码中表示`control_ready`反馈就绪，不能当作发送开关已开启。

证据：[同步记录](virtual_bridge_synchronized.json)、[只接收恢复](readonly_driver_after_mock_sync.json)、[同步后复测](feedback_after_virtual_sync.json)。当前虚实起点不一致的问题已修正；历史建模快照来源差异仍未归因，未据此认定现场调整。
