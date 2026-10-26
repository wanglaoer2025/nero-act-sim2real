# Nero：4090 MoveIt 实机链路测试

本轮按用户“直接用 MoveIt 规划开始实机迁移测试”的要求切换测试路线。ACT 补训及其未通过的验收记录保留，权重的真机限制保持。这个测试验证 ROS 规划到实机的链路，不能记为 ACT 策略迁移成功。

现场包：`/home/wq/sim2real/nero_moveit_real_test_20261015.tar.gz`。本机没有 4090 连接，操作者在 4090 旁运行。脚本不导入机器人 SDK、不打开 CAN、不发布关节状态、不改参数或调用使能服务。

## 第一步：复制并加载现场环境

将上述压缩包复制到 4090 的用户主目录。在 4090 终端执行：

```bash
tar -xzf ~/nero_moveit_real_test_20261015.tar.gz -C ~
cd ~/nero_moveit_real_test_20261015
source /opt/ros/jazzy/setup.bash
source /home/user/projects/nero_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=42
```

使用 ROS 对应的系统 `/usr/bin/python3`。不要用 5070 的 MuJoCo 虚拟环境。现场工作空间路径依据已有 4090 清单；若工作空间已搬迁，source 当前的 `install/setup.bash`。

沿用现场已验证的 ROS 启动方式。不要同时运行直接 SDK 的 ACT、响应探针或视觉抓取程序；同一 CAN 只能有一个控制端。不要重复启动一份 CAN 驱动，也不要运行本机仿真用的 `nero_moveit.launch.py`。后者发布 MuJoCo 状态且关闭轨迹执行。

## 第二步：读取真实起点并规划

```bash
/usr/bin/python3 nero_moveit_real_test.py --plan --output logs/plan_01
```

输出目录必须是新目录；重复测试换为 `plan_02` 等。它读取 `/feedback/joint_states`，解析现场 URDF/SRDF 选取恰含 joint1–joint7 的规划组，不硬编码仿真的组名或 TCP。若存在多个候选，报告会列出名称；用 `--group 名称` 选择。

规划目标是**本次实际关节反馈的 J1 加 0.005 rad（约 0.286°）**，其余六轴保持，夹爪不下发。调用 `/plan_kinematic_path`，检查全部轨迹点与插值状态，限制整条路径的关节范围，再将轨迹放慢到至少 4 秒。规划轨迹发布到 `/display_planned_path`，已有 RViz 可查看；这一步没有执行动作。

查看 `logs/plan_01/report.json`：

- `plan_complete=true`：本次小幅轨迹已规划并检查；不表示实机已运动。
- `execution_readiness_checks_passed=true`：当前接口检查通过；执行前仍重新检查并要求现场确认。
- `execution_blocker`：明确的执行阻塞。`success=true` 仅代表本次所选模式完成，不等于允许抓杯。

保存的 `robot.urdf`、`robot.srdf`、`planning_scene.json`、`trajectory.json` 和 `actual_feedback.json` 供复核。碰撞检查只针对现场 MoveIt 当前维护的场景，不能自动发现未加入模型的桌面、杯子或人员。现场须确认这是空夹爪、整臂周围留有余量的小幅空中运动。

## 第三步：执行一次小幅空中运动

仅在上一步无执行阻塞、现场 ROS 已按既有流程使能并同步后执行：

```bash
/usr/bin/python3 nero_moveit_real_test.py --execute --output logs/air_01
```

它重新读取、重新规划，不直接执行旧 JSON。驱动必须是真实 Nero/原厂夹爪、`control_enabled=true`、`0 < speed_percent <= 5`，MoveIt 允许执行。真实反馈、MoveIt 当前状态和控制桥输入须对齐，控制桥只能有预期的发布者。不能有其他 ROS 运动发布者或正在执行的轨迹。

操作者确认空中路径、单一 CAN 控制权和实体急停可用后，在终端输入大写 `MOVEIT`。脚本经 `/execute_trajectory` 只发送机械臂轨迹；不向夹爪下发目标、不自动返回、不启动杯子接触。

反馈超时/时间戳冻结、非目标轴位移、夹爪漂移或运动超时会使测试失败并请求取消动作。**软件取消不能保证 CAN 末条目标立即停止**；现场异常应使用实体急停。脚本不会自动失能机械臂，避免关节失去支撑。

成功须同时满足 MoveIt 动作成功和实际反馈：J1 实际正向移动至少 0.0035 rad、最终最大关节误差不超过 0.0012 rad、夹爪漂移不超过 1 mm，且过程位移满足限制。`GenericSystem` 返回成功而真机未动会被拒绝。

## 现场仍为只读驱动或缺少 MoveIt 时

最近返回资料的驱动是 `auto_enable=false, control_enabled=false, speed_percent=0`。这不是运动模式，不能因为已启动 ROS 就认为可以执行。旧快照的组合 launch 使用 `speed_percent=75, auto_control_gate=true`，本次不沿用该启动命令。

现场的 `control_gate` 是单独的同步桥；本机尚无它当前安装的实现。无法仅凭参数清单确定应如何把 GenericSystem 同步到真实起点后再打开控制。脚本因此不猜测激活顺序，也不通过给硬件发送旧模拟姿态来“同步”。

如果报告显示缺少服务、控制未打开、起点不同步或速度配置不符合要求，在同一终端执行只读采集：

```bash
/usr/bin/python3 collect_nero_moveit_stack.py --output logs/stack_01
```

它读取 ROS 图和参数，并复制已知工作空间内的驱动、gate、launch 和控制器相关源码（限文件类型/体积）。不启动、使能或发动作。把 `logs/plan_01`（若有则同时带 `logs/air_01`）和 `logs/stack_01` 目录带回 5070，即可按真实实现修复激活/同步链路，无需重复采集旧的相机、杯子和标定资料。

## 杯子接触的下一关

本次 RGB 记录只提供杯子/蓝卡的图像位置。射线与临时桌面求交得到的候选坐标尚未核实，不能下发；真实 TCP 与夹爪中心的定义也仍需统一。小幅关节测试可以独立推进，但它不验证这些接触条件。

抓杯前应把核实后的桌面、杯子、目标和 TCP 配到现场 MoveIt 场景，先做空夹爪预抓取悬停，再逐段接近、闭爪、抬升和放置。当前包有意只提供有实际反馈验收的第一段空中测试，不携带图像估计抓取坐标。

## 本机验证边界

已通过 8 项离线检查，包括实际 ROS Humble 生成消息的规划请求/返回结构检查；现场为 Jazzy，尚未执行其运行时联调。已检查两入口 `--help`，没有连接现场 ROS、相机或 CAN，也没有真实运动结果。

设计参考：[MoveIt 规划与执行接口及速度缩放](https://moveit.picknik.ai/main/doc/examples/move_group_interface/move_group_interface_tutorial.html)；现场接口依据 `sim2real_inventory` 与 clean 返回记录，不能用厂商最新分支替代现场改造后的 control_gate 实现。
