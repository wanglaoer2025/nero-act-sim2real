# 仿真 ACT → 真机动作响应测试

本轮目标是看到仿真训练的ACT产生真实机械臂运动，不要求抓起或放下杯子，也不要求重复J1探针或来回传文件。

## 做什么

现场ZED图像与真实七轴/夹爪反馈 → 配套8维仿真ACT v5（epoch65）→ 取预测前10步中的第10步绝对关节目标 → 保持七轴位移方向、整体缩放到最大轴0.005rad → 现场MoveIt规划/检查 → 操作员确认 → 一次低速真实运动 → 对照模型输出与实际反馈验收。

不是固定J1正弦动作；模型输出变化会改变目标。没有“模型预测不动时改跑固定动作”的兜底。夹爪预测只记录、不执行。没有自动回位。至少4秒运动，加1秒反馈观察，避免旧0.5秒窗口问题。

这里明确使用8维v5，不暗换成v9/v21：本轮只验证模型到真机链路，8维输入不需要当前缺失的杯子坐标或历史夹爪目标。权重、统计和源码已经配齐。原模型hardware_deployment_allowed=false保留，原始绝对目标不执行；这个独立入口只执行新的、现场确认的受限方向响应实验。示范范围之外的关节会写入prediction.json与report.json，不扩大训练范围，也不声称该姿态的策略动作正确。

本测试不以抓取成功、TCP接触精度或精确杯子坐标为前置条件。它也不证明原仿真轨迹/2Hz查询与20Hz闭环已经迁移；这是一次真实输入、单次推理、慢速受限动作。最大单轴约0.286度，肉眼可能不明显，实际反馈决定是否产生运动。

## 在4090现场执行

将压缩包和sha256文件放在同一目录，保持已有通过J1测试的ROS驱动/MoveIt链路，不运行直接SDK控制程序：

```bash
sha256sum -c nero_act_response_20261016.tar.gz.sha256
tar -xzf nero_act_response_20261016.tar.gz
cd nero_act_response_20261016
bash run.sh --execute
```

run.sh只加载现有ROS Jazzy工作空间，默认ROS_DOMAIN_ID=42，不启动第二份驱动，不改使能、速度或gate参数。现场驱动须已经按已验证流程处于control_enabled=true、0<speed_percent<=5，MoveIt与真实反馈同步；不满足时会在发目标前给出本地错误，不能通过发送旧模拟姿态来同步。

推理/相机子进程默认使用 `/home/user/nero/bin/python`，ROS主进程使用 `/usr/bin/python3`。如果现场ACT环境是另一路径：

```bash
bash run.sh --execute --policy-python /实际ACT环境/bin/python
```

先确认该环境已能导入torch、pyzed、cv2、numpy、h5py。可只检查导入，不连接相机或机械臂：

```bash
/home/user/nero/bin/python policy_worker.py check
```

预览模式也会读取ROS与相机，但不下发动作：

```bash
bash run.sh
```

预览和执行都重新采集、重新推理、重新规划；不会拿example_capture或上次预测代替现场输入。程序打印原始ACT第10步目标、缩放系数和实际规划目标；仅显式--execute并输入大写ACT_RESPONSE才会运动。确认时ROS继续接收反馈，避免旧脚本输入等待造成记录空窗。

## 现场边界

空夹爪悬空，整臂小幅扫掠范围清空，ROS驱动是唯一CAN控制者。现场可操作实体急停。规划检查使用当前MoveIt场景，不能发现未建模桌面/人员；操作者仍须确认空中路径。无须重新测杯子坐标。

实际反馈冻结、超时、夹爪漂移、轨迹越界或运动超时将失败并请求取消。软件取消不保证CAN末条目标立即停止，异常应使用现场实体急停。不会自动失能或回位；结束保持终点。不要在ROS驱动运行时启动SDK响应探针。

## 怎么判断成功

输出自动保存到logs/act_时间戳目录：

- camera_0_upright.png、rgb.npy：本次真实图像，旋转180度一次。
- input_state.json：本次真实反馈及主机时间。图像/状态采用静止姿态约束，不声称硬件同步。
- raw_act_chunk.npy、prediction.json：真实模型预测100步与前10步、模型SHA256、示范范围结果。
- trajectory.json、report.json中的projection：从模型预测到缩放目标、到实际轨迹的对应关系。
- actual_feedback.json：真实反馈；report.json记录目标提交时间与记录索引，区分确认前后。

执行模式成功须同时满足MoveIt动作成功、实际反馈沿模型主轴方向移动至少0.0035rad、最后5次反馈七轴误差≤0.0012rad、夹爪漂移≤1mm、全过程没有越出小幅走廊。Mock控制器返回成功但真机不动会失败。预览success=true只表示预览完成，execution_goal_submitted仍是false。

可直接在现场查看report.json，不要求把资料回传后才能继续判断。prediction显示示范范围不匹配是已知实验限制，不代表模型已掌握当前杯子任务。下一阶段才处理任务行为和抓取。

## 本机已验证

13项离线回归通过，包括预测改变→目标改变、静止反馈失败、负方向跟随、规划绕行拒绝、夹爪漂移拒绝，以及ROS Humble实际生成消息的规划构造与碰撞拒绝。已有真实记录的RGB/反馈已跑通v5模型推理并生成受限响应示例，见example_capture/offline_projection.json。

本机没有CAN/现场ROS或ZED，没有声称已在4090执行；现场为Jazzy，其运行时联调尚待本次实际运行。源码、权重部署标记和原始证据没有被改成“已验收”。
