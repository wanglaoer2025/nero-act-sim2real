# 正式起点只读采集（ACT sim2real）

用户已摆好正式起点，本次只采新状态，不重复测量杯子、相机或标定。仅使用4090已有ROS2 Jazzy/pyzed/numpy/opencv环境，保持机械臂静止，已有反馈节点应发布 /feedback/joint_states。不要为采集启动新的使能/控制程序。ZED应未被其他相机进程独占。

在本包目录运行：

```bash
python capture_nero_observations.py --seconds 5 --sample-hz 20 --output formal_start_capture_20261015
```

输出目录必须不存在。程序仅订阅ROS反馈和读取ZED，不连接机器人SDK、不发布控制话题、不操作夹爪、不发CAN命令；--help同样不连接硬件。若缺失反馈或图像，保留report.json及终端错误，不通过放宽超时或伪造反馈继续。

将输出的整个formal_start_capture_20261015目录带回5070的sim2real目录，至少含observations.npz、report.json、camera_0_upright_rgb.png，并保留终端输出。NPZ含顺序joint1..joint7弧度与gripper米制总开口、已旋转180度的RGB图像、主机接收时间和ROS时间戳；不要再次旋转图像。

此包不加载ACT、不运行抓杯，也不声明真实硬件同步。它不记录上一条夹爪下发命令，不能直接冒充15维ACT完整输入；后续需按执行器启动约定初始化并记录命令历史。这里只核对正式起点和实际相机观察，任何文件验证通过均不代表可发送运动。
