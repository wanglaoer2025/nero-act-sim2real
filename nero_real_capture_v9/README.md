# Nero 真机迁移实验：只读采集阶段

本包没有运动入口、策略模型或CAN控制代码。请在4090原有ROS/pyzed环境运行，现有驱动提供 /feedback/joint_states。

1. 保存当前真实相机与关节反馈（输出目录必须不存在）：

```bash
python capture_nero_observations.py --seconds 5 --sample-hz 20 --output capture_new
```

相机图像旋转180度并转为RGB；不再重复旋转。得到 observations.npz、report.json、camera_0_upright_rgb.png。
此程序只订阅反馈并打开ZED，不连接SDK机器人，不发送运动、夹爪、使能或CAN指令。
反馈配对依据主机接收时间，不等于硬件时间同步。

2. 将 capture_new 目录带回仿真电脑，保留完整NPZ和报告。此步不需要测量目标坐标。

3. 按现场测量填写 real_measurements.json。source/destination是杯子几何中心的Nero基坐标XYZ，单位mm；
桌面Z也是同一基坐标mm。杯子立在同一平面时，中心Z=桌面Z+杯高/2。不是TCP点，也不能套用仿真坐标。
安装与清单一致，不表示实体TCP或相机独立点标定已经通过；尚未核验的字段保持null。

4. 在有NumPy的Python环境核验保存的文件：

```bash
python audit_nero_real_readiness.py --capture capture_new --measurements real_measurements.json --output readiness_new.json
```

缺少坐标或有效观察时返回1并保存阻塞项；坐标尚未测量仍可先完成步骤1/2。
ready_for_recorded_shadow只表示可进一步检查记录输入，ready_for_real_motion固定为false。
当前v7/v8闭环均未通过自主抓杯验收，不能因只读核验通过而执行策略。
已有14维v7只读模型包可在实测坐标齐全后运行run_act_shadow，真实相机外观与视角还需单独对齐。
