# Nero ACT v9：纠正数据后的记录输入核验包

本轮10条教师纠正示范全部通过；ACT恢复开发任务0/4，新普通独立场景4/8，未达7/8门槛。
此包仅用于采集真实观察及保存文件的shadow推理，不能执行机械臂动作，也不代表默认候选已提升。
模型仍为14维输入、8维绝对动作；恢复尝试开始时需用当前杯子位置重新填写源中心坐标，之后保持固定。
没有自动重定位/重试功能，observations/object_xyz在仿真数据中仅供诊断，未输入策略。
当前仿真桌面Z=0仍是占位值；现场测量后应据实际桌高、杯子位置和相机重建场景，不能直接套仿真坐标。
先按README采集capture_new，并填写real_measurements.json；填写前可先采图像和关节。
此包用于当前真实观察的采集和离线推理，不代表实机抓杯已验收。
不要把此模型直接替换进旧 run_act_real 配置：它的同步执行方式与本次异步仿真不同。

## 1. 在 4090 保存当前图像和反馈

使用原有可导入 rclpy、pyzed、cv2 的环境，已有 ROS 驱动提供 /feedback/joint_states。
在本包根目录运行，输出目录必须不存在：

```bash
python capture_nero_observations.py --seconds 5 --sample-hz 20 --output capture_new
```

只订阅状态和打开相机，不创建 SDK 机器人，也不发送 CAN/运动/夹爪指令。
capture_new 中保存 observations.npz、report.json 和图像。图像已旋转 180 度转成 RGB。

## 2. 用配套模型进行记录数据推理

在已有 ACT 的 Python 环境运行；不必安装另一套 CUDA。无需 MuJoCo：

```bash
cd nero_act_source
python -m src.act.run_act_shadow --checkpoint ../model/policy.ckpt \
  --stats ../model/dataset_stats.pkl --safety-summary ../model/safety_summary.json \
  --capture-npz ../capture_new/observations.npz --output ../shadow_new --device auto
```

初始姿态不在示范覆盖内、非有限预测或安全检查失败会返回非零状态。
即使返回零也只说明输入/推理兼容：没有验证真实闭环、物体接触或任务成功。
此入口没有 --execute 参数。

若 package_manifest.json 标明 state_dim=14，还需在shadow命令中提供
`--task-coordinates-mm SX SY SZ DX DY DZ`：抓取杯子中心与放置杯子中心的基坐标，
单位mm；SX等仅为占位符，必须替换为已标定的实测/定位数值。缺少该参数会拒绝推理。
这不是末端TCP目标，也不能拿仿真坐标代替。state_dim=8的模型不使用此参数。

## 3. 真机动作前需核对的当前事实

- real_task_measurements.json 中的抓取、放置和桌高仍未知，由现场测量填写。
- 官方 SDK 数学 FK 与 URDF 法兰的固定映射已离线验证；安装 SDK 和实体 TCP 仍需核对。
- sdk_urdf_reference_frames.json 显示仿真抓取点在 SDK 法兰约 [152,0,-23.5] mm；
  现有手工 TCP [135,0,0] mm 与其相差约 29 mm，不自动覆盖标定。
- 仿真相机仍为合成临时视角；当前真实图像、内参和几何用于下一次对齐。
- model/safety_summary.json 的限制和统计来自本次训练；continuous_width 已支持米制连续夹爪开度。
- 实际命令响应通过项目原有 test_joint_stream 的空中小幅动作测试记录。
  普通 --dry-run 已修正为完全离线；--read-hardware 和 run_act_real --dry-run 仍会连接并使能。
- SDK 控制与 ROS CAN 驱动只能由一个控制者使用。只读采集阶段保留现有 ROS 反馈链路。

将 capture_new 和 shadow_new 的文件带回仿真电脑核验，保留现场原有源码和标定。
