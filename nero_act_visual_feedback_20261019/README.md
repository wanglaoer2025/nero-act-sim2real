# NERO 15维视觉条件ACT离线工具与正式训练

2026年10月19日，用户批准[设计](../docs/superpowers/specs/2026-10-19-nero-act-live-visual-feedback-design.md)与[顺序实施计划](../docs/superpowers/plans/2026-10-19-nero-act-live-visual-feedback.md)，并确认“尚无新标定，先完成离线工具”。本目录保存这一阶段的代码、测试与证据；旧模型和真机回放包保持封存。

当前已具备标定求解、同帧几何与时间配对、15维ACT模型接口、因果数据加载、正式训练入口和短前缀协议。用户随后要求“按照现在有的标定版本启动正式训练”，因此已允许使用旧标定开展离线基线训练。**实测精度、实时视觉前端及完整真机抓放仍未验收。** 新运动入口仍明确拒绝`--execute`。

## 本轮正式训练

使用此前接受的模型TCP `[152,0,-23.5]`毫米、10月14日相机外参、640×360内参和180°图像约定。版本见[训练标定参数](config/retained_nominal_training_calibration.json)及[原文件哈希](evidence/retained_calibration_sources.json)。这些仍是未独立验证的名义几何，未升级为实测标定；训练不授权真机动作。

已有48条完整物理抓放示教全部转换成功，36条训练、12条验证，来源分组不重叠。在实际采样的预动作物理状态重新渲染320×180 RGB，再从每帧颜色轮廓估计杯底和蓝卡位置，减去同帧名义TCP，得到新增6维观测。共16,104个有效输入；动作及物理状态未改写。仿真物体真值仅存作误差诊断，不进入策略观测。名义圆柱/轴对齐蓝卡估计器不具备真实杯把和遮挡资格，不能直接用于现场接近。

正式候选依次运行320×180和160×90，各20轮、seed42、batch32，均从原run27初始化。输入15维、输出20×8；ACT/CVAE、位置投影、七轴输出和非线性夹爪头参与训练，ResNet仅layer4解冻。学习率分别为1e-4/1e-5/3e-4，KL=.001、prior gripper=.005、pose=.05，前10点L1权重5。仅按固定零潜变量验证损失保存最佳检查点，独立16项协议在优化前登记，尚未实施验收。两种分辨率的小开发集测速权重已丢弃，不用于正式初始化。

- [进程与启动命令](evidence/formal_training_process.json)
- [任务总体状态](evidence/formal_job_status.json)
- [320×180实时优化状态](training/visual_retained_v1_320/status.json)
- [正式训练日志](evidence/formal_training.log)
- [数据验收记录](evidence/formal_data_acceptance.json)
- [分辨率开发测试](evidence/resolution_development_benchmark.json)

完整抓放成功率、位置泛化和真实精度要在训练后分别验证；损失收敛不等于已解决下降碰杯。导出前要求全部声明轮次和输入完整性核验完成，硬件许可始终为false。

## 已实现的离线能力

| 内容 | 验证与实际限制 |
| --- | --- |
| TCP求解 | 至少6个不同姿态，拒绝秩亏；独立TCP至少3点。求得法兰坐标下偏移，不假定法兰与link7相同 |
| 相机外参求解 | 至少6个非共面base控制点，rectified内参及正深度检查；独立相机至少9点。记录真实相机序列号、K、分辨率及180°映射 |
| 标定工件 | 绑定原始样本和哈希，加载时重算校验；区分实测验证、精度目标和合成数据，合成工件不可部署 |
| 图像与反馈 | 曝光时间与反馈最多差50ms；拒绝重复帧、未知时钟映射、过期或未来时间。ZED实际曝光映射尚待接入与实测 |
| 几何观测 | 从明确的杯底、杯口中心、杯把和蓝卡语义关键点求解；带注释保存帧可离线诊断。旧YOLO框不自动升级为合格关键点 |
| 15维ACT | 8维实测状态＋1维上一夹爪命令历史＋6维杯底/目标相对夹持中心位置；输出仍20×8，残差读出只用前8维 |
| 训练数据 | 新6维单独归一化；按实际提交时间取历史，拒绝观测前的未来动作标签、旧零超前真实标签和训练/验证来源泄漏 |
| 训练参数与正式入口 | CVAE、位置输入、动作网络、非线性夹爪及视觉layer4有梯度；其余视觉层冻结。已新增正式训练、逐轮零潜变量验证、哈希绑定和提前结束拒绝 |
| 短前缀协议 | 每次最多2个50ms点，只推进已确认提交的历史；拒绝命令前的旧图像、确认前旧块、超时ACK和停止后的再次提交 |

`VisualControlCore`是无ROS/SDK的协议验证器。真实目标完成、取消、接触监督和只读恢复仍需后续ROS适配；不能把协议测试算作实机执行。

## 已有验证证据

- 每项新行为先出现预期失败，再通过专项测试，记录为`evidence/task*_red.log`及`*_green.log`。
- 使用原run27真实权重和3张已保存的真实静态图像，9→15初始化的最大物理预测差为0，未保存新checkpoint。新增6列为零权重，这只证明初始化兼容，不证明已经学会位置泛化。见[真实输入初始化对照](evidence/real_input_initializer_equivalence.json)。
- 独立审查发现四项重要边界问题，已补充失败测试并修正：零值lead参数改写标签、过去命令作为未来标签、命令前图像重复使用、相机元数据不匹配。另将K纳入相机元数据哈希。见[审查结果](evidence/independent_review.md)和[修正测试](evidence/review_fixes_red.log)。
- 完整回归结果见[最终测试日志](evidence/offline_suite_final.log)。3项旧现场授权夹具测试明确跳过，因为未复制旧授权；另外2项沿用原有可选集成跳过。Matplotlib的Axes3D警告来自既有环境，当前工具不使用3D绘图。
- [实施状态](evidence/pipeline_status.json)按阶段分别记录；[持久账本](evidence/progress.md)保存实际取舍和未完成项。

## 使用入口

从本目录运行，复用已有`.venv`。以下命令不连接机器人：

```bash
.venv/bin/python -m calibration.solve --help
.venv/bin/python -m vision.observation --help
.venv/bin/python -m executor.visual_run --help
.venv/bin/python -m pytest -q
```

新现场样本填写后，可运行：

```bash
.venv/bin/python -m calibration.solve --samples recorded_samples.json --output calibration.json
```

[样本模板](config/calibration_samples.example.json)刻意未填实测值，直接使用会被拒绝。不得把合成样本或名义135/152毫米TCP填成新的实测结果。TCP样本格式为`sample_id`和米制`base_from_flange_4x4`；独立验证另有实测`reference_base_xyz_m`。相机求解数据包括`base_points_m`、`pixels_rectified`、`K`和独立样本ID；独立验证使用单独测得的base/相机3D对应点及测量不确定性，不能以重投影残差代替3D验证。

完整样本结构和误差拒绝案例见[标定测试](tests/test_calibration.py)。其中所有合成数据仅用于测试，默认`synthetic=true`、`verified=false`，不构成标定或运动许可。

带独立语义注释的旧图可运行纯几何诊断：

```bash
.venv/bin/python -m vision.observation --readonly --calibration calibration.json --frames-json annotated_frames.json --output observations.json
```

注释每帧包含图像路径、帧号、曝光/接收时间、相机参数哈希和该帧的关键点。该入口不会从旧检测框猜测遮挡或杯把位置，报告始终为离线诊断，不能授权运动。

## 下一阶段前置条件

1. 采集新TCP、相机、桌面、杯身/杯口/杯把、指面及独立验证样本；核对法兰至link7变换和旋转在指尖处的影响。
2. 在新相机配置下取得完整接近、夹持、持物和放置的可见性样本，完成真实语义关键点前端与曝光时钟校核。视角不足先调整并重新标定，不能用上次杯位顶替实时观测。
3. 用实测几何和合格观测器重新采集完整示教，再正式训练、冻结16项完整任务验收和只读shadow。
4. 接入并验证ROS目标完成、取消及只读恢复后，另行取得当时现场运动授权，执行在线ACT完整抓放。

目前没有新的实测数据。现阶段按用户最新要求先做旧标定下的正式离线训练；实测几何采集和真机步骤仍未启动，没有生成新现场授权或修改原保护阈值。最新阶段以[实施状态](evidence/pipeline_status.json)及上述训练状态为准。
