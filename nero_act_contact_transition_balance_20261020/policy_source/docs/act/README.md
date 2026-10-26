# ACT 文档索引

本文件是 ACT 子系统的唯一入口。`act-day*-nero-codex-prompt.md` 是开发任务书，可能包含尚未实现或已经被后续实验推翻的建议；实际行为以 `src/act/`、当前 YAML 和 `act-day*-report.md` 为准。

## 离线源码补充包（2026-10-13）

导出的源码已将 `pyAgxArm` 导入延迟到 `create_nero_arm()` 调用时。仅加载
ACT 数据、计算归一化统计、训练或查看训练/推理入口帮助不需要机械臂 SDK，
仍需要相应的 NumPy、HDF5、PyTorch、torchvision、YAML 等软件依赖。
真实机械臂创建仍需 `pyAgxArm`，真实相机采集仍需 ZED SDK/`pyzed`。

从补充包根目录执行 `python -m unittest discover -s tests` 可复测。
`test_act_offline_imports.py` 在隔离子进程内禁止硬件 SDK 导入，并验证 HDF5
写入、八维 state/action 数据加载和训练模块导入。本次源电脑共通过 23 项测试；
仿真电脑必须单独执行，不将本机记录视为远端验证。

`src.act.run_act_real --dry-run` 仍会在确认后连接并使能机械臂，不能作为离线验证。
单相机八维配置使用 `act_nero_day5.yaml`/`act_nero_day6.yaml` 作为接口参考；
示范数据、归一化统计和训练权重需要针对仿真任务重新生成。

## 夹爪命令历史实验（2026-10-15）

可选 `previous_gripper_conditioning=true` 使用15维状态：七关节反馈、实际夹爪总开口、六维静态任务中心、上一周期实际下发的夹爪目标总开口。动作仍为八维绝对目标。初始帧没有历史时，以实测初始宽度表示保持；后续严格读取 `action[t-1,7]`，不使用当前或未来标签。仅在明确的15维配置下启用，旧8/14维默认行为不变。

新增维度的统计只用训练回合生成。仿真电脑的扩维初始化将两处状态投影增加零列，使新特征尚未学习时保留旧模型输出。真实反馈宽度继续输入，不能用目标宽度替代它。

此配置尚未通过闭环验收，也未接入实机执行器。`run_act_shadow` 的反馈型 NPZ 没有命令历史，会明确拒绝此类检查点；不能凭空把静态反馈当作过去的闭爪命令。软件测试通过不代表新特征改善了任务。

## 解析运动学观察特征（2026-10-15）

可选`task_displacement_features`在15维观测之外，给编码器附加抓取/放置中心减去当前TCP的六维位移，尺度由`kinematic_feature_scale_m`指定。TCP由当前输入关节和检查点中的串联模型正运动学计算；观察加噪后重新派生，不使用干净关节、实时杯子真值或教师阶段。动作始终由ACT预测，不调用逆运动学生成动作。

外部观测仍为15维，归一化统计与嵌入配置严格配对；新状态投影增加零列初始化，旧8/14/15维默认行为不变。运动学核函数不依赖SDK/MuJoCo，训练期TCP损失复用同一实现。接口软件通过不代表闭环稳定性通过，所有新权重仍禁止实机执行。

## 阅读顺序

| 阶段 | 实施报告 | 解决的问题 |
| --- | --- | --- |
| Day 1 | [act-day1-report.md](act-day1-report.md) | NERO 七关节/夹爪映射与 5/10/20 Hz 流式控制 |
| Day 2 | [act-day2-report.md](act-day2-report.md) | Leader 示教、双相机帧和 HDF5 schema |
| Day 3 | [act-day3-report.md](act-day3-report.md) | episode 验收、可视化、起点移动和安全 replay |
| Day 4 | [act-day4-report.md](act-day4-report.md) | 官方 ACT 最小适配、shape smoke test 和 overfit |
| Day 5 | [act-day5-report.md](act-day5-report.md) | 正式训练、split、归一化和离线评估 |
| Day 6 | [act-day6-report.md](act-day6-report.md) | 真实相机/反馈闭环、安全层和运行日志 |
| 双相机固定起点流程 | [dual-camera-recording.md](dual-camera-recording.md) | 当前推荐的数据采集、训练和实机验证步骤 |
| Day 7 | [act-day7-nero-codex-prompt.md](act-day7-nero-codex-prompt.md) | 尚未完成；正式 benchmark 的需求草案 |

要理解某个阶段为什么这样实现，再读取同编号的 `act-dayN-nero-codex-prompt.md`。不要直接照任务书中的命令执行实机。

## 当前结论（2026-09-11）

- 20 Hz 相机、推理、关节发送和反馈链路已通过；推理延迟不是策略停滞的核心原因。
- 旧单相机/双相机模型可复现运动，但完整自主抓取未通过任务验收。
- `act_dual_xyz_lead10` 模型能向杯子靠近，随后落入局部小幅摆动且不闭爪。离线 `val_l1` 或运行 `status=passed` 不能代表任务成功。
- 根因包括 episode 开头静止标签冲突、训练起点组合不一致、每轮抽样覆盖不足、三个物体位置不均衡，以及 temporal ensemble 对弱动作的抵消。
- 当前推荐路线是 `act_dual_xyz_fixed`：固定预备姿态、移除/裁剪静止开头、关节前视 8 帧而夹爪保持原时序、遍历全部有效时间窗、三位置等量。
- 新 fixed 模型必须先关闭 temporal ensemble 验证持续推进和夹爪事件；通过后才开启融合。完整根因证据见 [故障排查 11.12](../robot-troubleshooting.md#1112-关节前视对齐后仍在目标附近微摆)。

## 目录对应关系

| 类型 | 位置 | 是否提交 |
| --- | --- | --- |
| 实现 | `src/act/` | 是 |
| 实验配置 | `configs/act_*.yaml` | 是 |
| 自动测试 | `tests/test_act_*.py` | 是 |
| episode 与 manifest | `data/<experiment>/` | 否 |
| checkpoint/statistics | `checkpoints/<experiment>/` | 否 |
| 图表和离线评估 | `artifacts/<experiment>/` | 否 |
| 实机 CSV/summary | `logs/<experiment>/` | 否 |
| 官方 ACT 固定代码 | `third_party/act/` | 是，避免无关修改 |

同一实验的 `<experiment>` 名必须在 data、checkpoint、artifact、log 和训练/推理 YAML 中一致。不要把旧数据统计量或 checkpoint 接到新 action shape、相机列表或对齐配置上。

## 开始下一轮前

2026-10-14 多杯实验补充：NeroActPolicy 新增可选 `joint_residual_anchor`。模型关节输出以当前观察的关节位置为基准学习修正，但对外仍返回归一化的八维绝对位置动作；夹爪仍为连续总开口。配置显式保存七维anchor scale/offset，由训练专用qpos/action统计推导，不从运行目录隐式读取其他统计。旧配置默认关闭，checkpoint参数键保持兼容；不允许与feedback-delta接口混用。此实验由仿真侧DeploymentMeanPolicy使用零潜变量监督，切换初始化时仅清零关节输出头、保留夹爪头；推理加载不会清零已学参数。当前只用于多尺寸/位置与物理扰动仿真实验，不能据此宣称真机泛化通过。

另建14维实验分支：前8维为机器人反馈，后6维为静态抓取/放置杯子中心的base XYZ（m），动作仍8维。它与旧8维模型/统计不混用。仿真从场景任务几何得到中心；真实部署必须由已标定定位或测量提供，不能把它称为无需定位的纯视觉迁移。放置XY仍固定，尚未训练任意放置点。`run_act_shadow`可通过 `--task-coordinates-mm SX SY SZ DX DY DZ`检查记录输入，缺少实测坐标拒绝14维推理；8维模型不接受此参数。SRC/DST均为杯子中心，不是TCP点，必须遵守实际数据的坐标语义。

2026-10-14 sim2real 接口修正：SafetyFilter 新增显式 `gripper_mode: continuous_width`，保留米制总开口；未设置时保持原有二值迟滞。机器人包装器新增 `send_gripper_width`，复用已有夹爪范围、力和连接检查。关节限位、目标步长与相对反馈领先限制同时求交；若不存在共同可行范围，报错停止，避免后一次裁剪破坏前一次限制。相关单元测试和 mock 控制循环已通过；没有发送真实运动。

新增记录数据入口 `python -m src.act.run_act_shadow --help`，读取只读采集的 upright RGB/named feedback NPZ，严格加载配套 checkpoint/statistics、检查初始姿态范围、保存预测块和限幅预览。该入口没有机器人/相机连接、SDK/CAN调用或 `--execute` 参数。它用于输入兼容性核验，不表示真实闭环成功。

当前 `run_act_real` 仍是同步每周期推理入口，不等于仿真侧异步10步前缀执行。未完成调度适配与真实端到端响应验证之前，不将新仿真权重直接替换进旧实机配置。现场任务坐标与桌高已由用户确认未测量；当前手工 TCP 与离线 SDK/URDF 映射后的仿真抓取点约有29 mm差异，不能直接覆盖标定。

2026-10-14离线补充：`src.act.test_joint_stream --dry-run`此前在打印轨迹前构造机器人，因此仿真电脑缺少pyAgxArm时退出。现改为普通dry-run在构造机器人前打印并返回，相关离线测试通过；`--read-hardware`及`--execute`保留原有确认与硬件行为。补丁保存于本机 `/home/wq/sim2real/nero-act-joint-stream-offline.patch`，原始导出源码保留。

1. 先按 `dual-camera-recording.md` 检查三组位置数量相同、固定起点一致、没有录入静止前缀，且每条轨迹包含接近、闭爪、搬运、放置、开爪和撤离。
2. 运行 `validate_episode` 和配置对应的 `dataset_stats`，确认 rejected episode 不进入 split。
3. 训练后先做完整 validation 离线评估，不直接连接机械臂。
4. 实机按 dry-run、单动作、10/50/150 steps、完整 rollout 逐级增加，并同时观察业务阶段，不只看 summary 的 `passed`。
5. 只有完整抓取在固定 benchmark 上重复成功后，才进入 Day 7。

重训前使用 `src.act.grasp_transition_audit` 审查所有 episode 的采集协议、闭爪前
task-relative TCP、双相机关键帧和夹爪时序，并对当前 checkpoint 分别统计 train/val 的
close 检出与时间误差。它只做离线 FK 和模型推理，不连接硬件；详细命令见项目 README。
