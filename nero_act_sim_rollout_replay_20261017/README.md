# ACT 仿真轨迹真机带杯抓放

本轮目标是实际夹住黄色空杯、抬升、搬运到蓝卡、落桌、松爪和退出。动作来自冻结 ACT 在 MuJoCo 中完成任务时的执行参考，模式为 `ACT_simulation_trajectory_replay_not_live_visual_policy`；这不是实时视觉 ACT 闭环，也不是 MoveIt 教师轨迹。YOLO 仅核对杯/蓝卡，MoveIt 仅做全机器人、桌面和杯子的碰撞检查。

前一次在线 ACT 在真机输入下持续输出接近张开的夹爪，未形成稳定夹持，已终止并恢复只读。用户随后选择“先完成 ACT 仿真轨迹真机回放”，并明确本轮必须实际搬杯。旧 ACT_DEMO/GRASP 与已终态尝试保留在 `../nero_act_transfer_completion_20261017`，不能用于本目录的新尝试。

## 来源与动作

- 模型：`model/frozen_run11`，checkpoint SHA256 `47e220e102aaede593004678b3941db99287b07a9a1bc3245bd1e16c3246b6f4`，统计、训练配置、场景配置、推理源码均校验。
- `replay/act_rollout.npz` 是前一次最终冻结名义仿真的原始文件副本；实际参考键为 `sent_targets`，不是原始网络输出 `raw_chunks`。
- 2655 点、20Hz、名义132.75秒，J2范围68.09°，J7范围74.43°。包含 ACT 动作及仿真协议的暂停、反馈保持和完成保持，各来源数量写入 `replay/manifest.json`，不把保持点称为原始网络动作。
- 六段：接近800点、闭爪191点、抬升361点、搬运与下降610点、释放193点、退出500点。人工确认时间、保护参考接入会增加实际时长。

## 执行保护

单一 ROS/CAN 发送者、转场5%、夹爪命令1N；独立 SDK 观察器封锁全部三个发送入口。保留150ms反馈时效、SDK/ROS一致性、45ms目标接收时限、最终结果250ms尾限、2mrad/50ms关节增量、0.5mm/50ms夹爪增量、1.25N实测力和12mm夹持差保护，检查每段完整路径与中间插值。

原幅度动作不缩成小幅角度测试。每段终点附加不改变位置的控制器保持，让人工确认期间门控持续受反馈/心跳监督，避免伺服偏差超过严格1mrad重开条件导致阶段无法衔接。保持不是新的任务动作；确认180秒超时即停止。最终退出段正常等待控制器成功，其他段记录到达参考终点及保持目标仍有效。

明确记录的保护变换：从当前反馈按原限速接入首参考；检测到稳定夹杯后，将闭爪目标下限限制为接触宽度减9mm，以减少过度夹紧；支持释放时用实测姿态进行零运动参考保持；退出确认后按原限速连接回后续 ACT 参考；刚性附着杯将穿桌时停止于此前的安全点。不能据此绕过接触、碰撞、落桌或限速检查。

必须收到本轮 `ACT_REPLAY` 及急停、空杯、清场、唯一控制者现场确认后，才能切换真机执行驱动。四次接触关键确认分别为 `GRASP`、离桌稳定后 `NEXT`、杯底落桌后 `RELEASE`、独立站稳且两指分离后 `NEXT`。未实测稳定夹持不抬升，未确认支撑不松爪；异常关闭门控并取消轨迹，不重试、不回位、不复位或失能。取消不保证机器人瞬间停止。

默认入口只离线验证：

```bash
/home/user/wq/nero_act_full_pick_place_20261017/.venv/bin/python run.py \
  --model model/frozen_run11 \
  --evaluation evidence/final_frozen_nominal_start_join/summary.json \
  --output evidence/offline_report
```

## 当前证据与边界

- `evidence/unit_tests_sealed.log`：48项通过。
- `evidence/final_frozen_nominal_start_join/summary.json`：最终源码绑定的完整 ACT 名义仿真1/1通过，不代表扰动鲁棒性或真机成功。
- `evidence/replay_physics.json`：受保护轨迹在接触物理中完成抓杯、搬运、支持释放、退出；无约束粘附，无提前释放或非预期碰撞。
- `evidence/replay_collision_preflight/report.json`：隔离ROS域43完整路径及保护夹宽检查通过，无轨迹客户端、禁用执行，CAN计数不变。
- `evidence/isolated_stop_gate.json`：永久停止及SIGINT后ROS关闭确认通过。
- `evidence/replay_transport.json`：真实ROS动作通信、接触后保护替换、人工暂停后的严格门控衔接通过；3mrad合成偏差未放宽1mrad门限。是假控制器，不能作为真机接触证据。测试进程最终另行清理。
- `evidence/package_integrity.json`：冻结632个本地和只读依赖文件。
- `evidence/mock_preparation.json`：仅同步 GenericSystem，CAN367206→367206，已恢复全部发送封锁的只读驱动。
- 以下为尝试01之前的离线基线；后续物理尝试状态见文末。执行前重新拍摄。旧的模型TCP与桌高仍是用户先前接受的近似值，不称为独立计量验收。

真机成功必须以本轮真实 SDK/ROS 动作幅度、夹持状态、现场接触确认及蓝卡上独立稳定观察共同证明；不能以仿真、控制器返回或发送点数替代。

最初的 `ACT_REPLAY` 收据为 `evidence/act_replay_authorization.json`，现随尝试01终态保留，不能再使用。后续用户接受当前约9.63mm偏差的记录如下。

## 用户接受当前偏差后的记录

用户明确“有一点偏差也没关系”，因此保留同一份 ACT 轨迹，以实测杯位约9.63mm偏移建立本轮场景。只调整碰撞/接触场景的杯位，后续相对该场景的5mm稳定核对仍有效；未修改桌高、TCP、关节/速度/力/健康/碰撞限制。`source_offset_acceptance.json` 保留用户原话及授权、模型场景、原始轨迹绑定。对应 `accepted_offset_replay_physics.json` 完成真实接触物理抓放，`accepted_offset_collision/report.json` 全路径检查通过。

尝试01在任何关节或夹爪目标提交前被门控健康时效拦截，已终态，不复用旧授权。CAN367206→367207为本次授权驱动启动帧，不能作为运动证据；随后恢复只读PID1450261，SDK/ROS十次核对通过且CAN保持367207。原源码与637项seal存于 `attempt01_frozen_sources`，原失败报告和日志保留。

启动修复把 YOLO 加载、首帧推理与相机冷初始化移至SDK/ROS接收器创建之前，在所有服务、图像和场景就绪后才启动原150ms运动监督。`startup_receive_only_02/03` 是不通过的同进程合成夹具记录；`startup_receive_only_04` 使用与部署一致的独立ROS进程，真实相机与只读SDK验证通过，90个健康样本，CAN未增加、未打开物理门、无动作目标。不是实机任务成功。

`cold_warm_fixed_unit_tests.log` 53项通过；`cold_warm_fixed_frozen_nominal/summary.json` 最终源码名义仿真1/1通过。`final_readiness_02.json` 记录新尝试就绪状态。旧尝试终态保留，只有新的现场 `ACT_REPLAY` 才启动尝试02，仍保留四次接触确认，不自动重试或回位。


## 尝试02与接收线程修复（2026-10-18）

尝试02使用新的 `act_replay_authorization_02.json`。真机完成约40秒原幅度接近（J2约65°、J7约75°），收到本轮 GRASP 后开始低力闭爪。尚未取得稳定夹持证据即因门控状态接收超时终止，未抬升、搬运或放置。`real_attempt_02/report.json`、`attempt02_terminal.json` 和 `attempt02_frozen_sources` 保留现场失败及旧源码，授权02与尝试02均不能重用。停止锁存、关门和两个目标取消已确认，随后恢复封锁发送的只读驱动。

日志显示 SDK 读取间隔曾达141ms，而门控状态订阅与它共用主执行线程；这支持接收回调被阻塞的判断，原日志没有发布端时间戳，不能声称已直接测得当时发布端的消息延迟。

修复将 `/act_gate_health` 移至独立节点/执行器线程，使用深度1的 best-effort 最新值，并同时校验发布端、接收端单调时钟均不超过原150ms。SDK读取后重新取当前时间、最新状态与对齐参考，未修改真实反馈、心跳、限速、夹持或碰撞门限。停止与取消先执行，随后停止并销毁独立接收节点，最后关闭主ROS上下文。新模块包含在冻结评估源码集合中。

`gate_receiver_unit_tests.log`：55项通过。`gate_receiver_receive_only_05/report.json`：隔离ROS域43的独立进程合成反馈/门控、真实相机和封锁全部发送的真实SDK；20次人为引入140.6～141.2ms读取阻塞，最新消息发布/接收年龄约4ms，停止发布后按原150ms拒绝；无动作目标、门未打开、CAN468895→468895。合成ROS反馈不是实机运动或夹持验证。修复已独立代码复核，无需修改的严重或重要问题。

用户第一次“已恢复”实际仅恢复杯位/夹爪，随后“机械臂已回高位”的只读SDK/ROS证实七轴恢复（最大约0.005°），但夹爪仍约80.9mm，杯子相对已接受位置向蓝卡移动约14.4mm，需在下一次回放前恢复张开与源位。此记录不代表下一次已获物理授权。

复查 `full_restored_03.json` 与 `accepted_scene_live_03_restored/report.json` 仍显示夹爪80.9mm、杯位偏移14.4mm，故未启动新尝试，也未切换执行驱动。`gate_receiver_fixed_frozen_nominal/summary.json` 最终源码绑定名义完整仿真1/1通过；`gate_receiver_unit_tests_final.log` 55项通过；`package_integrity_03.json` 包含644项文件（516项外部只读来源）。当前未满足起点/现场条件，见 `readiness_03_pending_manual.json`。

用户随后确认“暂时无法调整现场”。本阶段只完成接收超时修复及验证；新物理尝试未启动，继续保持只读PID1457412，不把软件就绪或仿真成功记为真机完整搬杯成功。
