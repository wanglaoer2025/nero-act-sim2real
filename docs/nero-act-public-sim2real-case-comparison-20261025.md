# 公开 ACT 仿真到真机案例与 NERO 差距分析

核查日期：2026-10-25。本文核对作者论文、项目页和公开源码；属于资料与代码研究，未在本机复现外部模型，未训练、未下发机械臂动作。NERO 结论沿用本日已完成的[历史根因审计](../nero_act_root_cause_audit_20261025/README.md)，不把本轮网页查阅算成新的机器人试验。

## 1 结论

**ACT 仿真迁移到真机有公开成功证据，但“网上很多 ACT 演示”不能等同于“仿真训练后完整迁移成功”。本次最明确的参考是 DISCOVERSE；其他项目需要按证据强弱区分。** 未核实到与当前 NERO 七轴、黄色杯、ZED 和夹爪完全一致的独立第三方复现。

我们当前主要缺口不是一个需要继续调大的闭爪权重。系统在仿真内尚不能稳定完成全任务，局部恢复训练又曾损坏原有接近；其上还有视觉有效域、接触几何和实时发送问题。真实图像与仿真渲染存在差异，但无法解释发生在同一仿真环境里的训练退化。

建议借鉴两条已公开的方法：用 DISCOVERSE 的完整随机示教和输入/动作合同建立稳定的仿真基线；用少量合格真机数据适配真实分布。最终部署仍应是“真实当前图像＋反馈→ACT→受保护控制”，目标位置变化时重新观测和推理，而不是每个杯位重新生成一条仿真轨迹。

## 2 哪些案例确实可以参考

| 项目与主源 | 实际证明的事情 | 本次采用的证据等级 |
|---|---|---|
| [DISCOVERSE GitHub](https://github.com/discoverse-dev/DISCOVERSE)、[原论文](https://arxiv.org/html/2507.21981v1#S5) | 仿真数据训练 ACT，零样本部署 AIRBOT 真机，有逐任务试验结果 | 有原论文真机统计和开源实现；未逐项取得绑定论文配置的全部权重及原始试验日志 |
| [OpenSO-101 GitHub](https://github.com/jixinyan/OpenSO-101) | Isaac Lab 示教、官方 LeRobot ACT 训练、同 checkpoint 真机部署入口，README 有 ACT 真机演示 | 有公开代码与作者演示；本轮未观看视频，未找到完整任务成功率与试验分母，不称为已统计验证的完整抓放 |
| [ANR 作者实验](https://www.anrrobot.com/robots/experiments/sim-to-real.html)、[LeRobot fork](https://github.com/anredlich/lerobot)、[MuJoCo 环境 fork](https://github.com/anredlich/gym-aloha) | ACT 仿真迁移能拾起，但未完成双臂转移；仿真预训练后加入真实数据明显加快训练 | 有实际正负结果和代码，适合学习失败及适配方法；不是零样本 ACT 完整转移成功案例 |
| [xBerry Week 4 作者报告](https://xberry.tech/xberry-lerobot-week-4-sim-to-real-transfer-80-in-simulation-50-in-reality-and-what-10-demonstrations-fixed/) | IsaacLab ACT：仿真 16/20，直接真机 10/20；加入 10 条真实示教后真机 13/20、仿真 15/20 | 作者有分母的工程报告；未找到其对应公开实验仓库，不能列为 GitHub 完整复现包 |
| [Buttergipfeli/robotics](https://github.com/Buttergipfeli/robotics)、[训练记录](https://github.com/Buttergipfeli/robotics/blob/04206bad988c3c71d87623866c3c2e3dcb109fc6/lerobot/training-progress.md) | MuJoCo→ACT→SO-101 真机链路，但已记录真机振荡/不连贯行为，后续结果待测 | 可审计的失败反例，不能因标题有 sim-to-real 就算成功 |

以上成功数来自作者报告，不是本项目重新测得。不同任务、硬件、接触判据和试验分布之间，不能直接按百分比比较优劣。

### 2.1 DISCOVERSE：本次最有针对性的 ACT 参考

它采用真实场景重建的 3D Gaussian Splatting 外观和 MuJoCo 物理，使用腕部及第三人称相机。论文 ACT 每任务 100 条仿真示教、50 次真机试验；Table II 逐项为：关笔记本 66%→86%、推鼠标 74%→90%、抓起猕猴桃 48%→76%，箭头表示加入图像增强。**抓起水果不等于完成本项目的杯子搬运、承托释放和退出。** [论文 §IV–V、Table II](https://arxiv.org/html/2507.21981v1#S5)

该表平均行存在算术不一致：印为 55%/86.5%，三任务等权复算是 62.67%/84%。已核对原 PDF 图像，因此本文只采用逐任务数字，不照抄平均收益。[原 PDF 第 6 页](https://arxiv.org/pdf/2507.21981)

公开代码值得逐段学习，而不是直接拿它的模型控制 NERO：

| 环节 | 已核源码 | 对 NERO 的直接启发 |
|---|---|---|
| 随机位置的完整数据生成 | [push_mouse.py](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/examples/tasks_airbot_play/push_mouse.py)、[place_kiwi_fruit.py](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/examples/tasks_airbot_play/place_kiwi_fruit.py) | 前者随机鼠标与鼠标垫，后者随机水果与碗；后者涵盖接近至退出。采集成功示教后再学习，而不是用几个抓前状态代替完整行为 |
| 动作与反馈分开保存 | [airbot_task_base.py](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/discoverse/task_base/airbot_task_base.py) | `obs.jq` 是观测、`act` 是施加的目标，不能用同帧实测关节冒充命令 |
| 可部署相机 | [随臂相机定义](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/models/mjcf/manipulator/airbot_play/airbot_play.xml)、[固定相机定义](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/models/mjcf/scene/qz_lab3.xml) | 采集选择两路 RGB；用真实可实现视角，不依赖运行时不可得的全局真值 |
| ACT 实现与部署 | [Imitate-All ACTPolicy](https://github.com/OpenGHz/Imitate-All/blob/f8ab39c8d829d85571ab6d0250412ba4d0698fa0/policies/act/act.py)、[评估主循环](https://github.com/OpenGHz/Imitate-All/blob/f8ab39c8d829d85571ab6d0250412ba4d0698fa0/policy_evaluate.py)、[真机接口](https://github.com/OpenGHz/Imitate-All/blob/f8ab39c8d829d85571ab6d0250412ba4d0698fa0/envs/airbot_play_5_with_cam.py) | 观测归一化、推理、目标反归一化、硬件发送有明确边界 |
| 外观重建 | [DISCOVERSE-Real2Sim](https://github.com/discoverse-dev/DISCOVERSE-Real2Sim) | 真实外观、物体几何、光照需专门处理；“零样本”不表示不使用真实照片和标定 |

公开配置和论文实验要分清：当前示例是六关节＋夹爪，某配置 chunk=25、时间聚合；不是 NERO 七关节＋米制夹爪，也没有证明这些示例参数就是论文三个任务逐项采用的精确参数。水果放碗脚本比论文抓起任务更长，脚本存在不代表该完整任务已有论文真机成绩。相机扰动代码在所查 `push_mouse.py` 中被注释，不能写成该脚本默认已启用。

### 2.2 OpenSO-101：最值得学的是仿真和真机共用部署合同

作者给出 leader 操作仿真 follower→记录→LeRobot ACT→真实 follower 的完整入口，并发布 ACT 真机演示。具体未公开的试验数量不能靠视频标题补全。[README 数据与部署流程](https://github.com/jixinyan/OpenSO-101#quickstart)

已检查的 [deploy.py](https://github.com/jixinyan/OpenSO-101/blob/5c466b47e03fdc1f8a499af5dfa3ad470f0db96f/src/openso101/sim2real/deploy.py)执行顺序是：

```text
真实关节＋两路当前图像
  → checkpoint 配套 preprocessor
  → policy.select_action
  → checkpoint 配套 postprocessor
  → 物理执行端所用 motor units
  → send_action
```

仿真端另行将 motor units 转换成仿真角度；真机端不重复做该转换。缺少归一化文件时直接拒绝。循环以数据采集频率为目标，并可打印实际频率。可学习的是这套明确的接口，不应抄其电机范围、起点自动运动或保护配置到 NERO。

它也不是无缺陷模板：源码的映射文件仍声明部分仿真关节范围与真实配置存在待协调问题。因此这里只把它当成可读的部署实现与作者演示，未证明它已解决所有标定和泛化。[机器人映射源码](https://github.com/jixinyan/OpenSO-101/blob/5c466b47e03fdc1f8a499af5dfa3ad470f0db96f/src/openso101/teleop/so101_mapping.py)

### 2.3 ANR 与 xBerry：少量真实数据有帮助，但不能保证

ANR 明确记录 ACT 零样本最佳约 75% 能拾起、仍不完成转移；换环境后敏感。仿真预训练后再用真实数据训练 1K～10K 次更新，作者称其效果接近真实数据从头训练 100K 次，但未给这组独立试验分母。网页的 90% 完整转移属于 π0，不能归给 ACT。[作者原始结果](https://www.anrrobot.com/robots/experiments/sim-to-real.html)

其代码提供真实数据进仿真、仿真数据进真机的回放校核，用于核对动作与机器人实际响应。特定关节偏置属于其 Trossen 设备，不能直接套用 NERO。[control_sim_robot.py](https://github.com/anredlich/lerobot/blob/bddd1b25635a1190dc81d48dfa4d7c52d1f78ec3/lerobot/scripts/control_sim_robot.py)、[gym-aloha 仿真](https://github.com/anredlich/gym-aloha/blob/c3d9b7f009f007572d87e814b25fad3459e16a6b/gym_aloha/tasks/sim.py)

xBerry 的 20 次试验报告显示，10 条真机示教可提高真实表现，也会损害仿真表现；这支持做真实适配及原行为回归，不支持“有十条就必然成功”。该作者建议混合训练，但网页没有给出混合训练优于顺序微调的完整受控结果，不能把建议写成实验结论。[作者报告](https://xberry.tech/xberry-lerobot-week-4-sim-to-real-transfer-80-in-simulation-50-in-reality-and-what-10-demonstrations-fixed/)

## 3 排除的常见误解

| 网上经常被一起提到的项目 | 原始证据实际是什么 |
|---|---|
| [原 ACT/ALOHA](https://tonyzhaozh.github.io/aloha/) | 真实任务由真实遥操作示教训练；另有仿真任务。原论文的真实成功不能自动当作仿真训练直接迁移成功 |
| [SplatSim](https://arxiv.org/html/2409.10161v3#S4.SS7) | 原论文 86.25% 主结果使用 Diffusion Policy。DISCOVERSE 另做了 SplatSim 渲染＋ACT 对照，两者应分开 |
| [RoboTwin 2](https://arxiv.org/html/2506.18088v2#S4.SS4) | 真机 §4.4 使用 RDT；ACT 出现在仿真 benchmark，不能借用 RDT 真机成绩 |
| GitHub 写“sim-to-real” | 可能只是代码链路、未完成的实机演示，或者模型分别在仿真和真机训练。必须追到具体 policy、数据来源、完整任务和试验记录 |

Buttergipfeli 是很有价值的反例：增加仿真数据、损失下降后，真机仍振荡；训练记录发现仿真 MJCF 范围与 follower 归一化范围不一致，腕旋可相差 78°。其物理说明还使用抓球 weld assist，不能按无辅助摩擦抓取的标准解读仿真成绩。[训练记录](https://github.com/Buttergipfeli/robotics/blob/04206bad988c3c71d87623866c3c2e3dcb109fc6/lerobot/training-progress.md)、[物理环境说明](https://github.com/Buttergipfeli/robotics/blob/04206bad988c3c71d87623866c3c2e3dcb109fc6/lerobot/physical-env.md)

## 4 NERO 当前具体差在哪里

本节区分已证明的问题和仍待验证的推断。当前保留 checkpoint 为 `1a28f4…`：单 RGB、15 维数值输入、20×8 动作块，通常执行前两点后重查。

| 环节 | 可借鉴方法 | NERO 已核事实 | 影响与归因边界 |
|---|---|---|---|
| 完整任务能力 | 完整任务运行及新位置试验 | 当前同一模型最近七个开发场景仅 1 个完整通过；历史 GPU 条件 3/7 | 在仿真内已不稳，不能仅归因真实外观；两套后端不同，不能把 3→1 直接算成训练退化 |
| 微调保持 | 每个候选都回归全部动作 | 同起点基线 780 点进入 GRASP，输出层微调后 2710 点未 GRASP；另轮完整回归 2/3→0/3 | 有直接配对证据：局部变好可能损坏整段策略 |
| 动作跨查询推进 | 合同匹配的动作语义和时间轴 | 旧夹爪块内闭合 −0.411 毫米，但重查询两点净闭合仅约 −0.000313 毫米 | 早期不闭爪有执行参数化根因；增量头已部分修好，不能继续用它解释全部当前失败 |
| 监督阶段 | 学习完整过渡及策略实际访问状态 | 8 条示教中 7 条，闭爪末点与首起抬观测近乎相同，动作标签却明显不同；局部 6/6 后完整仍失败 | 当前过渡学习不稳；尚不能声称加历史一定解决 |
| 输入有效域 | 训练和部署采用同一可获得观测 | 训练 `NominalImageEstimator` 与运行 `CardEdgesEstimator` 拒绝域不同；搬运后有蓝卡几何拒绝 | 训练里可见的输入不一定在真实运行能持续获得 |
| 真实适配 | 合格真机数据直接适配部署任务 | 79 条旧真实示教缺当前完整 15 维合同；仅辅助 RGB8 对照，留出改善约 0.010631%，未采用 | 有真实文件不等于当前主策略已学到真实分布；旧数据不是已证实的泛化修复 |
| 几何与动力学 | 命令、反馈、接触形态对应 | 模型声称约 7～10 毫米净空时现场发生碰杯；新手眼有拟合结果，未取得独立任务误差界 | 有几何不一致，不能精确分配给 TCP/杯把/视觉/跟踪各多少，也不能凭拟合 RMSE 宣称全部通过 |
| 实时部署 | 频率、图像时间和命令截止时间一致 | 最近 319 点演示仅发 66 点，第 67 点 381.648ms 超过 250ms；夹爪命令 0 条 | 该次停在闭爪之前有独立时序阻断，不是“夹爪又没学会” |
| 验收含义 | 接触、离桌、放置稳定分别判定 | 旧 `passed` 有仅跑完 50 步且距抓取点 326.846 毫米的反例；10月16日完整成功明确非 ACT | 过去看到的“通过”不全是全任务成功，进展感与实际任务能力被混淆 |

直接证据：[总根因与来源绑定](../nero_act_root_cause_audit_20261025/conclusions.json)、[输出层前后对照](../nero_act_restricted_heads_20261020/evidence/before_after_comparison.json)、[完整回归](../nero_act_event_finetune_20261024/results/summary.json)、[最新完整仿真](../nero_act_full_acceptance_20261024/run_01/acceptance.json)、[旧真实数据适配](../nero_act_real_data_adapter_20261024/README.md)、[真实辅助训练结果](../nero_act_real_visual_bridge_20261024/run_03/results.json)、[真机发送结果](../nero_act_transfer_demo_20261023/results.json)。

### 4.1 为什么不是一句“渲染差”

渲染差会改变视觉特征，公开成功案例确实为减少它投入了大量工作。我们也已证明：固定数值输入和 CPU 权重，仅换 RGB 来源，首两点关节预测可变约 0.000522 rad。[只读 RGB 对照](../nero_act_root_cause_audit_20261025/initial_image_probe.json)

但它不是当前全部失败的统一原因：同一仿真运行时的严格配对已经出现训练后退化；使用精确杯位的诊断也没有自动完成任务。新渲染器不会自动修好起抬标签、微调遗忘、蓝卡输入失效或过期发送。最近 CPU/GPU 与渲染后端一起变化，历史首帧运行 RGB 又未封存，因此尚不能单独归因最近分数差异。

### 4.2 为什么 ACT 有视觉却不能自动跟随新杯位

ACT 学的是示教分布内的“当前观测→未来动作”，没有因为名称叫 Transformer 就天然掌握所有位置的抓取规律。输入一个新坐标，也不保证网络会以正确方向、正确距离改变输出；必须由覆盖该范围、合同一致的示教和完整闭环评估证明。

摄像头实时刷新只表示有新信息。模型还要识别夹持中心相对杯子的误差、在下降时持续修正、避免杯把、维持夹持并处理遮挡。我们目前的数值几何输入和 RGB 已提供部分信息，但尚未学出在整个工作区稳定的响应。单纯把位置条件从 9 维扩成 15 维，不能替代这些能力。

“每换位置先仿真”是此前为了处理尚未合格策略而采用的固定场景轨迹验证/回放路径，不是 ACT 的理论要求。固定轨迹本身不会因为实时图像变化而学会改道；如果仍用这一方式，不能把它宣称为在线位置泛化。

### 4.3 哪些差异不能直接当作 bug

官方 LeRobot 将 `chunk_size` 与 `n_action_steps` 分开，允许只执行动作块前缀；单帧观测和关闭时间聚合也都存在。因此不能因 NERO “20 点只发 2 点”、单相机、15 维、没有 GRU，就断言实现错误。[官方 ACT 配置](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/act/configuration_act.py)

应判断的是具体实现是否有净推进、标签与执行节拍是否对应、同一输入归一化是否一致以及完整任务能否通过。原版 CVAE、历史上下文或更大网络都可以做对照，但现有证据不支持把其中任一项当成必然有效的单独药方。NERO 后验漏训曾修复；修复后仍失败，不能继续把全部问题归给 CVAE。

## 5 为什么一直没有稳定成功

最有证据的解释是一条相互影响的链：

```text
初始模型完整闭环不稳
  → 用保存的局部失败状态集中微调
  → 局部损失/夹爪探针改善，但共享输出改变接近和后续行为
  → 新策略进入示教未覆盖或标签不清的过渡
  → 停滞、下探或被几何/观察器拒绝
  → 换入口、换运行后端、换局部训练继续试
  → 指标口径不同，难以识别真正的整段提升
```

真实迁移又叠加外观与物理不一致、输入有效域和时序限制。这不是权重文件神秘损坏；来源哈希与冻结检查支持文件身份，真正发生的是允许训练的部分改变了策略行为。**反复优化局部指标，没有建立稳定的完整行为基线，是持续反复的主要研发问题。** 不同故障轮次的具体原因仍须分开，不能用一条总原因替代每次停止记录。

也有真实进展：增量头改善过旧闭爪失效，run25→run26 的同条件恢复数据修改使一个开发场景由失败变完整成功，真实控制链也确实运动过。问题是这些收益没有扩展为可复核的多位置完整真机能力。

## 6 从这些案例借鉴的下一步路线

以下是本轮研究建议，不表示已经实施、训练成功或授权了新的机械臂运动。

| 顺序 | 应完成的事情 | 需要回答的问题 |
|---|---|---|
| 1 固定比较条件 | 保留当前模型和封存日志；确定一套渲染/推理后端、视觉前端、动作单位、归一化和实际节拍。保存每次查询的真实 RGB、15维输入、预测块、发出目标与反馈 | 分数变化究竟来自训练还是运行环境？ |
| 2 建立标准 ACT 对照 | 以官方 ACT/LeRobot 为实现参考，与现有 15维＋增量头模型做并列对照；机器人仍为 NERO 的七轴与夹爪，不直接照搬 AIRBOT/SO-101 权重 | 自定义分支是否必要，还是当前数据与执行合同首先需要修复？ |
| 3 统一可学习数据 | 使用完整接近→闭爪→离桌→搬运→承托→松爪→退出；补充策略实际到达的失败恢复，明确等待/起抬监督。旧79条先尝试按日志恢复命令和时间，缺失不能造出15维真值 | 每条目标是否可执行？是否覆盖关键过渡和多个位置？ |
| 4 做真实适配对照 | 同一训练接口下，比较仿真训练、合格真实数据训练、仿真预训练＋真实/仿真混合微调；固定数据分组和计算预算，允许被证伪 | 真实数据是否带来真实收益，并保持原完整动作？ |
| 5 独立位置完整验收 | 开发位置与留出位置分开；每个候选从高位跑到退出。统计各阶段和完整任务，人工确认/恢复不得计为自主策略成功 | 是会走完整流程，还是只会局部闭爪？是否根据位置变化调整？ |
| 6 受保护在线迁移 | 复用已存在标定和记录，仅对证据明确缺项补验证；同时解决发送链时限。实机保持当前图像推理，不按每个目标重新生成轨迹 | 真机中的同一策略能否连续完成，而不是轨迹回放看起来动了？ |

第 2 步的标准模型是诊断基线，不能提前声称换回标准网络就能成功。第 4 步的真实示教也不能直接复用他人的“10条/1000步”当保证。旧数据若只有未来实测关节，应明确其代理动作标签性质；缺真实提交命令时，不能装成相同的执行目标合同。标准 RGB＋物理状态对照可以利用相应格式的数据，但不能伪填缺失的实时位置和上一命令来训练现有15维主策略。

外观改进应进入受控对照：先处理能直接核对的曝光、裁剪、颜色、杯子/背景及相机视角，验证后再决定是否值得引入 3DGS。DISCOVERSE 的重建工具值得借鉴，但当前一上来整体移植会同时改变资产、渲染、相机和数据，很难判断收益来源。它也不能替代力学和接触一致性。

若目标是尽快达到完整真机抓放，我更倾向“仿真学习完整行为＋合格多位置真机示教适配”，而不是继续要求纯仿真零样本一步完成。两者最终都仍由 ACT 在线输出动作；前者更直接利用我们已有的真实设备和资料。是否成功最终要由留出位置的实体抓放记录证明。

## 7 版本与研究范围

本轮核验的公开源码版本包括：

- DISCOVERSE：`d67f47c084aba0e0cf422a8725235f8b9238655a`；其 ACT 子模块 Imitate-All：`f8ab39c8d829d85571ab6d0250412ba4d0698fa0`。
- OpenSO-101：`5c466b47e03fdc1f8a499af5dfa3ad470f0db96f`；ACT 演示 README 更新早于本次核查日期。
- Buttergipfeli/robotics：`04206bad988c3c71d87623866c3c2e3dcb109fc6`。
- 官方 LeRobot ACT：`v0.6.0`；原 ACT 使用作者公开源码核对。版本与日期见配套来源清单。

OpenSO-101 的演示、DISCOVERSE 的作者试验和 xBerry 的工程报告都不是本机实际复现实验。没有核实到的权重、训练配置、试验分母与独立复现不补写。没有把其他策略的成功率改名为 ACT，也没有用 GitHub 星数作为结果可信性的替代指标。

相关本地资料：[项目完整复盘](nero-act-project-review-and-real-demonstration-plan-20261024.md)、[全历史根因核查](../nero_act_root_cause_audit_20261025/README.md)、[完整仿真验收](../nero_act_full_acceptance_20261024/README.md)、[本轮来源与版本清单](evidence/nero-act-public-sim2real-20261025/sources.json)。
