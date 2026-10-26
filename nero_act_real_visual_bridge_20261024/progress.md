# SDD ledger — plan: /home/user/wq/docs/superpowers/plans/2026-10-24-nero-real-visual-bridge.md

用户“继续调整”，按现有资料做受限离线视觉适配A/B，顺序实施。

工程选择：保持原15维冻结，新增视觉残差和独立真实8维辅助头；损失仅用原语义真实标签。风险：辅助任务未必改善原ACT，必须有留出和行为保持检查。

工程选择：A/B各200次，37/10事件窗口与多阶段保持；未通过预定门槛就拒绝，不扩大预算。

环境：当前nvidia-smi不可通信，torch.cuda.is_available=False。使用CPU同权重和冻结特征缓存，不修驱动、不操作硬件。

Pre-flight: Task 1产生128通道残差和8维辅助接口，Task 2缓存同一投影，真实与15维保持输入分开；没有缺失信息填充。无Git，独立目录和哈希留证。

Task 1: complete — 5项RED因模块缺失而失败；实施后5/5 PASS。包含恒等初始化、hook移除、state8、padding和新模块可学习且冻结投影不变。

Task 2 preload correction — run_01在训练前严格load_state_dict失败：1a28是FrozenIncrementPolicy，原实现误构造底层BoundedActPolicy。没有删前缀或忽略键；追踪原增量模型后复用同包装，完整加载base与head。新增完整checkpoint和嵌套hook回归，先观察失败再修正。run_01为0更新，仅保留失败记录；预算仍是每分支200次。

Task 2 sampling correction — run_02读入370 train/100 val真实窗口后，在保持抽样处停止。240个按阶段直接均匀抽样点有176个为原HDF5明确无效的视觉行；原VisualReferenceDataset正确拒绝。新增valid-mask抽样RED→GREEN，按原有效性、有限几何和非负误差界过滤后再分阶段。run_02为0更新，原schema未放宽。

Task 2 complete — run_03真实370 train/100 val；120多阶段特征rehearsal、120多阶段check和96访问接近check。完整220键严格加载，零残差恒等，CPU投影提取差0。CPU相对历史CUDA差明确记录；同CPU基线做保持。

Task 3 complete — A/B同初始头、同批次，各200次更新。A验证loss 0.127269104，B 0.127255574，改善0.010631%；分类相同90.55%。原15维all20最大变化6.25844e−5 rad/1.51262e−6 m，原权重和buffer无变化。未过5%收益门槛，拒绝采用B，不扩大训练。

Task 4 complete — 实验后只读10次图像/状态排列及特征空间重排，无优化、原协议和权重不变。辅助头对状态更敏感，逐点变换后全局池化不足以监督明确空间顺序；不外推原ACT无视觉。下一轮方向另列README，未追加训练。

Review — 独立复核15测试、51来源SHA、拆分、完整包装、保持和诊断；无需修正的正确性/泄漏发现。完整仿真/独立位置/真机不在范围。交付日志、README、结果和主报告更新，保留1a28。
