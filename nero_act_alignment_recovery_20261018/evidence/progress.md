# SDD ledger — plan: docs/superpowers/plans/2026-10-18-nero-act-position-generalization-training.md

本轮继续已批准方案的局部恢复和训练任务；用户在查看run17下降停点后明确要求“修复”。顺序实施，旧来源/权重/数据只读。

- 根因基线：run17名义场景83.192秒处cup与gripper_base首次接触；末态水平偏13.62mm、高度偏高24.91mm、姿态偏12.87°，尚未GRASP。开发0/16、独立0/16，不部署。
- 恢复性探针：run17名义query_step770、TCP约120.02mm的完整快照，经原离线教师descend_to_retreat完成后续，抬升193.06mm、搬运177.67mm、无非预期接触。只证明专家可恢复，不算ACT通过。
- Task 1 complete：独立复制代码，无Git仓库，来源SHA记录在isolation.json。首次基线测试6项因模型fixture/run.py未复制而失败，已补齐只读依赖；不隐藏该失败。
- Task 2：4项新合同测试RED：拒绝新角色、不能混合来源批次、无密集下降输入、允许偷换纠正姿态。最小修复后GREEN；旧角色/分组/物理约束保留。
- Decision：保持现有ACT架构及执行协议，仅增加实际访问的下降偏差状态的离线监督；遵循批准方案6.3，不另建运行时规划器。若无效，不能将新训练完成当作修复。
- 数据计划：12条新局部纠正，6来源各160/120mm；nominal、cup-20/0、cup0/+20、historical_site0训练，cup0/-20和cup+20/+20验证。来源组不可跨划分。原48+6保留。

- 一次独立整体代码复核发现两项Important：跨批次相同origin可换组跨划分、重复启动会覆盖已有流水线状态。均已用真实失败测试RED复现并最小修复GREEN；最终全套回归重跑中。不二次派发复核。
- 复核未裁定学习效果与部署能力；这部分由冻结后完整物理任务结果判断，当前未宣称修复。
- 85.14mm名义快照（query838、姿态偏3.28°）也已由原教师恢复完整后续，无接触失败；near_contact_recoverability.json仅为诊断，不纳入12条训练清单。

- Task 2 complete：复核两项Important均RED→GREEN；最终全量129/129通过（含两项opt-in完整物理回归），full_suite_final.log，无遗留复核事项。
- Task 3 complete：新增12/12局部恢复全部物理通过、原前缀合同通过、精确重放误差0；8train4val，同源不跨组。教师后续抬升约193mm、搬运154～199mm。训练仍需单独验证，不能将教师结果视为ACT结果。
- Task 4 started：run18声明20轮，从冻结run17初始化；保留48完整+6抬升+12下降纠正，共66条、48train/18val。新独立16项seed910000起，SHA83ce2c3db1f019e9518d0178be03acc869b0e93fd93c3fad9a67ac39556c8413，已在训练前写入。名义不通过将停止扩大评估，不宣称修复。

- 训练期间额外只读诊断：run17名义query1500，TCP76.72mm、姿态偏11.09°，原descend_to_retreat教师仍物理完成后续（抬升193.07mm、搬运177.66mm、无非预期接触）；late_drift_recoverability.json，明确未加入run18训练。原三例越限停止核查也证实实测J7已越过原限位约5～8微弧度，保护没有放宽。
- run18实际初始化等价核查：actual_initializer_equivalence三点物理预测最大差2.38e-7，非成功判据。实际输入config与48train/18val、启用视觉学习、密集下降纠正一致。
