# SDD ledger — plan: /home/user/wq/docs/superpowers/plans/2026-10-24-nero-act-event-finetune.md

用户“继续微调”，沿用既有顺序执行。应用systematic-debugging、ponytail、执行计划与TDD；只做离线工作。

Ruling: 最新请求授权一组最多1000步、保留现有观察器的监督消歧实验；旧“优化预算0”阶段已结束，但RGB-D未通过仍不进入训练。该实验不构成观测修复/部署通过，若有偏观察仍使策略失败需如实保留。
Ruling: 无git仓库，使用独立目录和来源哈希；不修改封存源文件。源码副本与导入路径都纳入实际证据。
Pre-flight: 新专家输出仍为原20Hz绝对关节/夹爪目标；collect读取同周期物理状态后发送，RGB重渲染保留相机和配对统计；新接近保持必须绑定实际查询的前命令而非反馈宽度。

Task1 RED→GREEN: event门5项、保持首两点/参考无梯度1项、监督抽样1项，共7/7。初次prepare缺输出父目录，已修正并保留启动失败目录/日志。
Task2 reference: 当前相机96个pre-GRASP查询完整重建，原始20×8输出最大绝对差0，未使用旧相机回放。
Ruling: 事件稳定仍可能留下最后数帧近似等待标签；在原判据下仅排除与首个抬升查询冲突的close抽样起点，保留原始所有动作/状态/图像及未来标签，审计原始与选用样本数量 — 这不证明全部时序歧义已消除。

Task1 complete: 5/5完整物理示教、精确动作回放和前缀合同通过；每条close7.3秒/release6.5秒。RGB有效查询2321条；相同近似判据原始冲突0，故本轮未实际排除任何样本，也未改写动作。数据门通过，开始单组1000步微调。

Task2 complete: 1000/1000更新，冻结参数/缓冲核对通过，70允许张量中52变化，原初始化/统计/架构未改。当前相机保持最大1.198mrad/0.0384mm；训练抬升探针前后均5/5，不把训练探针当全程改善。
Task3 running: 原模型和最终模型均使用封存contact_escape运行时及同一保护；固定三项开发回归、两项最终新位置。新20项测试通过，1个既有Axes3D警告。

Task3 startup correction: 封存rollout只创建case目录，初轮缺variant父目录导致第一个仿真未保存；dense审计还要求ROOT/contact_guard.py。已补父目录和同原源symlink，保留失败目录/日志，未改保护或重新训练。

Final review: 独立新上下文 /root/event_finetune_final_review；无Critical/Important，7项CPU测试通过，checkpoint独立核对52/70及冻结范围正确。
Final minor (deferred): 自动评估结束来源集合遗漏实际13项XML/STL，交付前另核对训练/数据完整来源作为缓解，后续入口自动闭合。
Final minor (deferred): 新位置协议顶层split/source_group继承train元数据，但实际训练只读独立data manifest，没有泄漏；保留本次冻结协议，后续统一元数据。
Final Ruling: 等完整评估生成后才判全程效果；真实几何/发送链路/部署资格及±20毫米泛化均维持未通过；复核未重跑GPU数值的边界由本轮实际GPU复现与baseline逐数组核对补充，不宣称复核者完成该项。

Task3 complete: 8/8评估完成；原回归2/3，新候选0/3、新位置0/2。9005从原GRASP停滞推进到205.15mm抬升/194.93mm转运并进入RELEASE，但蓝卡观测拒绝，仍不算完整。新模型不采用、旧基线保留且无真实资格，不追加第二组训练。
最终验证: 新7+旧保护13共20passed、1既有Axes3D警告；原三个baseline各六组数组全部完全一致；训练/数据/保持/评估全量来源结束核对，含13项XML/STL资产，无变化。中心附着airborne状态反事实仅作诊断，没有改保护或重报通过。完整结果见results.json/README.md/verification.json。
Finish: 工作目录无git，保留独立实验和来源证据；没有可合并分支，也不制造提交/发布。
