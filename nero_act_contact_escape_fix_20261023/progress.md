# SDD ledger — plan: /home/user/wq/nero_act_contact_escape_fix_20261023/plan.md

Ruling: 停止追加训练，回退已证实接近退化的第二轮模型；依据原始几何诊断修正离线支撑接触逃离判定 — 原候选下一步实际向上，原解释错误；如判断错误，完整物理回归仍应拦截，不能用于真实部署。
Pre-flight: 不修改已封存运行时，复制后只替换一处纯拒绝判定；动作合同、物理接触检测和结果验收函数原样保留。100微米限制仅是离线名义接触重叠检查，不是实测TCP/桌面精度。

Task rejection: 6项回归通过，包括原始上升失败帧、下压/恶化转动/静止、无支撑/超界、普通自由空间碰桌、真机误用拒绝、端点安全但中间下穿的扫掠检查。运行时差异仅1个导入与1个原判定替换，见runtime_patch.diff；不改模型或动作。

Final review: fresh-context gpt-6-astra/high，只读复核完成；无Critical。结果说明中的全局“目标未改写”为Important，已限定为本接触补丁不改写目标，并明确保留原有接触臂保持及抓取前夹爪保持；对应测试RED→GREEN。
Final: Ruling: 将越过原地板阈值第一采样点未检查TCP向上的边界问题按Important处理 — 虽未证明杯底恶化，但会违背本次新增逃离保证；已用真实边界反例RED→GREEN补齐。
Final: Ruling: 松爪图像仅两角可见时，接受至少三条连续可见边支持同一名义矩形 — 反算尺寸和三边直线支持仍充分，避免假装完全遮挡；若用于复杂真实画面仍可能误识，因此保持离线限定和全部真实部署标志为false。新分支为最终复核后追加，经回归但未二次独立复核。
Final: Ruling: 独立复核未判断实机安全/标定精度与暗色遮挡物实景误识 — 保留未验收状态，不将名义仿真结论升级为真机许可；未完成七场景结果继续保持complete=false。
初轮离线评估为修复上述问题中断，源文件前后哈希核对通过；已保留results_initial_review_01、完整初轮日志和源码版本，不冒充最终验收。

Final: fixed 全局动作说明 — test_action_provenance_discloses_retained_holds_without_global_false_claim RED→GREEN；新增边界 — test_first_floor_crossing_still_requires_upward_tcp RED→GREEN；完整当前回归13/13。
Task final evaluation: complete — 7场景全部完成，301份实际代码/资产/权重来源前后哈希一致；3/7完整通过，新增两位置0/2，固定中心抬升199.79毫米、转运190.30毫米、稳定101帧并退出。full_task_passed保持false，非泛化或实机通过。
Remaining: 9005为严格微接触逃离单调条件停止，包含约8毫米杯底图像估计偏差；9201/9301低位接近不可靠，9300附着杯地板判定停止。本轮不追加训练、不扩大例外；ROS/CAN和所有硬件命令未由本轮启动。
Workspace: 无git仓库，保留全部实验目录和日志；没有合并、提交、推送或删除工件。
