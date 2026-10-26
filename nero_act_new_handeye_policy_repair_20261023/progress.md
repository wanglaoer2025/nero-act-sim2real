# SDD ledger — plan: /home/user/wq/nero_act_new_handeye_policy_repair_20261023/plan.md

用户要求修正策略，沿用既有15维方案和顺序实施。当前目录为独立实验，工作区无Git仓库，不创建虚假提交或删除证据。

Pre-flight: 数据协议→采样器→训练→评估共享场景、split和哈希；使用同一新外参、原统计量及名义TCP。动态恢复保留完整physics_state/qvel/servo历史，禁止用失败ACT动作作标签。
Ruling: 本轮按用户新修正请求执行一轮1000步新标定适配，不重开已批准架构或反复申请标定；不授权硬件动作。
Task 1: in_progress — 先写数据和参考初始化回归，再生成全流程数据。

Task 1: complete — 10/10 full physical demonstrations,8 train/2 val,3 exact dynamic recoveries;4 CPU tests green.
Ruling: Context initialization validates accepted history and settled feedback; physical joint bounds remain in original FullTaskContract and are covered by integration test.
Preparation startup output-directory error fixed; failed log/protocol retained, no hardware action.
Task 2: in_progress — fixed9dd initializer, unchanged statistics/architecture,1000 fresh AdamW updates.

Task 2: complete —1000/1000 updates,52 changed tensors within allowed scopes;visual/CVAE/buffers unchanged. Source hashes verified by trainer.
Task 3: in_progress — center and2 heldout full closed-loop ACT rollouts with unchanged guards.

Final review: Important — initial evaluator bound historical runtime copies, missing actual imported runtime/guard paths. Original results preserved as results_initial_unbound_01; no retroactive pre-execution seal claimed.
Final: fixed source binding — added independent wrapper without modifying sealed training/evaluator sources; guard-mutation and actual-source coverage tests RED→GREEN, suite6/6. Re-running same frozen candidate with full before/after bindings.

Task 3: complete — bound repeat0/3 complete,3 GRASP,1 CARRY; full-task failure retained.257 actual eval source hashes verified after repeat.
Follow-up: user repair objective remains active; exact contact/vision failure modes now isolated. New controlled stage in nero_act_new_handeye_contact_repair_20261023, preserving this1000-update experiment immutable.
