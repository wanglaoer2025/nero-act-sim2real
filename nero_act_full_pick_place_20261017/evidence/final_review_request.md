# Whole implementation review package

No Git repository: review the entire new package `/home/user/wq/nero_act_full_pick_place_20261017` as the new implementation. Original sources outside this directory are read-only borrowed dependencies. No merge/commit/publish is requested.

Spec: `/home/user/wq/docs/superpowers/specs/2026-10-17-nero-act-full-pick-place-design.md`
Plan: `/home/user/wq/docs/superpowers/plans/2026-10-17-nero-act-full-pick-place.md`
Ledger: `evidence/progress.md`, including all Ruling lines.

Scope: complete simulation/teacher/data/training/evaluation and ACT-only guarded hardware entry implementation. Actual ACT physical trial is NOT complete, permission has NOT been given for this new model, and no physical command has been sent. Review before any physical trial. Frozen model run6 evaluation is currently running; do not modify source files. Prior run4/run5 were4/14 and cannot authorize hardware. Current corpus18 full demonstrations plus12 explicitly partial corrective episodes; same train-only normalization, no inference large-jump clipping. Approximate TCP/table assumptions remain explicitly experimental, critical contacts require human observation.

Review `executor/{runtime,ros_execution,gate,driver,contact,phases,permission,bundle,contract,virtual_bridge}.py`, `run.py`, `training/`, `simulation/`, tests, README. Follow borrowed critical helper implementations in `nero_aligned_air_sequence_20261016_onsite/{full_runtime,ros_support}.py` and `nero_full_task_20261016/task_replay.py`; never execute their hardware entries. Main source is within writable isolated package; copied `policy_source` implementation itself was not modified.

Review Focus from plan, verbatim:
1. 杯子原已在蓝卡上：必须有非零搬运和释放后独立承托，不能仅按终点距离判断任务成功。
2. 起点和相机基准不一致：先核对 link7 与 TCP，再比较同关节状态 FK，禁止删相机内参属性绕过版本。
3. 仿真接触夹力高于真机：低力条件失败时禁止部署，不能以自由宽度跟随要求替代持物健康判据。
4. 人工等待使预测过期：恢复时重新采集和推理，不能消费旧动作块。
5. 仿真任务通过但实际执行合同违规：两项都必须通过，不能只汇报物体成功。

Read-only review. Do not edit project files, connect ROS/CAN/camera, start hardware processes, issue actions or run physics/GPU evaluation. Offline tests allowed with `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/nero_review_mpl .venv/bin/python -m unittest discover -s tests -v`. Scratch files only in /tmp. Do all review yourself; do not dispatch other agents.

Report Critical/Important/Minor findings with concrete path/line and effect, covering safety/correctness and whether runtime can actually execute the documented chain. Separate model-readiness failures from code defects. List every behavior considered but set aside under "Declined to judge" with reason. Include a verdict without claiming physical validation. Prefer tests/reproductions that root can perform without physical motion.
