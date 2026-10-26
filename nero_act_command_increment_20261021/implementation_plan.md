# ACT命令增量分支 Implementation Plan

> For agentic workers: use superpowers:executing-plans, sequential implementation.

Goal: 实现并离线验证有因果命令锚点的学习夹爪分支。
Architecture: 新目录引用原冻结ACT，仅训练新增量头；原输出转换和保护通过旧仿真调用原样执行。
Tech Stack: existing PyTorch, h5py, NumPy, MuJoCo; no new dependencies.
Spec: design.md

Global Constraints: old artifacts/data immutable; no CAN/ROS/hardware; RGB+15→20x8; first2 only;1000 updates final checkpoint; no oracle stage input; all original guards and numeric thresholds preserved.
Review Focus: causal previous-state reconstruction; padding/target range; frozen base buffers and cached features; source-bound checkpoint loading; unchanged original scene/qualification snapshots.

- [ ] Task2 TDD increment readout/model: zero head keeps previous command, physical/feedback bounds, invalid observations rejected, arms unchanged, trainable head only, actual features reentrant cleanup; tests/test_increment.py RED→GREEN.
- [ ] Task3 cache/train: all original training views map to deduplicated same current frames, cache first2 ACT features and paired targets; verify runtime feature agreement, padding and increment range, seal source/data; fixed1000 updates and stats with frozen digest checks. tests/test_training.py RED→GREEN.
- [ ] Task4 qualification: original3 inputs/423 arms/original6 snapshots; conditional highstart only; no scene/guard edits, no runtime manual actions. Reuse original mechanism_passed/qualified functions.
- [ ] Task5 final fresh whole-branch review, one Important fix pass, green suite, artifact verifier and actual outcome report. No second1000 or deployment.
