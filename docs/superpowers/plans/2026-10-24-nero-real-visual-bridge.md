# Real Visual Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. 用户沿用顺序实施。

**Goal:** 用现有真实资料实现并验证不伪造15维输入的有限视觉适配对照。

**Architecture:** 冻结1a28；新增零初始化视觉残差与独立真实轨迹辅助头。A/B各200次更新，在CPU缓存冻结特征，多阶段保持，最终只形成离线结论。

**Tech Stack:** 已有PyTorch、NumPy、h5py、pytest；不安装依赖。

**Spec:** [design](../specs/2026-10-24-nero-real-visual-bridge-design.md)

## Global Constraints

- 原15维结构/状态统计/原checkpoint不改，真实8维不补零或混入几何真值。
- A/B各200次、固定batch顺序及最后一步，无追加训练。
- CPU加载同权重，device变化明确记录，不宣称旧CUDA重现。
- 37/10源episode保持分开，32 reserve不使用。
- 原始数据、8维转换产物、旧实验及保护只读；无硬件操作。

## Review Focus

- 零初始化必须使原15维输出恒等；hook需可移除，不改变base state_dict。
- 辅助输入必须8维且来自真实状态，不能接受15维拼接。
- padding不得进入损失，关闭/释放样本都保留，clip不能造成训练死梯度。
- 真实val和保持留出查询不得进入优化，统计仍仅train来源。
- 辅助loss改善或保持通过不等于完整仿真、空间精度或真机通过。

## Task 1: 小模块与边界测试

Files: `nero_act_real_visual_bridge_20261024/bridge.py`, `tests/test_bridge.py`。
Interfaces: `VisualResidual`, `AuxiliaryHead`, `auxiliary_loss`, `projected_features`, `attach_visual_adapter`。

- [ ] 先写测试并观察失败：恒等初始化、辅助8维、20点形状、有效mask、只更新新模块、hook移除、原统计恢复物理状态。
- [ ] 实施最小模块，完整pytest通过。

## Task 2: 数据和有限A/B

Files: `experiment.py`, `tests/test_experiment.py`。
Consumes: 8维派生manifest/statistics、5条既有visual15 HDF5、96条当前相机保持NPZ、1a28。
Produces: 固定样本索引和batch顺序、特征缓存、A/B最后一步sidecar、结果及来源核对。

- [ ] 先测试事件窗口不越尾、train/val分离与采用门槛。
- [ ] 准备缓存、检查零初始化和原模型权重，封存协议再训练。
- [ ] A/B各200次，评估val及全部20点多阶段保持，检查原参数未变。

## Task 3: 交付

- [ ] 本目录及原适配目录测试，独立整体复核并修正必要问题。
- [ ] 写清机制结果、采用或拒绝、未验证内容；更新复盘和HTML/证据。

非Git工作区，使用独立目录及来源哈希；本轮不创建分支或覆盖旧实验。
