# ACT Observation Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 分离杯位误差和策略误差，建立可验证的遮挡观测修复，不把诊断真值或只读深度质量当作真实抓放验收。

**Architecture:** 复用已封存ACT与仿真运行时，新增单独命名的诊断观测适配器。准确杯位来自匹配曝光快照，仅替换15维中的杯位三项，其他输入和动作保护不变；正常观察器与诊断类分开。

**Tech Stack:** Python、NumPy、MuJoCo、PyTorch、OpenCV、现有pyzed。

**Spec:** `docs/superpowers/specs/2026-10-23-nero-act-observation-repair-design.md`

## Global Constraints

- 固定候选1a28f4fddd3095791403028b639b99f9775b91caab45c950c2d0f518955bc0f0；本阶段优化预算0。
- 不改已封存源文件；独立工程 `nero_act_observation_repair_20261023/`，无git仓库，保留来源哈希和变更记录。
- 诊断真值不算迁移验收，不用于正常训练或真机观测；杯位真值必须匹配曝光，蓝卡/RGB/反馈/夹爪上下文不变。
- 原动作、接触、碰撞、时限保护不变；不启动ROS/CAN、不发送硬件运动。

## Review Focus

- 延迟/确认等待期间多帧曝光不得绑定当前时刻真值，适配器测试验证逐帧快照配对。
- 观测适配不得原地修改RGB、状态或蓝卡三项，逐数组比较。
- 诊断成功不得写成真实或独立验收通过，汇总测试固定qualification=false。
- 深度单位、光学坐标与图像180度旋转分开，保存元数据并以已知变换的数值测试核对。
- 低置信度、非有限深度、完全遮挡或物体移动不得伪装成精度通过。

## Task1: 观测因果诊断

**Files:** `oracle_observation.py`、`test_oracle_observation.py`、`diagnose.py`、`diagnostic_protocol.json`。
**Interfaces:** `replace_cup_for_diagnostic(features,cup,tcp,*,diagnostic_only)->np.ndarray(6)`；`DiagnosticHistory.sample(model,data,renderer,observer)`复用原采样合同。

- [ ] 写并运行失败测试：只改杯位三项、输入不变、禁止非诊断调用、同曝光快照匹配、诊断结果不能计验收。
- [ ] 最小实现并运行全套测试，生成固定场景/来源协议。
- [ ] 正常9005重跑，对比封存停止点、原始ACT块和状态；一致后执行中心及9005/9201/9300/9301真值诊断。
- [ ] 检查实际代码、资产、模型前后哈希，记录每个停止原因及因果结论；不修改正常验收。

## Task2: 遮挡观测与深度只读核对

**Files:** `capture_failures.py`、`test_cup_observer.py`、经诊断确定的`cup_observer.py`、`depth_quality.py`及测试。
**Interfaces:** 当前RGB→杯底中心和显式质量；深度质量工具只写文件，不导入机器人SDK/ROS；表面点统计不输出伪杯底。

- [ ] 保存正常失败与诊断轨迹的匹配曝光帧，复现旧观察器的伪抬升；独立划分修复/验证帧。
- [ ] 根据误差形态确定一个可验证的遮挡修复，先RED→GREEN再测试未用于拟合的帧；若无法观测，输出无效。
- [ ] 测试无效深度、置信度过滤、单位/旋转和物体移动拒绝；再进行只读ZED采集，硬件不可用时如实记录。
- [ ] 只有观测机制通过才提出后续小预算训练；当前任务不自动扩大预算。

## Task3: 结果复核与交付

- [ ] 当前测试及相关旧回归全部运行，逐项核对；保存结果和未解决项。
- [ ] 一次独立只读最终复核，重要问题按RED→GREEN修正，不重写历史证据。
- [ ] 汇总是否视觉误差为主、是否策略仍失败、深度质量是否足以开展下一步；真实部署标志保持false。
