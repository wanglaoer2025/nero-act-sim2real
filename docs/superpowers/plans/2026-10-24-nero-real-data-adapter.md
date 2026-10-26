# Existing Real Demonstration Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. 顺序实施，每项用测试验证；不启动新训练或硬件。

**Goal:** 将79份旧真实教师资料适配成可读取的独立RGB8未来状态训练视图。

**Architecture:** 在独立目录生成小型派生HDF5、源快照、训练统计和清单。图像用VDS引用，加载器先验证源；保持现有ACT五项batch格式和原15维拒绝边界。

**Tech Stack:** 已安装Python、h5py、NumPy、PyTorch、pytest；无新依赖。

**Spec:** [design](../specs/2026-10-24-nero-real-data-adapter-design.md)

## Global Constraints

- 原始文件只读、拒绝覆盖；不改旧模型、保护或历史证据。
- 默认lead=2，夹爪事件不移位，不重复前视，不复制尾部标签。
- 不伪造15维、深度、曝光同步或命令回执。
- 旧37/10划分仅作episode机制对照，32份reserve。

## Review Focus

- 丢失VDS来源必须报错，不能输出零图像。
- 相同源文件不能进入两个split，统计不能用val或reserve。
- 非有限值、倒序时间、错误声明或不完整流必须在写出前拒绝。
- 输出已存在或源文件在适配后变化必须拒绝读取。
- 模型前向通过仅代表接口可用，不代表任务或泛化通过。

## Task 1: 转换、索引与受保护读取

Files: `nero_act_real_data_adapter_20261024/adapter.py`, `tests/test_adapter.py`。

Produces: `convert_episode(source, destination, lead_steps=2, split='reserve')`；`read_window(path,start,stats,image_size=(180,320))`；`fit_statistics(train_paths)`。

- [x] 写真实小HDF5测试，验证索引/夹爪时序、尾部mask、源缺失、错误数据、覆盖拒绝和train统计。
- [x] 观察测试失败，实施最小转换和读取，运行本目录全套pytest。

## Task 2: 准备实数据与训练接口

Files: `prepare.py`, `preflight.py`, `tests/test_prepare.py`。

Consumes: Task 1函数、已封存审计、两份旧manifest和split。
Produces: `data/`79份派生HDF5、`manifest.json`、`statistics.json`、`results.json`。

- [x] 先测试同源split冲突、缺失split文件及预处理/统计来源。
- [x] 导出79份，比较全部关节/夹爪/时间，核对源快照未变。
- [x] 使用真实batch运行旧ACT CPU前向及loss；不optimizer.step，不迁移当前15维权重，不写checkpoint。
- [x] 独立调用原15维审计确认仍拒绝旧语义。

## Task 3: 交付

- [x] 完整pytest、CLI帮助、来源指纹与输出链接检查。
- [x] 独立整体复核；必要修正用RED→GREEN验证。
- [x] 写README、结果边界和主复盘补充；更新HTML与证据索引。

工作区`/home/user/wq`没有Git，使用独立目录和哈希留证，无法创建worktree或提交；不改外部nero仓库。进度保留在该目录progress.md。
