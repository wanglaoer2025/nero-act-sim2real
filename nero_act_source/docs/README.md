# NERO 文档索引

这里是项目文档的唯一导航入口。先按任务选择文档，不需要从头阅读整个 `docs/`。

## 我应该看哪个文档

| 任务 | 文档 | 性质 |
| --- | --- | --- |
| 安装、CAN、常用运行命令 | [项目 README](../README.md) | 用户入口，持续维护 |
| 新 Agent 接手、代码地图、当前状态 | [AGENTS.md](../AGENTS.md) | Agent 必读，持续维护 |
| 文件摆放、API、配置、测试和 Git 规范 | [CONTRIBUTING.md](../CONTRIBUTING.md) | 工程规范，持续维护 |
| 位姿、单位、坐标系、TCP、速度 | [development.md](development.md) | 技术参考，持续维护 |
| 查询已经遇到过的问题和根因 | [robot-troubleshooting.md](robot-troubleshooting.md) | 故障知识库，持续维护 |
| 视觉标定、定位、抓取路线 | [vision-grasp-roadmap.md](vision-grasp-roadmap.md) | 视觉路线与验收 |
| 三烧杯定位与监督对齐检查 | [项目 README](../README.md#yolo-视觉抓取放置) | 实验命令；默认仅 dry-run |
| 透明烧杯与器皿倾倒 | [transparent-beaker-pouring-implementation-plan.md](transparent-beaker-pouring-implementation-plan.md) | 详细设计参考；以代码和现场配置为准 |
| 亚甲基蓝可逆变色完整演示 | [methylene-blue-experiment-implementation-plan.md](methylene-blue-experiment-implementation-plan.md) | 实施任务书；分阶段开发与验收 |
| 亚甲基蓝 Demo 可执行版本与回档 | [methylene-blue-version-history.md](methylene-blue-version-history.md) | 版本台账、启动命令与安全回档 |
| ACT 数据、训练、回放、推理 | [act/README.md](act/README.md) | ACT 唯一入口 |
| 仿真电脑离线数据导入与 SDK 依赖 | [ACT 离线源码补充包](act/README.md#离线源码补充包2026-10-13) | 2026-10-13 导出包修复与复测说明 |

## 文档职责

- `README.md` 回答“怎么安装、怎么运行”。不要继续添加故障分析和内部实现细节。
- `src/demo/*.py` 的模块说明只记录该 demo 的直接运行命令、硬件影响和确认要求；完整流程仍由 README 或对应专题文档维护。
- `AGENTS.md` 回答“这个项目是什么、改哪里、当前做到哪、什么不能做”。
- `CONTRIBUTING.md` 回答“新代码、配置、测试和文档应该怎样写”。
- `development.md` 保存不会频繁改变的技术约定，例如单位、坐标系和运动接口。
- `robot-troubleshooting.md` 用“现象 → 证据 → 根因 → 修复 → 验证”记录真实问题。
- 路线/方案文档说明某一子系统，不复制全部命令；常用命令仍回到 README。
- `docs/act/act-day*-nero-codex-prompt.md` 是当时的开发任务书，不是当前操作手册。实施结果优先看 report 和代码。

## 新文档规则

只有内容无法合理加入上表中的现有文档时才新建文件。新建后必须：

1. 使用小写 `kebab-case.md` 文件名；只有 `README.md` 等固定入口名例外。
2. 在本索引登记用途、状态和目标读者。
3. 在开头写清日期、适用范围，以及它是规范、操作、设计还是历史记录。
4. 命令必须从仓库根目录可运行；涉及实机时明确 dry-run、确认口令和安全边界。
5. 过时内容直接更新，并在确需追溯时标成“历史”，不要再创建 `new/final/final2` 文档。

## 内容写到哪里

| 新内容 | 记录位置 |
| --- | --- |
| 新用户命令 | `README.md` 对应功能小节 |
| 新公共接口或单位约定 | `docs/development.md` 和必要的 docstring |
| 新故障及修复 | `docs/robot-troubleshooting.md` |
| ACT 当天实施结果 | 对应 `docs/act/act-dayN-report.md` |
| 项目总体状态发生变化 | `AGENTS.md` 的“当前状态” |
| 文件/API 规则变化 | `CONTRIBUTING.md` |
