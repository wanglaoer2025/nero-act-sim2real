# NERO 贡献与工程规范

本规范用于后续代码和文件保持一致。已公开入口只有在明确迁移任务中才能改名；改名时必须在同一次修改中更新代码、测试、当前文档和迁移说明。

## 文件放在哪里

| 内容 | 位置 |
| --- | --- |
| 机械臂使能、失能、准备、复位和夹爪命令 | `src/arm/` |
| 机械臂连接、确认、转换、反馈和运动辅助 | `src/utils/arm.py` |
| camera→base、手眼和 TCP 标定 | `src/calibration/` |
| 已完成、可独立运行的能力演示或基线流程 | `src/demo/` |
| 尚未完成验收的案例流程 | `src/<case_name>.py` |
| ACT 专用实现和命令 | `src/act/` |
| 倾倒、视觉和语音解析辅助 | `src/utils/pouring.py`、`vision.py`、`voice.py` |
| 语音识别和抓取可运行入口 | `src/voice/` |
| CAN、固件、环境维护工具 | `scripts/` |
| 本机标定/现场参数 | `config/` |
| ACT 训练与推理实验配置 | `configs/` |
| 自动测试 | `tests/` |
| 用户操作入口 | `README.md` |
| 坐标、单位和长期开发约定 | `docs/development.md` |
| 已发生问题、证据、根因和修复 | `docs/robot-troubleshooting.md` |
| 某子系统的设计或路线 | `docs/`，并登记到 `docs/README.md` |

`data/`、`captures/`、`calibration/`、`logs/`、`artifacts/`、`checkpoints/` 和模型权重是运行产物，不放源码，也不提交 Git。需要共享的小型默认配置应提交 `.example.json` 或 YAML；密切依赖现场安装的实际参数留在本机。

## 命名规范

| 对象 | 规则 | 示例 |
| --- | --- | --- |
| Python、Shell 文件及源码目录 | 小写 `snake_case` | `src/arm/enable.py`、`can_multi_activate.sh` |
| Python 测试 | `test_<subject>.py` | `test_auto_hand_eye.py` |
| Python 函数、方法、变量 | 小写 `snake_case` | `wait_joint_reached()`、`speed_percent` |
| Python 类 | `PascalCase` | `ArucoMarkerDetector` |
| Python 常量 | `UPPER_SNAKE_CASE` | `MOVE_TIMEOUT_S` |
| Markdown 文档及文档目录 | 小写 `kebab-case` | `robot-troubleshooting.md`、`docs/act/` |
| YAML、JSON、数据及实验名 | 小写 `snake_case` | `act_nero_day6.yaml`、`camera_to_base.json` |
| CLI 选项 | 小写 `kebab-case` | `--speed-percent` |
| 配置键 | 小写 `snake_case` | `camera_names` |

`README.md`、`AGENTS.md`、`CONTRIBUTING.md`、`LICENSE` 和 Python 的
`__init__.py`/dunder 方法属于约定例外；私有标识符允许一个前导下划线，
`unittest.setUp()` 等框架回调沿用框架名称。缩写进入路径和普通标识符后按
普通单词处理，例如 `hand_eye`、`tcp`、`yolo`、`act`。包名已经表达领域时，
文件名不再重复包名，例如使用 `src/arm/enable.py`，不用 `src/arm/arm_enable.py`。
不要同时保留 CamelCase 和
snake_case 两套实现或兼容壳；确需兼容外部发布接口时，必须写明删除版本和期限。
`third_party/` 保持上游命名，不纳入本仓库重命名规则。

## Python 与接口

- Demo 模块开头写明仓库根目录运行命令、是否连接硬件，以及实机确认要求。
- CLI 提供 `parse_args()`（参数较少时可直接放在 `main`）、`main() -> int` 和模块执行保护。
- 未完成案例允许暂存 `src/` 根目录；完成验收后移入 `src/demo/`。没有 `main()` 且只被其他流程调用的公共辅助模块放入 `src/utils/`。
- 优先使用标准库和已有依赖；没有真实第二个实现时不增加抽象层、工厂或插件系统。
- 公共函数参数明确写单位后缀，如 `_mm`、`_m`、`_deg`、`_rad`、`_s`、`_hz`。不要依靠注释猜单位。
- 坐标或 action 使用固定 shape，并在边界处检查维度、有限值和范围。机械臂 qpos 是七维；夹爪独立；ACT action 是八维。
- 路径参数使用 `pathlib.Path`。版本化配置使用仓库相对路径；外部权重或设备路径必须可由 CLI/config 覆盖。
- 日志和异常使用中文说明现场含义，并保留底层异常作为上下文。运动失败至少报告步骤名、目标、实际反馈和控制器状态。

## 实机 CLI

- 默认模式必须不运动；实机动作要求显式 `--execute`。
- 动作前调用 `confirm_action(action, warning, token)`；口令为大写动作词，空回车取消。
- 普通失败停止后保持当前位置，不自动失能或急停；严重通信/碰撞风险和人工中断按入口定义处理。
- 速度在调用运动函数时显式传入 `speed_percent`。区分空载转场与精密接近，但不要依赖控制器残留的全局速度状态。
- 等待真实反馈判断到位，不用固定 `sleep` 替代状态检查。允许控制器路径失败时，输出诊断后再尝试明确、安全且数量有限的候选路径。
- 每个新实机流程都按影子模式、悬停、空夹爪、空物体、完整任务逐级验收。

## 配置与实验命名

- `config/*.example.json`：可提交的字段模板和保守默认值。
- `config/*.json`：当前机器的标定或现场参数，通常被忽略。
- `configs/<system>_<task>_<variant>.yaml`：训练/推理实验，训练和实机配置共享清晰的实验前缀。
- `data/<experiment>/`、`checkpoints/<experiment>/`、`artifacts/<experiment>/`、`logs/<experiment>/` 使用同一个 `<experiment>` 名称，避免模型、统计量和日志串用。
- 配置改变 action shape、相机列表、图像尺寸、坐标条件或时间对齐时，必须使用新的实验名，不能覆盖旧 checkpoint。

## 测试和文档

修改后运行最小相关测试：

```bash
source /home/user/nero/bin/activate
python -m unittest discover -s tests
```

硬件代码的自动测试只验证纯函数、参数、状态机和 mock 边界；不能宣称完成真机验收。新增 CLI 至少运行 `python -m <module> --help` 或其 dry-run。

文档不复制同一套命令到多个文件：README 保存用户常用命令，专题文档解释原理和完整流程，troubleshooting 保存问题闭环，ACT 的阶段计划和实施报告由 `docs/act/README.md` 导航。

## Git

- 提交前检查 `git status` 和 `git diff`，不混入数据、权重、日志或无关用户改动。
- 使用 Conventional Commits，例如 `feat(vision): ...`、`fix(arm): ...`、`docs: ...`、`refactor(act): ...`、`test: ...`。
- 一个提交只表达一个可回退的目的。未经用户明确要求，不提交、不推送，也不改写历史。
