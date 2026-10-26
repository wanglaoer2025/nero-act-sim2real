# NERO 项目开发说明

本文只记录项目内部统一使用的数据格式、坐标约定和硬件开发注意事项。面向用户的安装与运行步骤见根目录 [`README.md`](../README.md)，文件摆放、接口、测试和提交规则见 [`CONTRIBUTING.md`](../CONTRIBUTING.md)，完整文档入口见[文档索引](README.md)。

## 位姿与单位

项目中“位姿”可能指关节角，也可能指末端在笛卡尔空间中的位置和姿态。两者不可混用。

| 名称 | 格式 | 单位 | 用途 |
| --- | --- | --- | --- |
| 关节角（可读配置） | `[J1, J2, J3, J4, J5, J6, J7]` | `deg` | Demo 中人工填写和打印 |
| 关节角（SDK） | `[J1, J2, J3, J4, J5, J6, J7]` | `rad` | `move_j()`、`get_joint_angles()` |
| 用户笛卡尔位姿 | `[X, Y, Z, Roll, Pitch, Yaw]` | `mm, mm, mm, deg, deg, deg` | 人工测量和填写抓取点 |
| SDK SI 位姿 | `[x, y, z, roll, pitch, yaw]` | `m, m, m, rad, rad, rad` | `move_p()`、`move_l()`、`get_flange_pose()` |

### `deg` 与 `rad`

- `deg`（度）方便人阅读，例如 `90°`。
- `rad`（弧度）是 SDK 使用的角度单位，例如 `90° = π/2 rad`。
- 关节目标在发送给 SDK 前使用 `degrees_to_radians()` 转换。
- SDK 返回的关节角在显示时使用 `radians_to_degrees()` 转换。

```python
from src.utils.arm import degrees_to_radians, radians_to_degrees

target_rad = degrees_to_radians([0, -30, 0, 120, 0, 0, 0])
robot.move_j(target_rad)

feedback = robot.get_joint_angles()
if feedback is not None:
    print(radians_to_degrees(feedback.msg))
```

### 用户位姿与 SI 位姿

抓取参数使用毫米和度，便于测量：

```python
PICK_POSE_MM_DEG = (-526, -145.51, 179.77, 113.6, 83.563, 33.254)
```

发送给 SDK 前统一转换：

```python
from src.utils.arm import pose_mm_deg_to_si

pick_pose = pose_mm_deg_to_si(PICK_POSE_MM_DEG)
```

不要把毫米直接传给 SDK，否则 `100 mm` 会被解释为 `100 m`；也不要把度直接传给要求弧度的接口。

## Roll、Pitch、Yaw

SDK 位姿后三项表示法兰姿态：

- `Roll`：绕 X 轴旋转；
- `Pitch`：绕 Y 轴旋转；
- `Yaw`：绕 Z 轴旋转。

`pyAgxArm` 的法兰位姿采用固定坐标系的 XYZ 外旋，等价于机体坐标系的 ZYX 内旋，旋转矩阵为：

```text
R = Rz(Yaw) · Ry(Pitch) · Rx(Roll)
```

欧拉角具有周期性，例如 `179°` 与 `-179°` 只相差 `2°`。项目使用 `angle_error()` 计算跨越 `±π` 时的最小角度误差。

## 法兰位姿与 TCP 位姿

- `flange pose`：机械臂最后一个连杆安装法兰的位置和姿态；`get_flange_pose()` 返回这一位姿。
- `TCP pose`：工具中心点（Tool Center Point）的位置和姿态，例如夹爪指尖中心。
- 安装工具后，TCP 通常相对法兰存在固定偏移；未配置工具偏移时，不要把法兰位置当作真实抓取接触点。

当前 Demo 以法兰位姿为运动基准。需要精确控制夹爪指尖时，应先测量工具偏移，再统一使用 SDK 的 TCP 转换接口。

## 绝对目标与相对偏移

SDK 运动接口接收绝对目标：

- `move_j(joints)`：绝对关节角目标；
- `move_p(pose)`：绝对笛卡尔目标，路径不保证为直线；
- `move_l(pose)`：绝对笛卡尔目标，控制器规划直线路径；
- `move_c(...)`：笛卡尔圆弧路径。

项目中的 `offset_pose()` 只是先在 Python 中复制起始位姿并叠加 `dx/dy/dz`，再生成新的绝对目标。偏移量使用米：

```python
from src.utils.arm import offset_pose

target = offset_pose(start_pose, dx=0.0, dy=0.1, dz=0.05)
```

上例表示 Y 方向移动 `100 mm`，Z 方向移动 `50 mm`，Roll/Pitch/Yaw 保持不变。

## 运动速度约定

每条运动必须在公共函数调用处显式传入 `speed_percent`，不要预先调用
`robot.set_speed_percent()` 并依赖控制器保留的全局状态：

```python
move_pose_and_wait(
    robot,
    "move_l",
    "低速接近",
    target,
    speed_percent=5,
    timeout=30.0,
)
```

`move_joint_and_wait()`、`move_to_prepare_pose()` 和 `execute_linear_path()` 采用同一参数。参数是控制器相对速度百分比，范围为 `1..100`；相同百分比不代表 `move_j`、`move_p` 与 `move_l` 具有相同的末端物理速度。抓取流程按运动阶段选择速度：空载移动使用 `transit_speed_percent`，接近、抓取和放置使用 `precision_speed_percent`。

## 现场确认基坐标方向

不要根据相机画面方向或机械臂安装朝向猜测 base 轴。将机械臂放在工作区中部，确保三个正方向和返回路径各有至少 `50 mm` 空间并准备好急停，然后运行：

```bash
python -m src.demo.base_axes
```

程序保持法兰姿态不变，以 `5%` 速度分别沿 base `+X`、`+Y`、`+Z` 点动 `30 mm`，每次观察后返回同一起点，并在每一轴前要求输入对应的 `X`、`Y` 或 `Z`。实体运动方向就是控制器真实基坐标方向，负方向与其相反。不要同时运行其他 CAN 控制程序。

## 到位误差

项目分别判断位置和姿态误差：

- `position_tolerance`：米，例如 `0.004` 表示 `4 mm`；
- `angle_tolerance`：弧度，例如 `math.radians(3)` 表示 `3°`；
- `tolerance_deg`：部分关节等待函数为方便配置，明确使用度。

不要为了绕过不可达、碰撞或奇异位姿问题而直接放宽误差或提高速度。

## 固件配置

创建机械臂时的 `firmeware_version` 必须与真实固件匹配。SDK 参数名本身拼写为 `firmeware_version`，不要自行改成 `firmware_version`。

| NERO 固件 | SDK 配置 |
| --- | --- |
| ≤ 1.10 | `NeroFW.DEFAULT` |
| 1.11 | `NeroFW.V111` |
| 1.12 | `NeroFW.V112` |
| ≥ 1.20 | `NeroFW.V120` |

本项目机械臂当前使用 `NeroFW.V120`。固件 ≤ 1.11 需要开启 CAN 推送；1.12 及以上上电即推送，`set_normal_mode()` 不再生效。

## 开发约定

- 从项目根目录使用 `python -m ...` 运行模块，避免导入路径随工作目录变化。
- 可复用的连接、转换、反馈等待和打印函数统一放在 `src/arm/`。
- Demo 只保留用户参数、安全确认和动作顺序。
- 硬件动作统一调用 `confirm_action(action, warning, token)`；提示必须说明操作和风险，口令使用大写动作词，空回车默认取消。
- 通用运动使用 `MOVE` 或 `PATH`；自动标定使用 `AUTO`；视觉抓取使用 `PICK`、`GRASP`；使能、失能和急停复位分别使用 `ENABLE`、`DISABLE`、`RESET`。
- 录音开始和标定菜单不触发机械动作，可以保留回车或单字符命令。
- 所有运动开始前检查状态和反馈，并准备实体急停。
- `disconnect()` 只断开 CAN，不会自动失能机械臂。
- 失能、复位或断电前必须托住机械臂，防止关节卸力下落。
- 修改运动路径后，先使用低速度和较小位移进行现场验证。

## 调试顺序

出现无反馈或无法运动时，按以下顺序排查：

1. `ip -details -statistics link show can0`：确认接口、波特率和错误计数；
2. `candump can0`：确认总线上持续存在反馈帧；
3. 查询固件并核对 `NeroFW`；
4. 读取 `get_arm_status()`；
5. 读取 `get_joint_angles()`；
6. 读取 `get_flange_pose()`；
7. 全部正常后才发送运动或失能命令。

CAN 接口为 `UP` 只表示主机网络接口已启用，不代表机械臂正在发送反馈。

## 2026-09-01 cup-to-book 实机验收记录

`src.demo.yolo_pick_place` 已在当前固定场景中完整跑通一次监督抓取与放置流程：ZED RGB-D 检测 `cup` 和 `book`，将坐标转换到 NERO base，机械臂从固定初始位姿侧向抓取空杯并放到 book 中心，随后松爪退出。

2026-09-01 实机验收使用的固定关节姿态为 `[10.5, -7.25, 5.73, 90, 5.73, 5.73, 30] deg`，SDK 正运动学估计法兰约为 `[-214, -91, 482] mm`。该值继续作为已有 ACT 数据的固定起点，避免改变旧实验的数据分布。

当前 `python -m src.arm.prepare` 以 `10%` 速度通过一次 `move_j` 进入抓杯准备姿态 `[10.5, -7.25, 5.73, 90, -1.37, 16.14, 7.51] deg`。SDK 正运动学估计法兰约为 `[-219, -91, 479] mm / [-90°, 0°, 180°]`，使夹爪 TCP 朝 base `-X`，两指沿 base `Y` 水平开合；它只调整腕部三关节并保持前四关节不变。该新目标仅完成离线 FK 验证，而且项目没有碰撞规划，首次必须从已知姿态空载低速验证完整关节扫掠路径。

### 已验证配置

| 项目 | 当前值 |
| --- | --- |
| 相机 | ZED，S/N `39602891`，HD720@30，NEURAL 深度，画面旋转 180° |
| 检测模型/类别 | `yolov8n.pt`；只使用 `cup`、`book` |
| 相机外参 | `config/camera_to_base.json`；19/20 个样本，平移 RMSE `6.65 mm`、最大 `11.27 mm`，旋转 RMSE `1.08°`、最大 `1.98°` |
| TCP | 法兰局部 `+X` 方向尺测 `135 mm`，仍属于近似值 |
| 横向姿态 | `Roll=-90°`、`Pitch=0°`、`Yaw=180°` |
| 物理方向 | TCP 朝 base `-X`；两指沿 base `Y` 水平张开 |
| 杯子 | 高 `220 mm`；底部直径 `60 mm`；顶部直径 `100 mm`；半高抓取直径约 `80 mm` |
| book | `175 × 95 mm`，放置点取检测中心 |
| 视觉杯心修正 | `1 mm`；杯子物理半径仍为 `40 mm` |
| 稳定门 | 连续 8 帧，最大三维波动 `5 mm` |
| 工作空间 | base XYZ：`[-650, 0] × [-300, 300] × [0, 500] mm` |
| 运动 | 无接触段 `30%`，接近/放置段 `5%`，单段超时 `60 s` |
| 夹爪 | 张开 `95 mm`，夹紧目标 `80 mm`，夹持力 `0.5 N` |
| 放置与返回 | 抬升 `80 mm`；book 上方余量 `80 mm`；落点间隙 `5 mm`；松爪后沿 base `+X` 退出 `50 mm`，垂直抬高后返回启动关节位置 |

最初曾把杯子物理半径 `40 mm` 同时用于视觉表面到杯心修正。按当前相机位置，这会额外引入约 `+23 mm` 的 base Y 修正，与实测约 `+30 mm` 的一致偏差相符。将视觉修正独立设为 `1 mm` 后，悬停和后续流程对齐良好；不要再次把 `--source-radius-mm` 用作视觉修正量。

### 已验证操作

```bash
source /home/user/nero/bin/activate
python -m src.demo.yolo_pick_place \
  --source-class cup \
  --destination-class book \
  --pick-place \
  --workspace-mm -650 0 -300 300 0 500
```

运行前先把机械臂放在已验证且偏向 base `+Y` 的安全初始位置，并清空从 book 上方返回该位置的关节运动扫掠区域。画面稳定后按 `m` 锁定目标；输入 `PICK` 后程序连接并记录启动关节角。机械臂在杯面前 `50 mm` 停止，输入 `GRASP` 后自动完成抓取、放置、退出、垂直抬高和返回启动位置。

### 当前边界

- 这是固定相机、固定初始位姿、空杯和当前桌面布局下的一次成功验收，尚未统计连续成功率；
- TCP 为尺测近似值，不能替代枢轴标定；相机、底座、夹爪安装或桌面变化后必须重新验证；
- 当前依赖控制器 `move_p` 解算，不包含逆解候选筛选、碰撞检测或通用路径规划；
- 返回位置取程序连接后的启动关节角，不是硬编码坐标；若启动位置不安全，程序也会返回该位置；
- YOLO 通用模型只适用于当前 `cup`/`book`，没有覆盖遮挡、多实例或未知物体；
- 保留 `PICK`、`GRASP` 两个人工确认和实体急停，达到连续多次成功的验收标准前不要取消。

全套无硬件单元测试必须在提交前通过；测试不代替实体机械臂的重复性和安全验收。
