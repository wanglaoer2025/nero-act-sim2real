# 现有真机示教的离线训练视图适配

2026-10-24。用户要求基于已有资料完成适配。本目录包含转换器、带源检查的数据集、独立核对入口、测试和实际派生数据。没有启动正式训练、更新模型或操作硬件。

## 已完成的实际结果

| 项目 | 结果 |
|---|---|
| 只读原始数据 | 79份、56,419帧，约49.18GB |
| 派生数据 | 79份、56,261帧，HDF5合计5,985,209字节；图像引用原文件 |
| 完整20点窗口 | 合计54,760个候选窗口，含reserve |
| train | 37条，20,423帧，19,720个完整窗口；stride4时4,945个查询 |
| val | 10条，6,190帧，6,000个完整窗口；stride4时1,503个查询 |
| reserve | 32条，29,648帧，29,040个窗口；本轮不进入统计或模型batch |
| 独立转换核对 | 79/79关节、夹爪、状态、时间及有效窗口精确一致 |
| VDS图像核对 | 每条两路各首帧，共158对派生/原图逐像素一致 |
| 既有ACT接口 | CPU真实batch：图像[2,1,3,180,320]、状态[2,8]、输出[2,20,8]，loss有限 |
| 权重更新 | 0；前向使用新初始化的既有ACT网络，未加载部署checkpoint |
| 当前15维审计 | 原审计仍拒绝，未放宽schema或保护 |

完整结果见[results.json](results.json)。CPU前向及有限loss只证明数据接口能运行，不表示学会了抓放。本次没有完整物理任务、真实部署或位置泛化验收。

## 本视图的精确定义

profile为`real_rgb8_future_feedback_v1`，action_semantics为`future_measured_qpos_plus_recorded_gripper_intent_v1`。

对原始长度N、前视K=2：

```text
派生长度 M = N - K
state[t] = [原始七轴qpos[t], 实测夹爪宽度[t]]
arm_action[t] = 原始qpos[t + K]
gripper_action[t] = 原始action[t, 7]
图像[t] = 原始RGB[t]
查询t的输出块 = action[t : t+20]
```

从原始qpos重建七轴，不再次移动旧action七轴，所以原来已有2帧前视的20份资料不会变成4帧。夹爪意图事件和对应主机录制时间保持原位；它不冒充已收到确认的下发命令。

默认2帧约100毫秒，是本次接口验证的工程选择，不是已证明最优或测得的控制延迟。`--lead-steps`可显式改变；重新生成须使用新的空输出目录，不能叠加前视或覆盖现有资料。

末尾缺少未来反馈的K行被移除。默认dataset只枚举完整20点窗口；显式读取尾部时，不足部分为中性归一化填充值且padding=true，不能参与loss，不复制最后状态当作有效未来动作。

原始单位沿用项目定义：关节rad，夹爪米。主表是主机录制时间，未宣称设备曝光同步。两个标签分别保存`joint_target_feedback_timestamp`和`gripper_intent_timestamp`，不伪造统一的提交时间。

## 原文件和图像如何保留

原始79份HDF5没有修改。派生文件只写小型数值数组和来源信息，RGB使用HDF5 Virtual Dataset引用原文件。

**应通过本目录`FutureStateDataset`或`read_window`读取。** 它们检查原文件存在、大小和mtime；数据集初始化还检查原数值指纹，原图文件缺失或改变会报错。HDF5底层VDS在丢源时可能返回填充值，不能绕过源检查直接使用旧加载器，把零图当作真实画面。

图像未旋转、改色或重新定位；保留原RGB内容。默认只用camera_0（旧ZED左图），camera_1（独立USB相机）也保留引用。训练读取时RGB/255，双线性缩放到180×320，与旧ACT预处理一致。

## 分组与归一化

选择来源为旧`manifest_compatible38.json`与`manifest_fixed_destination_9.json`及其各自split，合计37 train、10 val。其余32份转换后保留为reserve，未判定为物理失败。

只对37条train计算状态和动作mean/std，std下限为旧接口的1e−2；清单中记录全部来源及文件哈希。同一源文件不能跨split或重复计数。**这是旧episode留出，不是独立杯位留出**；不能用本划分宣称位置泛化已通过。旧valid也不等于独立实物成功。

## 与当前15维模型的边界

本次交付的是可供辅助预训练/适配研究使用的8维真实数据视图，不能当作当前15维在线迁移已经完成。现有文件没有可靠记录以下信息：

- 逐帧、与观测配对的6维实时视觉相对几何；
- 曝光前已确认的夹爪命令历史和七轴实际提交/回执；
- 当前RGB-D前端要求的原始XYZ与置信度；
- 设备曝光同步和可核对的标定版本绑定。

这些信息没有补零、改名或用仿真真值替代。构造dataset时要求`policy_input_dim=8`；传15会拒绝，原15维视觉审计也独立验证仍拒绝。当前1a28和已拒绝eec权重均未改动，不能将8维batch直接用于它们的15维输入。

后续如要通过这些真实RGB资料影响主模型，需要明确辅助任务或有限观测适配的训练方案，以及原15维完整行为保持与独立评估；本次不暗中改网络结构或扩大训练预算。

## 使用已生成的训练视图

在本目录，使用已有离线环境：

```bash
cd /home/user/wq/nero_act_real_data_adapter_20261024
/home/user/wq/nero_act_command_increment_20261021/.venv/bin/python -m pytest -q
```

加载器与既有ACT五项batch接口一致：

```python
import json
from torch.utils.data import DataLoader
from adapter import FutureStateDataset

manifest = json.load(open('data/manifest.json'))
stats = json.load(open('data/statistics.json'))
paths = [r['path'] for r in manifest['episodes'] if r['split'] == 'train']
dataset = FutureStateDataset(paths, stats, stride=4)
images, state8, actions20x8, padding20, starts = next(iter(DataLoader(dataset, batch_size=2)))
```

这个例子只读取数据，不训练。需要生成另一份视图或复核时，使用新的目标路径，两个入口均拒绝覆盖：

```bash
/home/user/wq/nero_act_command_increment_20261021/.venv/bin/python prepare.py --help
/home/user/wq/nero_act_command_increment_20261021/.venv/bin/python preflight.py --help

# 换成尚不存在的新目录/报告路径再运行
/home/user/wq/nero_act_command_increment_20261021/.venv/bin/python prepare.py --output /tmp/nero-real-data-new-view --lead-steps 2
/home/user/wq/nero_act_command_increment_20261021/.venv/bin/python preflight.py --manifest /tmp/nero-real-data-new-view/manifest.json --report /tmp/nero-real-data-new-preflight.json
```

## 文件索引

- [adapter.py](adapter.py)：转换、源检查、归一化和ACT数据集。
- [prepare.py](prepare.py)：79份来源核对、旧split及派生清单。
- [preflight.py](preflight.py)：独立数组/图像核对，既有ACT CPU接口及原15维拒绝检查。
- [data/manifest.json](data/manifest.json)、[statistics.json](data/statistics.json)、[training_view.json](data/training_view.json)：实际生成的数据入口、train统计与配置。
- [tests](tests/)：索引、事件、mask、错误数据、覆盖、来源、分组及独立核对测试。
- [progress.md](progress.md)：实际执行与复核记录。
- [final-review.md](final-review.md)：独立整体复核，无待修正问题。
