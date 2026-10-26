# ACT 七轴增量与尺度限定对照

原v2正式20K已经完成。2K/5K/10K/20K四个检查点均完整0/44，均无GRASP事件；原状态`awaiting_real_data_or_development_result`不代表策略通过。

**本轮已结束：**A/B各1000步，三模型短测均0/3。A小幅接近后触发原始前缀跳变保护；B误差比源20K降低35.49%，但仅接近约3.81mm后停滞，未进入完整抓放。另从B同一停滞物理状态执行的离线教师恢复通过全部8阶段；这不是ACT成功，也不准入训练。详见[实际结果与判断](results.md)。

本目录执行用户要求继续的下一轮离线验证：保留20K核心/视觉/CVAE/夹爪并冻结eval/z0，只训练同容量143→7七轴读出。A输出绝对位置，B输出相对当前曝光配对反馈的增量。不是手工接近、仿真轨迹回放或真机动作。

两组各1000步。读出参数坐标不同，但初始化物理行为与原模型等价；两组新AdamW，不复用不匹配的旧动量。旧20K、上一轮实验、训练数据和所有保护均不修改。

## 已完成的零更新验证

- 120train拟合702950个有效首2点增量；std约0.001～0.005rad，原绝对位置std约0.075～0.316rad。
- 18条代表轨迹668窗口：物理七轴初始化等价最大误差6.03e-7rad，夹爪误差0。
- 原失败上下文query0/975/1949：原预测重放误差0，A/B新读出等价误差不超过4.52e-7rad，夹爪误差0。
- 上次97.86GiB完整SHA证明、原清单及177路径dev/inode/size/mtime/ctime均匹配，复用元信息连续性证据；本轮没有重新声称全量字节重读。
- TDD专项11项通过，覆盖因果增量/还原、无cumsum、padding、只读出有梯度、grip等价、快照被改拒绝、预算防重复和短测不可冒充完整成功。

## 实际进度和入口

`execution.json`、`training/A/status.json`、`training/B/status.json`是本轮动态进度；`training_completion.json`证明实际预算完成；`evaluation/progress.json`和`results.json/results.md`记录实际物理诊断结果。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl OMP_NUM_THREADS=2 \
  nero_act_transfer_v2_20261025/.venv/bin/python -u \
  nero_act_arm_increment_20261026/run.py
```

已有`training`时入口拒绝重复启动。源码/权重/数据证明绑定与冻结核心在每100更新复核。原20K的pipeline_state不属于本次支线进度。

两组及源20K各做3次600点接近诊断；到达高位目标10mm内且保护完整，才进入原3900点中心完整抓放，再由一个完整中心3/3候选运行原44项开发验收。诊断截断不代表完整超时，也不是完整通过；独立40次最终测试不消耗。

限制：本轮共同增加当前观察直连，冻结完整核心；它检验的是受限读出坐标/尺度方案，不能把失败推断为所有增量ACT都无效，或把成功直接宣称真实泛化。未启动ROS/CAN或真机。

计划：`../docs/superpowers/plans/2026-10-26-nero-act-arm-increment.md`；实际决策：`progress.md`。
