# NERO 15维ACT完整闭环验证与真机组件

**完整16项名义仿真最终5/16通过，尚不具备本版本真机执行条件。** 成功场景完成抬升、搬运、承托放置、松爪和退出；其余仍有停滞和保护停止。

| 结果类别 | 数量 |
| --- | ---: |
| 完整成功 | 5 |
| 点预算耗尽，夹持/抬升停滞 | 7 |
| 意外接触停止 | 2 |
| 持杯模型桌面保护停止 | 1 |
| 延迟观测早于上次提交，时效保护停止 | 1 |

9个杯位中3个成功，4个起点中2个成功，最后3项均失败。16个ID在训练前冻结，但最后3项的数值扰动在训练后补齐，明确不是严格预声明的真实部署资格。TCP、杯体和相机仍为保留名义参数。

- [54项问题与解决办法](../docs/nero_act_sim_to_real_issues_and_validation_20261019.md)
- [逐项效果与失败分类](evidence/full_process_assessment.json)
- [冻结16项原始汇总](results/frozen_320_suite16/summary.json)
- [完整成功过程视频](evidence/ACT_visual15_full_success_simulation_2x.mp4)
- [视频来源及显示范围](evidence/success_video.json)

视频按记录的ACT查询状态以2倍速显示，间隔用已有画面保持；不重新积分生成控制，不连接真机。它展示一例成功，不代表16项全部成功或真机通过。

## 模型与观测

模型读取原工程最佳检查点并校验训练完整性，输入RGB及15维数值，输出20×8绝对目标。新增6维由同帧图像名义杯轮廓和蓝卡估计，不用物体真值作为策略输入。每次最多提交2个50毫秒点，近抓取区可为1点，总有效点预算保持3900；阶段确认是仿真监督，不是现场授权。

320×180及160×90开发场景均0/1，仅到GRASP。320模型末段开度与前2点目标差约1.34～1.83毫米，不满足原3毫米稳定夹持差值。采样审计发现36个训练来源中2232个差值≥3毫米close帧仅30个进入采样；这是具体监督缺口，尚未通过修正训练证明为唯一根因。

## 真机软件进度

- `executor/visual_inference.py`：同帧相机/反馈/几何/因果历史的15维推理，输入内容、时刻和校准版本绑定。
- `executor/visual_dispatch.py`：等待推理时持续健康检查；碰撞后重验新鲜度；1～2点ROS提交、ACK和双控制器结果监督；停止、取消与只读恢复回调。
- `executor/visual_control.py`：原已测试的短前缀状态机及ACK后历史提交。

这些组件尚未组装成合格的真实相机、语义前端、完整阶段runner、一次性授权消费及驱动恢复链路。`executor.visual_run`仍为离线入口并拒绝执行。本轮未发机械臂、夹爪或使能指令。

## 复核命令

以下仅仿真或离线测试，输出需选择新目录。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/nero_visual_mpl .venv/bin/python -m pytest -q

MUJOCO_GL=egl .venv/bin/python -m simulation.visual_evaluation \
  --training-run ../nero_act_visual_feedback_20261019/training/visual_retained_v1_320 \
  --nominal-only --output results/NEW_NOMINAL

MUJOCO_GL=egl .venv/bin/python -m simulation.visual_suite \
  --training-run ../nero_act_visual_feedback_20261019/training/visual_retained_v1_320 \
  --workers 4 --output results/NEW_SUITE16
```

独立审查发现的观测延迟、套件几何冻结和预完成future时效漏洞均经RED→GREEN修正。最终280项测试通过、5项明确跳过，见`evidence/full_suite_after_freeze.log`。阶段核验见`evidence/final_verification.json`。旧训练源码和检查点保持不变。
