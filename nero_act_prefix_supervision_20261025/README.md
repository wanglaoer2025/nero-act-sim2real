# ACT 执行前缀监督对照

用户已批准在原20K任务外执行 A/B 各1000步的限定离线对照。目标是验证七轴前2点监督是否不足，不预设改动有效，不冒充已完成抓放。

**已完成：**A/B各1000步，匹配初值/优化器/首批预测/全部批次。B接近前缀误差比A下降25.40%，但源5K/A/B的接近短测均0/3，末端仍在约480mm高位；未进入完整抓放/44项验收。详见[结果与判断](results.md)和`final_verification.json`。

- 源模型：`../nero_act_transfer_v2_20261025/training/formal/policy_5000.ckpt`。
- A：原20点有效七轴L1；B：有效前2点与有效后18点均值各50%，后缀全padding只取前缀。
- 同一模型、AdamW动量、归一化、120条训练轨迹、批次顺序、新随机种子20261025、冻结范围及学习率。旧检查点未保存RNG，不能宣称复现原20K的后续随机轨迹。
- 夹爪/CVAE/KL/15维输入/20×8输出及实际前2点执行保持。原训练封存代码只读，未发送真机命令。
- `diagnosis_5k/report.json`：零更新复核；接近前缀MAE0.00385035rad，教师动作0.00268085rad，闭环仍停高位。

## 入口和结果

已启动一次：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl OMP_NUM_THREADS=2 \
  nero_act_transfer_v2_20261025/.venv/bin/python -u \
  nero_act_prefix_supervision_20261025/run.py
```

不要重复运行；已存在`training`即拒绝，不能借重启追加预算。命令日志保存`execution.log`。

- `execution.json`：本对照任务所处阶段。
- `training/A/status.json`、`training/B/status.json`：实际新增更新次数。
- `training/*/initialization.json`：模型和优化器初值摘要。
- `training_completion.json`：两组实际1000步及同批次/初值核对。
- `evaluation/progress.json`：物理诊断和完整评估进展。
- `results.json`、`results.md`：全部完成后生成，不以启动记录替代结果。
- `source_binding.json`：本次源码、原v2封存文件及源检查点/数据清单摘要；训练中持续核对。

先比较相同18条代表轨迹668窗口的分阶段部署误差，再对源5K/A/B各做3次600点接近诊断。短测明确不是完整验收；仅接近目标10mm内、无非诊断错误/意外杯碰/合同拒绝时进入原3900点完整中心抓放。中心3次完整通过后，一个候选才运行原44项开发验收。不消耗独立40项，不自动追加训练或改成七轴增量输出。

## 软件验证

14项专项测试通过（`all_tests.log`），包括padding/NaN、初始推理和非七轴损失等价、优化器恢复且不污染共享源、随机数复现、数据及依赖的字节/快照验证、输出防覆盖和短测不能冒充完整通过。各项RED日志保留。训练和MuJoCo结果须另行读取，软件测试通过不代表策略成功。

首次启动因严格CUDA确定性不支持位置编码cumsum，在第1次更新前停止，零更新证据保存在`attempts/strict_determinism_failed`。现使用相同种子/CUBLAS及确定性cuDNN，A/B首批随机预测和非七轴损失必须完全匹配，不宣称所有GPU算子严格位级确定性。启动前重新验证150条数据和全部声明依赖的SHA，进度见`data_verification_progress.json`。

本轮无ROS/CAN/真机动作、无重新测量。完整实施计划见`../docs/superpowers/plans/2026-10-25-nero-act-prefix-supervision.md`，实际决策与进度见`progress.md`。
