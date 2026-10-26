# 离线复现入口

工作目录：`/home/user/wq/nero_act_contact_recovery_20261020`。依赖本机共享数据路径和`.venv`，不是可独立拷贝的数据包。GPU/EGL命令需可访问宿主GPU的环境；这些命令不连接真机。

已生成的权重、评估和封存文件禁止覆盖；训练及结果路径必须不存在。重复试验应建立新目录并重新封存。以下记录本轮实际命令，供核对，不表示应在既有目录直接再次运行。

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/nero_recovery_mpl
export MUJOCO_GL=egl

.venv/bin/python -m pytest -q

.venv/bin/python -m training.history_seal \
  --data-dir training_data/contact_complete_v1 \
  --initializer /home/user/wq/nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt \
  --verification evidence/recovery_verified_sources.json

.venv/bin/python -u -m training.history \
  --data-dir training_data/contact_complete_v1 \
  --initializer /home/user/wq/nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt \
  --config config/history_experiment.json --variant A --stage pilot \
  --output training/pilot_A

.venv/bin/python -u -m training.history \
  --data-dir training_data/contact_complete_v1 \
  --initializer /home/user/wq/nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt \
  --config config/history_experiment.json --variant B --stage pilot \
  --output training/pilot_B

.venv/bin/python -u -m simulation.history_mechanism \
  --training-run training/pilot_A --captures training_data/recoveries_v4 \
  --output results/mechanism_A

.venv/bin/python -u -m simulation.history_mechanism \
  --training-run training/pilot_B --captures training_data/recoveries_v4 \
  --output results/mechanism_B
```

## 证据链

- `recovery_verified_sources.json`：388通过/5跳过及测试前后源码一致。
- `pre_optimization_freeze.json`：优化前161个源码/资产文件、172个数据/配置输入哈希，旧源码保留。
- `training_data/contact_complete_v1/data_acceptance.json`：67条、37/12组、训练统计及同批次。
- `training/pilot_{A,B}/artifacts/training_integrity.json`：实际预算、输入未变、最终权重哈希。
- `results/mechanism_{A,B}/summary.json`：原六项机制，不是完整高位起点任务验收。
- `final_verification.json`和`repair_results.md`：本轮最终物理结果和限制。
