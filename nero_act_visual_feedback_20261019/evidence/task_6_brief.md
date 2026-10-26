## Task 6 真实外形场景与完整示教

**Files:** 修改`simulation/{scene,teacher,collect_generalized,refresh_observations}.py`、`simulation/scene_template.xml`、`training/pose_objective.py`；修改`executor/ros_execution.py`的`RobotGeometry`与`cup_object`；创建`tests/test_measured_scene.py`。

**Interfaces:** `scene_from_calibration(calibration: dict, scenario: dict) -> dict`位于`simulation/scene.py`；`cup_collision_parts(calibration: dict) -> list[dict]`位于`calibration/bundle.py`，输出以杯底坐标表示的杯身、杯口及杯把碰撞体，两侧MuJoCo/MoveIt共用。旧模型只在legacy profile保留固定75/95毫米断言；新profile要求已验证实测包。

- [ ] 写失败测试`test_handle_collision_rejected_before_body_contact`、`test_real_dimensions_match_mujoco_and_moveit`、`test_nominal_tcp_cannot_generate_deployment_data`：复现旧圆柱有空隙但新杯把/杯口相交的案例，整臂与夹爪基座也不得漏检。
- [ ] 固定几何夹具断言`assert old_body_gap_m > .005 and measured_handle_gap_m < 0`，新间隙检查必须拒绝；两侧几何尺寸/局部位姿用`np.testing.assert_allclose(..., atol=1e-9)`比较。
- [ ] 由标定包生成一致外形和TCP，保持低力物理接触、无粘附/物体绑定，质量/摩擦未测量时保留假设并纳入评估变化。新几何不改动旧封存场景。
- [ ] `fk_spec_from_scene`及姿态损失使用同一实测场景生成的TCP/指面坐标；若实测TCP包含旋转，训练FK与执行FK均表达它，不沿用“TCP site必须无旋转”的旧限制。比较训练FK、MuJoCo和执行FK的独立姿态样本，禁止使用源配置中的旧外部XML路径计算新姿态损失。
- [ ] 在48个既有场景定义上重新采集36训练/12验证完整示教，记录几何、视觉有效性、20Hz实际目标与反馈；保留失败、不计入48，不通过则停止数据发布。杯把朝向按已测范围采样并记录。
- [ ] 全部通过精确重放、全动作合同、误差预算间隙及完整抓放，再发布新manifest。局部下降接触、闭爪、确认后恢复及释放退出数据单独登记来源，不能冒充完整示教。

