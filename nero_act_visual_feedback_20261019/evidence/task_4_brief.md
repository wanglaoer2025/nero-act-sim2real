## Task 4 统一15维特征与几何误差预算

**Files:** 创建`executor/visual_observation.py`、`tests/test_visual_features.py`；修改`calibration/bundle.py`、`executor/execution_guards.py`。

**Interfaces:** `relative_positions(state: ndarray, geometry: GeometryObservation, robot_geometry: RobotGeometry) -> ndarray[6]`；`require_visual_observation(frame, geometry, paired_feedback, *, now_s: float, calibration: dict) -> None`；`required_clearance_m(bounds: dict, *, speed_m_s: float, detection_s: float, stopping_s: float) -> float`。误差预算为TCP、视觉、外形、跟踪保守界限之和，再加上速度×检测/停车时长及2毫米额外余量，下限5毫米；这里“速度×时长”使用实测的最大扫掠速度界限，非仅TCP平移速度。

- [ ] 写并运行失败测试`test_relative_positions_are_base_metric_and_current_tcp`、`test_geometry_stats_use_train_only`、`test_stale_invalid_or_wrong_calibration_is_rejected`、`test_clearance_adds_errors_and_stop_distance`。相机/状态任一未来时间、过期、误标坐标、完全遮挡或缺失停车距离证据均拒绝。
- [ ] 测试使用具名bounds`tcp_m,vision_m,shape_m,tracking_m`，断言`assert required_clearance_m(bounds, speed_m_s=.01, detection_s=.1, stopping_s=.1) == pytest.approx(.009)`，各误差依次1/2/1/1毫米。
- [ ] 计算`[cup_bottom-current_tcp,target-current_tcp]`，不输入真值或手写动作目标。杯把朝向在已测范围外时拒绝；当前夹爪几何由开度及实测模型决定，FK不能使用旧TCP。
- [ ] 误差预算中未实测项不填零、无穷大不变成默认通过；测试1毫米TCP+2毫米视觉+1毫米外形+1毫米跟踪、速度0.01m/s和总停车0.2s时要求9毫米。不同位姿旋转影响折算到包络后计入。
- [ ] 专项测试通过，给Task5/7/10固定schema和质量检查入口；旧回放的固定5毫米规则不变。

