## Task 2 标定求解与独立误差验收

**Files:** 创建`calibration/{__init__,solve,bundle}.py`、`tests/test_calibration.py`、`config/calibration_samples.example.json`。

**Interfaces:** `solve_tcp(samples: list[dict]) -> dict`；`solve_camera(base_points_m: ndarray, pixels_rectified: ndarray, K: ndarray) -> dict`；`validate_calibration(solution: dict, holdout: dict) -> dict`；`load_calibration(path: Path, *, require_verified: bool) -> dict`。TCP样本固定为`sample_id,base_from_flange_4x4`，求解输出`tcp_offset_flange_xyz_m,fixed_point_base_xyz_m,rank,residual_m`。报告定义`schema=nero_visual_calibration_v1`、具名坐标变换、相机序列号、图像空间/尺寸、TCP与指面几何、杯口/杯把/桌面、训练与独立样本ID、误差界限、哈希、`verified`及`accuracy_target_met`。

- [ ] 写并运行失败测试：`test_tcp_recovery_and_rank_deficiency`用已知TCP和固定参考点验证毫米精度、拒绝同姿态秩亏；`test_camera_holdout_is_disjoint_and_metric`拒绝样本重用、米毫米错配和错误图像空间；`test_approximate_bundle_cannot_authorize`拒绝135毫米旧粗测、152毫米名义参数冒充实测。
- [ ] 合成无噪姿态断言`np.testing.assert_allclose(solution['tcp_offset_flange_xyz_m'], true_tcp, atol=1e-8)`；`assert report['synthetic'] is True and report['verified'] is False`。
- [ ] TCP使用`R_i*t+p_i=c`最小二乘，至少6组覆盖姿态且矩阵满秩；固定标定夹具/参考点与夹持中心关系须实测。相机使用已知base控制点的PnP和独立点检验，至少6个非共面控制点，检查正深度和刚体变换。求解残差不能代替独立验收。
- [ ] 独立TCP至少3个未参与求解姿态，相机至少9个覆盖源/目标与接近高度的控制点；报告最大误差、RMSE及旋转在指尖处的影响。`accuracy_target_met`仅在独立最大误差满足1/2毫米目标时为真；`verified`另表示真实独立采样、仪器不确定性、外形/桌面和实际误差界限已审核。未达精度目标时记录失败；仅可在按实测误差重新验证的较小工作区及更大间隙中使用，不自动加补偿或宣布达标。
- [ ] 实现CLI`python -m calibration.solve --samples ... --output ...`，默认不连接相机/机器人；示例与合成数据只证明求解器，输出标记`synthetic=true`且不可部署。运行专项测试与`--help`，记录通过。

