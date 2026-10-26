# SDD ledger — plan: /home/user/wq/docs/superpowers/plans/2026-10-23-nero-act-observation-repair.md

用户已回复继续，按已确认方案顺序实施，不重新索要运动授权。本阶段只做离线诊断与只读观测。
Pre-flight: 正常与真值组必须共用图像/蓝卡估计/反馈/保护；采用原SimulationHistory采样并逐一绑定pending曝光快照，避免当前真值混入延迟帧。
Ruling: 将上一轮七个场景全部视为开发回归，真值替换明示diagnostic-only且qualification=false；新独立验收留到观测修复和训练后另封存 — 如判断错误，结果也不能进入真实部署许可。

Task1 unit: 5/5通过，覆盖只改杯位、拒绝正常模式、成功不算验收、两个延迟曝光各自真值绑定、禁止正常运行时误用。RED为明确缺失模块断言；修正测试模型多关节夹在同body的夹具错误后整套GREEN。

Ruling: 几何观测使用原始1280×720，同曝光ACT RGB仍为320×180 — 首个边界拟合在320图上因量化误差未达到2毫米目标，需按既定原分辨率要求验证而不降低误差门槛。原生渲染初次因默认640宽帧缓冲拒绝，0帧；仅扩大visual framebuffer，物理与相机参数不变，启动日志保留。

Task1: complete — 正常9005六类数组与基线完全一致；9000/9005/9201/9300/9301准确杯位诊断0/5完整，禁止计入正常验收；来源实际模块/资产/模型均封存于results/summary.json。
Task2 RGB: 未达到全帧误差门槛，保留失败结果observer_results.json，不替换正常观察器。
Task2 depth: 只读NEURAL_LIGHT新曝光20/20，有效共享9492像素，静止Z标准差95分位0.163825毫米；重复性不算绝对精度。重复时间戳跳过且时钟回退拒绝测试RED→GREEN。
Ruling: 在本阶段新增RGB-D离线候选，但不更改正常策略/训练/真机入口 — 原生RGB候选失败而已保存深度可用，采用同曝光深度和可见首交点消除内外壁歧义；如果名义模型不适用于真实杯，会显式拒绝或存在偏差，因此真实精度和部署仍false。
Task2 RGB-D: RED为缺失候选；首版最近表面法遮挡测试2.91毫米失败；加入沿相机射线首交点约束后原测试GREEN。另补SDK向量不重复旋转、裁边反光拒绝和双完整目标歧义测试，均RED→GREEN；共23项实验测试通过。
Task2 frame audit: ideal开发14/14、留出21/22；合成噪声开发13/14、留出21/22，所有有效帧≤2毫米，机制通过。95分位耗时415～842毫秒，实时性未通过。
Task2 real frontend: 已保存实景0/9/19三帧全部在原蓝卡轴对齐模型的解释性检查处被拒绝，0/3就绪。未降低门槛，未要求现场重复测量；此问题及观测/训练一致性作为后续明确工作。
Task2: complete as offline mechanism and read-only feasibility audit — 不含完整抓放、实时部署、示教重写或新训练承诺。近似监督冲突审计7/8，原样本未改，补充代码哈希后另保留初次审计。
Task3: running — 实验23/23、原保护回归13/13；旧工程完整pytest执行中，随后只做一次独立最终审查和证据封存。

Final review: fresh-context只读 /root/observation_repair_final_review，两项Important，详见final_review.md。原23项复核独立通过；源哈希与正常9005六类数组一致。
Final: fixed 有符号深度残差漏检 — test_one_sided_depth_outliers_cannot_pass_unsigned_quality_gate 未修正版明确误接收RED，rgbd_observer_v2取绝对值后GREEN；原v1源/证据保持，72项帧审计输出v2。
Final: fixed 同ROI轴向/横向运动被误记静止 — test_axial_surface_motion_cannot_be_called_static_from_unchanged_roi和test_lateral_surface_motion_is_also_rejected_when_roi_stays_the_same均RED→GREEN；depth_quality_v2增加共享XYZ时间变化门槛及stationary_scene_verified=false，小静态噪声仍通过。
Final: suite37/37，含两组v2沿用原行为的10项回归；旧工程完整443passed5skipped1warning，保护13/13。新增capture_depth_quality_v2只检查--help，未重开相机。
Final correction: v2更严格后9005/query593压力帧绝对残差2.191毫米被拒绝，开发压力组12/14<90%，最终mechanism_passed=false；上文v1整体通过为历史结果，不能继续计当前通过，不扩大训练预算。
Final declined-to-judge disposition: 原20帧只留3原始数组，只对0/9/19按v2复算（0.229毫米、稳定性启发式而非独立静止证明），未宣称重算20帧；真机运动、时钟、绝对标定/TCP和完整ACT均不在本离线阶段验收，继续false。无延期minor；不派复审。

Task3: complete — 最终实验37/37、保护13/13、旧工程443/5skip/1既有warning；单次独立复核两项重要问题RED→GREEN，453份绑定来源无变化；最终v2机制未整体通过（压力12/14），真实前端0/3、训练0、运动0，全部限制已交付README与verification.json。
