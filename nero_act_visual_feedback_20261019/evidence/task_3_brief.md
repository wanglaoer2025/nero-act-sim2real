## Task 3 相机与状态的实时只读观测

**Files:** 创建`vision/{__init__,observation,synchronization}.py`、`tests/test_visual_observation.py`、`tests/test_visual_sync.py`；修改`executor/camera.py`。

**Interfaces:** `CameraFrame`含`frame_id,rgb,exposure_monotonic_s,received_monotonic_s,camera_metadata_hash`；`GeometryObservation`含`frame_id,cup_bottom_base_m,target_base_m,cup_handle_yaw_rad,quality,error_bound_m,calibration_sha256,valid,invalid_reason`。`VisualObserver.measure(frame: CameraFrame) -> GeometryObservation`；`upright_to_optical(pixel_xy: tuple, image_size_wh: tuple) -> tuple`放在`vision/observation.py`；`pair_feedback(frame: CameraFrame, feedback: list[dict], *, max_skew_s: float=.05) -> dict`返回该帧配对的8维状态和时间误差。

- [ ] 写失败测试`test_unrotation_and_rectified_intrinsics_are_paired`、`test_cup_handle_and_occlusion_are_not_valid_centres`、`test_pair_rejects_51ms_skew_repeated_future_frame`；包括黄杯把、夹指遮挡、蓝色手机干扰、多目标和无目标，不能按最高置信度猜目标。
- [ ] 像素映射断言`assert upright_to_optical((0,0),(640,360)) == (639,359)`；同一映射不得做两次。遮挡样本断言`assert observation.valid is False`，51ms配对须抛`ValueError`。
- [ ] 图像旋转180°只在显示/模型预处理做一次，几何像素明确映回rectified光学图；每次相机启动保存实际K、尺寸、图像空间和时间映射。SDK曝光时间戳若不能可靠映射单调时钟，标记未满足在线时序，不伪造为接收时间。
- [ ] 从现有YOLO黄杯检测、颜色轮廓和蓝卡检测改造观测入口：分别拟合杯身/杯口并识别杯把，误差界限来自独立标定；不靠固定75/95毫米轮廓吃掉杯把。不能区分遮挡/杯身时`valid=false`。深度作为单独比较，不默认开启或替代未测高度。
- [ ] 用保存帧及合成像素验证上述测试，再实现`python -m vision.observation --readonly --calibration ... --output ...`，无动作客户端/SDK发送。相机设备采集仅在可用时只读运行，前后记录CAN TX；没有现场可见性样本时记录该项未验收。

