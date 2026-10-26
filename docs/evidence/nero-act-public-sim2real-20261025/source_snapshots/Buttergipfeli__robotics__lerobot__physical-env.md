# Simulation environment

MuJoCo model of an SO-101 arm that picks up a stress ball from a cork mat and
places it on top of a toilet paper roll, observed by a single wrist camera.

## Physical scene

| Part | Dimensions |
| --- | --- |
| Cork mat | 250 mm deep x 300 mm wide, 3 mm thick, speckled cork texture, near edge 72 mm in front of the arm's base foot |
| Platform under the arm | 237 mm wide x 167 mm deep, 12 mm high, front edge flush with the mat |
| Stress ball | 66.8 mm diameter, 45 g, light blue foam with dark print, free body |
| Toilet paper roll | 121 mm outer diameter, 44.6 mm core hole, 96 mm high, white paper with grey core, static |

The arm faces the **short** side of the mat, so its 300 mm run left to right.
Ball and roll are placed randomly on the mat each episode: the roll fully on
the mat with 10 mm margin, the ball at least 115 mm from the roll centre, and
both clear of the folded arm.

## Task and success

The ball is larger than the core hole, so it seats on the opening like an egg
in a cup. An episode succeeds when the ball centre stays within 15 mm of the
roll axis, above the rim, and nearly still for 10 consecutive ticks (1 s).
It fails when the ball leaves the mat at ground level or after 30 s.

## Arm

Six joints, restricted to the travel the follower's own calibration reports
rather than the MJCF limits:

| Joint | Range |
| --- | --- |
| shoulder_pan | 167° |
| shoulder_lift | 187° |
| elbow_flex | 165° |
| wrist_flex | 156° |
| wrist_roll | 320° |
| gripper | 98° |

One servo tick is **0.088°** (4096 steps per turn) and LeRobot truncates anything
finer.

Episodes start from a folded rest pose (keyframe `rest`): shoulder fully back
(-100°), elbow and wrist folded in, wrist rolled so the camera faces up, and
the gripper closed. Actions and states use LeRobot's normalised units
(-100 to 100 per joint, 0 to 100 for the gripper).

## Camera

A single standard camera mounted on the follower's wrist, in the same position
as on the SO-101 follower arm described in the NVIDIA Isaac sim-to-real guide:
640 x 480, 70° nominal vertical field of view. The mount (riser, tilted plate,
lens) is modelled with collision geometry. There is no screen capture, the
camera frame is the only image observation.

## Grasp assist

The rigid claw cannot wrap a rigid 67 mm sphere the way the real foam ball
deforms, so the sim uses a weld constraint as grasp assist: when the gripper
command closes and the ball is within 50 mm of the canonical hold point, the
ball is welded into a deep, physically plausible grip (jaws flanking the
equator). Opening the gripper releases the weld. The same rule applies to the
expert and to any policy at evaluation time.

## Observation and control

- one 640 x 480 wrist-camera frame per tick
- six joint values in the follower's normalised units
- control at 10 Hz, physics at a 2 ms timestep, episodes up to 30 s

## Per-episode randomisation

| Quantity | Range |
| --- | --- |
| Mat position | ±10 mm in x and y |
| Ball sliding friction | 0.8 to 2.5 |
| Ball mass | ±10% (inertia scaled) |
| Camera position | ±8 mm per axis |
| Camera tilt | ±0.05 rad about two axes |
| Camera fovy | 64 to 76° |
| Light position / intensity | ±30 cm / 0.5x to 1.3x |
| Mat shade | 0.7x to 1.2x |
| Roll paper shade | 0.85x to 1.02x |
| Floor shade | 0.3x to 1.1x |
| Ball colour | hue ±0.04, saturation 0.5x to 1.3x, value 0.6x to 1.15x |
| Ball markings | 0 to 6 dark ellipses, optional light label patch and dark dot, drawn into the ball texture |
| Camera exposure / gamma | 0.35x to 1.3x / 0.8 to 1.3 |
| White balance | ±10% per channel |
| Sensor noise / blur | Gaussian, sigma up to 8/255 / up to 1.2 px |

Ball and roll positions are re-sampled every episode as described above.
Randomisation runs by mutating the compiled model at reset time, so no
recompilation is needed; the ball texture is repainted and re-uploaded to the
renderer, and exposure, gamma, white balance, blur and noise are applied to
every rendered frame while randomisation is on. Nominal colours and light were
tuned against a real wrist-camera frame of the ball, mat and roll.

## Real robot joint mapping

The real follower reports joints in its own calibration units (recorded range
mapped to ±100), the sim in MJCF-range units. `lerobot/real/joint_mapping.py`
converts at the robot boundary: real units → encoder ticks → degrees from the
calibration middle → MJCF degrees (per-joint `JOINT_OFFSETS_DEG` in
`real_config.py`) → sim units, and back for actions. Policies, `REST_ACTION`
and everything else in the real scripts are in sim units.

The offsets are the MJCF angle of the pose held as calibration middle, so they
belong to one calibration and must be re-measured after recalibrating: put the
arm by hand into the sim rest pose, read the raw encoder ticks and compare the
degrees from the calibration middle with the rest pose angles. For
shoulder_lift and elbow_flex the mechanical stops coincide with the MJCF
limits, so their offsets follow from the calibration ranges alone.
