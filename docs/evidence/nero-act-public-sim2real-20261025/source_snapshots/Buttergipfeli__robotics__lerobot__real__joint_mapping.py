import numpy as np

from real_config import JOINT_NAMES, JOINT_OFFSETS_DEG

TICKS_PER_TURN = 4096
TICKS_MIDDLE = 2048
MJCF_JOINT_RANGES_RAD = {
    "shoulder_pan": (-1.9198621771937616, 1.9198621771937634),
    "shoulder_lift": (-1.7453292519943224, 1.7453292519943366),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.6580628494556928, 1.6580627293335335),
    "wrist_roll": (-2.7438472969992493, 2.841206309382605),
    "gripper": (-0.17453297762778586, 1.7453291995659765),
}
NORM_LO = np.array([-100.0, -100.0, -100.0, -100.0, -100.0, 0.0])
NORM_SPAN = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 100.0])
RANGE_MARGIN = 4.0


class JointMapping:
    def __init__(self, calibration):
        self.ticks_min = np.array([calibration[j].range_min for j in JOINT_NAMES], dtype=np.float64)
        self.ticks_max = np.array([calibration[j].range_max for j in JOINT_NAMES], dtype=np.float64)
        ranges = np.degrees([MJCF_JOINT_RANGES_RAD[j] for j in JOINT_NAMES])
        self.mjcf_lo, self.mjcf_hi = ranges[:, 0], ranges[:, 1]
        self.offsets = np.array(JOINT_OFFSETS_DEG, dtype=np.float64)

    def real_to_sim(self, real):
        ticks = self.ticks_min + (np.asarray(real) - NORM_LO) / NORM_SPAN * (self.ticks_max - self.ticks_min)
        mjcf_deg = (ticks - TICKS_MIDDLE) * 360.0 / TICKS_PER_TURN - self.offsets
        return NORM_LO + (mjcf_deg - self.mjcf_lo) / (self.mjcf_hi - self.mjcf_lo) * NORM_SPAN

    def sim_to_real(self, sim):
        mjcf_deg = self.mjcf_lo + (np.asarray(sim) - NORM_LO) / NORM_SPAN * (self.mjcf_hi - self.mjcf_lo)
        ticks = TICKS_MIDDLE + (mjcf_deg + self.offsets) * TICKS_PER_TURN / 360.0
        real = NORM_LO + (ticks - self.ticks_min) / (self.ticks_max - self.ticks_min) * NORM_SPAN
        return np.clip(real, NORM_LO + RANGE_MARGIN, NORM_LO + NORM_SPAN - RANGE_MARGIN)
