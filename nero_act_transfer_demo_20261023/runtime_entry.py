"""Use the installed system numeric stack and the existing robot SDK, without installs."""
import sys
from pathlib import Path
import runpy

import numpy
from scipy.spatial.transform import Rotation
import rclpy

sys.path.insert(0,'/home/user/nero/lib/python3.12/site-packages')
sys.path.append('/home/user/wq/nero_act_command_increment_20261021/.venv/lib/python3.12/site-packages')
import mujoco
import pyAgxArm

if __name__=='__main__':
    print(f'Runtime ready: NumPy {numpy.__version__}, MuJoCo {mujoco.__version__}; hardware not connected yet',flush=True)
    runpy.run_path(str(Path(__file__).with_name('entry.py')),run_name='__main__')
