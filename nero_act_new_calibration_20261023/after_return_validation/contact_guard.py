"""Reject a pregrasp path using the full nominal collision model, never create targets."""
import math

import mujoco
import numpy as np

from simulation.scene import build_scene


class PregraspSweptGuard:
    def __init__(self, scene):
        self.model = build_scene(scene)
        self.data = mujoco.MjData(self.model)
        self.cup = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cup')
        self.base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self.table = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'table')

    def validate(self, current, proposed, *, close_authorized):
        current, proposed = np.asarray(current, float), np.asarray(proposed, float)
        if (current.shape!=(8,) or proposed.ndim!=2 or proposed.shape[1]!=8
                or not len(proposed) or not np.isfinite(current).all() or not np.isfinite(proposed).all()):
            raise ValueError('Finite current8D and nonempty proposed Nx8 prefix required')
        if close_authorized:
            # The existing dynamic attached-cup and contact guards own later phases.
            return 0
        checked = 0
        previous = current
        for target in proposed:
            count = max(1, math.ceil(float(np.max(abs(target-previous)))/.0005))
            for alpha in np.linspace(0., 1., count+1):
                state = previous + alpha*(target-previous)
                self.data.qpos[:7] = state[:7]
                self.data.qpos[7:9] = [state[7]/2, -state[7]/2]
                mujoco.mj_forward(self.model, self.data)
                checked += 1
                for contact in self.data.contact[:self.data.ncon]:
                    geoms = {int(contact.geom1), int(contact.geom2)}
                    bodies = {int(self.model.geom_bodyid[g]) for g in geoms}
                    if self.cup in bodies and self.table not in geoms:
                        raise RuntimeError('Predicted pregrasp cup contact; no ACT target sent')
                    if self.cup not in bodies and bodies!={0,self.base} and contact.dist < -1e-6:
                        raise RuntimeError('Predicted robot/environment contact; no ACT target sent')
            previous = target
        return checked
