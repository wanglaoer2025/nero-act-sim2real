"""Exact initialization snapshots, including solver warmstart and delayed target history."""
import mujoco
import numpy as np

SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def capture(model, data, servo, previous):
    state = np.empty(mujoco.mj_stateSize(model, SPEC))
    mujoco.mj_getState(model, data, state, SPEC)
    return dict(physics_spec=SPEC, physics_state=state.tolist(), qpos=data.qpos.tolist(),
                qvel=data.qvel.tolist(), previous_target=np.asarray(previous).tolist(),
                servo_active=servo.active.tolist(), servo_queue=[row.tolist() for row in servo.queue])


def restore(model, data, servo, saved):
    state = np.asarray(saved['physics_state'], float)
    qpos, qvel = np.asarray(saved['qpos'], float), np.asarray(saved['qvel'], float)
    previous, active = np.asarray(saved['previous_target'], float), np.asarray(saved['servo_active'], float)
    queue = np.asarray(saved['servo_queue'], float)
    if not servo.delay and not queue.size:
        queue = queue.reshape(0, 8)
    if (saved['physics_spec'] != SPEC or state.shape != (mujoco.mj_stateSize(model, SPEC),)
            or qpos.shape != (model.nq,) or qvel.shape != (model.nv,)
            or previous.shape != (8,) or active.shape != (8,) or queue.shape != (servo.delay, 8)
            or not np.isfinite(np.r_[state, qpos, qvel, previous, active, queue.ravel()]).all()):
        raise ValueError('Complete finite integration state and matching servo history required')
    mujoco.mj_setState(model, data, state, SPEC)
    if not np.array_equal(data.qpos, qpos) or not np.array_equal(data.qvel, qvel):
        raise ValueError('Snapshot observable values disagree with integration state')
    servo.active = active.copy()
    servo.queue = [row.copy() for row in queue]
    mujoco.mj_forward(model, data)
    return previous.copy()
