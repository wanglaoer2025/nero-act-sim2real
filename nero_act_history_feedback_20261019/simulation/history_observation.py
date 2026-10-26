"""History sampling at completed-prefix exposures, including supervised holds."""
from collections import deque
import copy
import numpy as np

from executor.observation_history import HistoryBuffer,lookup_receipt
from executor.model import make_policy_observation


class SimulationHistory:
    def __init__(self,config,stats,scenario,rng,previous):
        self.config,self.stats,self.rng=config,stats,rng
        self.scope=f'simulation:{scenario["seed"]}'
        self.delay_s=scenario.get('observation_delay_steps',0)*.05
        if scenario.get('observation_delay_steps',0) not in (0,1):raise ValueError('Undeclared image delay')
        self.jitter=bool(scenario['jitter'])
        self.buffer=HistoryBuffer();self.buffer.clear(epoch=0,after_s=-1.)
        self.epoch=0;self.receipts=[];self.pending=deque();self.cached=None
        # Explicit simulation servo initialization, never a real driver receipt.
        self.record_prefix(np.asarray(previous)[None],-1e-9)

    def record_prefix(self,rows,stamp):
        self.receipts.append(dict(receipt_id=f'{self.scope}:{len(self.receipts)}',rows=np.asarray(rows).tolist(),
            sent_s=float(stamp),ack_s=float(stamp),accepted=True,kind='submitted_prefix',scope=self.scope))

    def capture_eligible(self,data):
        stamp=float(data.time)
        if stamp<=self.buffer.after_s:return
        if self.pending and stamp<=self.pending[-1]['stamp']:raise ValueError('Repeated completed-prefix exposure')
        self.pending.append(dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),stamp=stamp))

    def sample(self,model,data,renderer,observer):
        import mujoco
        while self.pending and self.pending[0]['stamp']+self.delay_s<=float(data.time)+1e-9:
            captured=self.pending.popleft();stamp=captured['stamp']
            render_data=mujoco.MjData(model);render_data.qpos[:]=captured['qpos'];render_data.qvel[:]=captured['qvel']
            render_data.time=stamp;mujoco.mj_forward(model,render_data)
            renderer.update_scene(render_data,camera='calibrated_nero_camera')
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            image=renderer.render().copy()
            observed=np.r_[render_data.qpos[:7],render_data.qpos[7]-render_data.qpos[8]].astype(np.float32)
            if self.jitter:observed[:7]+=self.rng.uniform(-.0002,.0002,7)
            features,record=observer(image,observed)
            receipt=lookup_receipt(self.receipts,stamp);previous=float(receipt['rows'][-1][7])
            normalized=make_policy_observation(observed,self.stats,self.config,previous,geometry_features_m=features)
            self.buffer.append(dict(observation=normalized,frame_id=int(round(stamp/.05)),exposure_s=stamp,
                feedback_s=stamp,admitted_s=float(data.time),receipt_id=receipt['receipt_id'],receipt_ack_s=receipt['ack_s'],
                epoch=self.epoch,calibration_sha256=self.config.get('calibration_sha256','nominal-simulation-only'),
                camera_sha256='frozen-simulation-camera-v1',session_id=self.scope,admission_valid=True,prefix_complete=True))
            record.update(exposure_simulation_s=stamp,consumed_simulation_s=float(data.time),
                paired_feedback_state=observed.tolist(),causal_previous_gripper_target_m=previous)
            self.cached=dict(image=image,observed=observed,features=features,observation_stamp=stamp,
                             context_width=previous,record=record)
        return self.cached

    def confirmation_received(self,stamp):
        self.epoch+=1;self.buffer.clear(epoch=self.epoch,after_s=float(stamp))
        self.pending.clear();self.cached=None

    def export(self):
        return copy.deepcopy(dict(items=list(self.buffer.items),epoch=self.epoch,after_s=self.buffer.after_s,
            stopped=self.buffer.stopped,pending=list(self.pending),cached=self.cached,receipts=self.receipts))

    def restore(self,saved):
        self.epoch=saved['epoch'];self.buffer=HistoryBuffer()
        self.buffer.clear(epoch=self.epoch,after_s=saved['after_s'])
        for item in saved['items']:self.buffer.append(item)
        self.buffer.stopped=saved['stopped'];self.pending=deque(saved['pending'])
        self.cached=saved['cached'];self.receipts=saved['receipts']
