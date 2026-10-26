"""Same-frame15D worker core, for a qualified camera/frontend supplied by caller.

No robot interfaces, trajectory teacher or simulator truth are imported. The
caller runs this worker on a background thread while the ROS thread checks health.
"""
import hashlib
import copy
import json
from dataclasses import asdict
import time

import numpy as np

from executor.model import make_policy_observation
from executor.visual_observation import relative_positions, require_visual_observation
from vision.synchronization import FrameSynchronizer


def await_fresh_observation(capture,*,after_s:float,deadline_s:float,clock,health,spin)->dict:
    if not np.isfinite([after_s,deadline_s,clock()]).all() or not 0<=deadline_s-clock()<=.100000001:
        raise ValueError('Fresh observation wait may not exceed100ms')
    while True:
        health()
        if clock()>=deadline_s:raise TimeoutError('Fresh post-submission frame unavailable within100ms')
        frame=capture()
        health()
        if clock()>deadline_s:raise TimeoutError('Capture exceeded100ms wait')
        stamp=(frame['exposure_s'] if isinstance(frame,dict) else frame.exposure_monotonic_s)
        if not np.isfinite(stamp):raise ValueError('Finite exposure required')
        if stamp>after_s:return frame
        spin(min(.002,max(0.,deadline_s-clock())))


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def input_digest(packet):
    frame,details=packet['frame'],packet['details']
    value=dict(rgb_sha256=digest(frame.rgb),frame_id=frame.frame_id,
        exposure=frame.exposure_monotonic_s,received=frame.received_monotonic_s,
        camera_metadata_hash=frame.camera_metadata_hash,clock_verified=frame.clock_verified,
        geometry=asdict(packet['geometry']),paired_feedback=packet['paired_feedback'],
        previous_gripper_target_m=details['previous_gripper_target_m'],
        policy_observation=details['policy_observation'],geometry_features_m=details['geometry_features_m'])
    if 'history' in packet:
        packet['history'].verify()
        value.update(history_context_sha256=packet['history'].context_sha256,
            numeric_history_profile=details['numeric_history_profile'],history_epoch=details['history_epoch'])
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def verify_prediction_packet(packet):
    frame,paired,details=packet['frame'],packet['paired_feedback'],packet['details']
    raw=np.asarray(packet['raw'],float)
    if 'history' in packet:
        packet['history'].verify()
        if (packet['history'].epoch!=details['history_epoch']
                or not np.array_equal(packet['history'].observations[-1],np.asarray(details['policy_observation'],np.float32))):
            raise ValueError('History/current prediction binding changed')
    if (raw.shape!=(20,8) or not np.isfinite(raw).all()
            or details.get('rgb_sha256')!=digest(frame.rgb)
            or details.get('raw_sha256')!=digest(raw)
            or details.get('frame_id')!=frame.frame_id
            or details.get('input_sha256')!=input_digest(packet)
            or not np.array_equal(details.get('feedback_state'),paired['state'])
            or details.get('geometry_calibration_sha256')!=packet['geometry'].calibration_sha256):
        raise ValueError('Prediction/frame/feedback binding changed')


class VisualInferenceWorker:
    def __init__(self, policy, calibration, observer, robot_geometry, *, capture, feedback, clock=time.monotonic,health=None,spin=time.sleep):
        if (policy.config.get('visual_observation_profile')!='visual_relative_base_v1'
                or policy.config.get('policy_input_dim')!=15
                or policy.config.get('calibration_sha256')!=calibration.get('calibration_sha256')):
            raise ValueError('Trained15D policy must bind the same live calibration; nominal calibration cannot be silently replaced')
        self.policy,self.calibration,self.observer,self.robot=policy,calibration,observer,robot_geometry
        self.capture,self.feedback,self.clock=capture,feedback,clock
        self.synchronizer=FrameSynchronizer();self.request_id=0
        self.history_common=policy.config.get('history_runtime_profile')=='history_common_v1'
        if policy.config.get('numeric_history_profile','none')!='none' and not self.history_common:
            raise ValueError('History policy requires the common qualified history runtime')
        self.health,self.spin=health,spin
        self.receipts=[];self.pending_receipt=None;self.history_epoch=0;self.confirmed_s=None
        self.history_stopped=False;self.completed_s=None
        if self.history_common:
            from executor.observation_history import HistoryBuffer
            if health is None or not policy.config.get('session_id'):
                raise ValueError('History requires independent health and explicit session binding')
            self.history_buffer=HistoryBuffer()

    def record_submission(self,receipt):
        from executor.observation_history import lookup_receipt
        if self.history_stopped or self.pending_receipt is not None:
            raise ValueError('Stopped history or unfinished prefix')
        candidate=lookup_receipt(self.receipts+[receipt],float(np.finfo(float).max))
        if candidate['receipt_id']!=receipt['receipt_id'] or receipt['scope']!=self.policy.config['session_id']:
            raise ValueError('New actual prefix receipt for the same session required')
        self.receipts.append(candidate);self.pending_receipt=candidate['receipt_id']

    def prefix_completed(self,receipt_id,*,completed_s=None):
        completed_s=self.clock() if completed_s is None else completed_s
        if (self.history_stopped or self.pending_receipt!=receipt_id
                or not np.isfinite(completed_s) or completed_s<self.receipts[-1]['ack_s']
                or completed_s>self.clock()):
            raise ValueError('Actual pending prefix completion required')
        self.pending_receipt=None;self.completed_s=completed_s

    def confirmation_received(self,*,now_s):
        if self.history_stopped or self.pending_receipt is not None or not np.isfinite(now_s):
            raise ValueError('Cannot confirm stopped or active history runtime')
        self.history_epoch+=1;self.confirmed_s=now_s
        self.history_buffer.clear(epoch=self.history_epoch,after_s=now_s)

    def predict(self, *, previous_gripper_target_m, after_s):
        if self.history_common:
            try:return self._predict_history(previous_gripper_target_m=previous_gripper_target_m,after_s=after_s)
            except BaseException:
                self.history_stopped=True
                self.history_buffer.stopped=True
                raise
        started=self.clock();frame=self.capture()
        if frame.exposure_monotonic_s<=after_s:
            raise ValueError('A fresh post-submission/post-confirmation frame is required')
        paired=self.synchronizer.pair(frame,self.feedback(),now_s=self.clock())
        geometry=self.observer.measure(frame)
        require_visual_observation(frame,geometry,paired,now_s=self.clock(),calibration=self.calibration)
        state=np.asarray(paired['state'],np.float32)
        features=relative_positions(state,geometry,self.robot)
        observed=make_policy_observation(state,self.policy.stats,self.policy.config,
            previous_gripper_target_m,geometry_features_m=features)
        raw=np.asarray(self.policy.predict(frame.rgb,state,previous_gripper_target_m,
            geometry_features_m=features),float)
        require_visual_observation(frame,geometry,paired,now_s=self.clock(),calibration=self.calibration)
        self.request_id+=1
        details=dict(request_id=self.request_id,frame_id=frame.frame_id,feedback_state=paired['state'],
            previous_gripper_target_m=previous_gripper_target_m,policy_input_dim=15,
            policy_observation=observed.tolist(),geometry_features_m=features.tolist(),
            geometry_calibration_sha256=geometry.calibration_sha256,
            camera_metadata_hash=frame.camera_metadata_hash,rgb_sha256=digest(frame.rgb),raw_sha256=digest(raw),
            exposure_monotonic_s=frame.exposure_monotonic_s,feedback_monotonic_s=paired['monotonic_s'],
            inference_started_s=started,inference_finished_s=self.clock(),
            checkpoint_sha256=self.policy.metadata.get('checkpoint_sha256'))
        packet=dict(raw=raw,frame=frame,geometry=geometry,paired_feedback=paired,details=details)
        details['input_sha256']=input_digest(packet)
        verify_prediction_packet(packet)
        return packet

    def _predict_history(self,*,previous_gripper_target_m,after_s):
        from executor.observation_history import lookup_receipt,make_history_item
        if self.history_stopped or self.pending_receipt is not None or not self.receipts:
            raise ValueError('No prediction during active prefix or without accepted history')
        if self.history_buffer.epoch is None:
            self.history_buffer.clear(epoch=self.history_epoch,after_s=after_s)
        boundary=max(after_s,self.confirmed_s if self.confirmed_s is not None else after_s,
            self.completed_s if self.completed_s is not None else after_s)
        started=self.clock()
        while True:
            frame=await_fresh_observation(self.capture,after_s=boundary,deadline_s=self.clock()+.1,
                clock=self.clock,health=self.health,spin=self.spin)
            frame=copy.copy(frame)
            rgb=np.ascontiguousarray(frame.rgb)
            frame.rgb=np.frombuffer(rgb.tobytes(),dtype=rgb.dtype).reshape(rgb.shape)
            paired=self.synchronizer.pair(frame,self.feedback(),now_s=self.clock())
            geometry=copy.deepcopy(self.observer.measure(frame))
            require_visual_observation(frame,geometry,paired,now_s=self.clock(),calibration=self.calibration)
            receipt=lookup_receipt(self.receipts,frame.exposure_monotonic_s)
            state=np.asarray(paired['state'],np.float32)
            features=relative_positions(state,geometry,self.robot)
            previous=float(receipt['rows'][-1][7])
            observed=make_policy_observation(state,self.policy.stats,self.policy.config,previous,geometry_features_m=features)
            details=dict(session_id=self.policy.config['session_id'],admitted_s=self.clock(),
                admission_valid=True,prefix_complete=True,geometry_features_m=features.tolist(),
                policy_observation=observed.tolist())
            admission=dict(frame=frame,paired_feedback=paired,geometry=geometry,details=details)
            self.history_buffer.append(make_history_item(admission,self.receipts,self.policy.stats,self.policy.config,epoch=self.history_epoch))
            boundary=frame.exposure_monotonic_s
            if len(self.history_buffer.items)==5:break
        if previous!=previous_gripper_target_m:
            raise ValueError('Current target does not match exposure-causal receipt')
        batch=self.history_buffer.snapshot(observed)
        kwargs={'geometry_features_m':features}
        profile=self.policy.config.get('numeric_history_profile','none')
        if profile!='none':kwargs['history']=batch
        self.health()
        details.update(previous_gripper_target_m=previous,numeric_history_profile=profile,history_epoch=self.history_epoch)
        packet=dict(frame=frame,geometry=geometry,paired_feedback=paired,details=details,history=batch)
        bound_input=input_digest(packet);bound_rgb=digest(frame.rgb)
        raw=np.asarray(self.policy.predict(frame.rgb,state,previous,**kwargs),float)
        if input_digest(packet)!=bound_input:raise ValueError('Prediction input changed during inference')
        require_visual_observation(frame,geometry,paired,now_s=self.clock(),calibration=self.calibration)
        self.request_id+=1
        details.update(request_id=self.request_id,frame_id=frame.frame_id,feedback_state=paired['state'],
            previous_gripper_target_m=previous,policy_input_dim=15,geometry_calibration_sha256=geometry.calibration_sha256,
            camera_metadata_hash=frame.camera_metadata_hash,rgb_sha256=bound_rgb,raw_sha256=digest(raw),
            exposure_monotonic_s=frame.exposure_monotonic_s,feedback_monotonic_s=paired['monotonic_s'],
            inference_started_s=started,inference_finished_s=self.clock(),
            checkpoint_sha256=self.policy.metadata.get('checkpoint_sha256'),numeric_history_profile=profile,history_epoch=self.history_epoch)
        packet=dict(raw=raw,frame=frame,geometry=geometry,paired_feedback=paired,details=details,history=batch)
        details['input_sha256']=bound_input;verify_prediction_packet(packet)
        return packet
