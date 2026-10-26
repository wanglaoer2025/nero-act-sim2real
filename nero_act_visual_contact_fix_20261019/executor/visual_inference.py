"""Same-frame15D worker core, for a qualified camera/frontend supplied by caller.

No robot interfaces, trajectory teacher or simulator truth are imported. The
caller runs this worker on a background thread while the ROS thread checks health.
"""
import hashlib
import json
from dataclasses import asdict
import time

import numpy as np

from executor.model import make_policy_observation
from executor.visual_observation import relative_positions, require_visual_observation
from vision.synchronization import FrameSynchronizer


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
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def verify_prediction_packet(packet):
    frame,paired,details=packet['frame'],packet['paired_feedback'],packet['details']
    raw=np.asarray(packet['raw'],float)
    if (raw.shape!=(20,8) or not np.isfinite(raw).all()
            or details.get('rgb_sha256')!=digest(frame.rgb)
            or details.get('raw_sha256')!=digest(raw)
            or details.get('frame_id')!=frame.frame_id
            or details.get('input_sha256')!=input_digest(packet)
            or not np.array_equal(details.get('feedback_state'),paired['state'])
            or details.get('geometry_calibration_sha256')!=packet['geometry'].calibration_sha256):
        raise ValueError('Prediction/frame/feedback binding changed')


class VisualInferenceWorker:
    def __init__(self, policy, calibration, observer, robot_geometry, *, capture, feedback, clock=time.monotonic):
        if (policy.config.get('visual_observation_profile')!='visual_relative_base_v1'
                or policy.config.get('policy_input_dim')!=15
                or policy.config.get('calibration_sha256')!=calibration.get('calibration_sha256')):
            raise ValueError('Trained15D policy must bind the same live calibration; nominal calibration cannot be silently replaced')
        self.policy,self.calibration,self.observer,self.robot=policy,calibration,observer,robot_geometry
        self.capture,self.feedback,self.clock=capture,feedback,clock
        self.synchronizer=FrameSynchronizer();self.request_id=0

    def predict(self, *, previous_gripper_target_m, after_s):
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
