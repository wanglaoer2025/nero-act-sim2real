"""Isolated simulation-truth ablation, never a deployable visual observer."""
from collections import deque
import copy

import mujoco
import numpy as np

from simulation.history_observation import SimulationHistory

MODE='cup_truth_diagnostic_v1'


def replace_cup_for_diagnostic(features,cup,tcp,*,diagnostic_only):
    f,c,t=[np.asarray(x,np.float32) for x in [features,cup,tcp]]
    if (diagnostic_only is not True or f.shape!=(6,) or c.shape!=(3,) or t.shape!=(3,)
            or not np.isfinite(np.r_[f,c,t]).all()):
        raise ValueError('Finite matched geometry and explicit diagnostic-only scope required')
    result=f.copy();result[:3]=c-t
    return result


def diagnostic_record(case_id,result,*,full_flow_observed):
    return dict(id=case_id,mode=MODE,diagnostic_full_flow_observed=bool(full_flow_observed),
        result=result,geometry_truth_policy_input=True,full_task_passed=False,
        migration_acceptance=False,hardware_deployment_allowed=False)


class DiagnosticHistory(SimulationHistory):
    def __init__(self,config,stats,scenario,rng,previous):
        if scenario.get('diagnostic_observation_mode')!=MODE or scenario.get('jitter') is not False:
            raise ValueError('Explicit isolated cup-truth diagnostic without feedback jitter required')
        super().__init__(config,stats,scenario,rng,previous)

    def sample(self,model,data,renderer,observer):
        # The original sampler may consume multiple delayed exposures in this call.
        due=deque(copy.deepcopy(row) for row in self.pending
            if row['stamp']+self.delay_s<=float(data.time)+1e-9)
        cup_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cup')
        if cup_id<0:
            raise ValueError('Diagnostic requires the frozen physical cup body')
        def matched_observer(rgb,observed):
            if not due:
                raise ValueError('No matching exposure snapshot for diagnostic truth')
            captured=due.popleft()
            expected=np.r_[captured['qpos'][:7],captured['qpos'][7]-captured['qpos'][8]].astype(np.float32)
            if not np.array_equal(expected,observed):
                raise ValueError('Diagnostic exposure and observed feedback differ')
            rgb_before,observed_before=rgb.copy(),observed.copy()
            features,record=observer(rgb,observed)
            if not np.array_equal(rgb,rgb_before) or not np.array_equal(observed,observed_before):
                raise ValueError('Underlying observer modified RGB or feedback')
            exposure=mujoco.MjData(model)
            exposure.qpos[:]=captured['qpos'];exposure.qvel[:]=captured['qvel']
            exposure.time=captured['stamp'];mujoco.mj_forward(model,exposure)
            # build_scene requires the retained95mm physical cup. This is its bottom
            # centre convention, not the tilted cylinder's lowest safety extent.
            cup=exposure.xpos[cup_id]-[0,0,.0475]
            replaced=replace_cup_for_diagnostic(features,cup,record['tcp_base_m'],diagnostic_only=True)
            record=dict(record,image_estimated_cup_bottom_base_m=record['cup_bottom_base_m'],
                image_geometry_features_m=np.asarray(features).tolist(),cup_bottom_base_m=cup.tolist(),
                method='DIAGNOSTIC_TRUE_CUP_current_RGB_target_measured_FK',
                geometry_truth_policy_input=True,migration_acceptance=False,runtime_qualified=False,
                diagnostic_truth_exposure_s=float(captured['stamp']))
            return replaced,record
        result=super().sample(model,data,renderer,matched_observer)
        if due:
            raise ValueError('Some diagnostic exposure snapshots were not consumed')
        return result
