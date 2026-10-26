import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))


class OracleObservationTests(unittest.TestCase):
    def module(self):
        path=ROOT/'oracle_observation.py'
        self.assertTrue(path.is_file(),'Missing isolated diagnostic-only cup substitution')
        spec=importlib.util.spec_from_file_location('_tested_oracle_observation',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_only_cup_features_change_and_input_arrays_remain_unchanged(self):
        m=self.module();features=np.arange(6,dtype=np.float32);original=features.copy()
        result=m.replace_cup_for_diagnostic(features,[.1,.2,.3],[.01,.02,.03],diagnostic_only=True)
        np.testing.assert_array_equal(features,original)
        np.testing.assert_allclose(result[:3],[.09,.18,.27],atol=1e-7)
        np.testing.assert_array_equal(result[3:],features[3:])
        self.assertFalse(np.shares_memory(features,result))

    def test_truth_substitution_rejects_normal_mode_and_nonfinite_data(self):
        m=self.module()
        with self.assertRaises(ValueError):m.replace_cup_for_diagnostic(np.zeros(6),[0]*3,[0]*3,diagnostic_only=False)
        with self.assertRaises(ValueError):m.replace_cup_for_diagnostic(np.zeros(6),[np.nan,0,0],[0]*3,diagnostic_only=True)

    def test_diagnostic_success_never_becomes_migration_acceptance(self):
        m=self.module();record=m.diagnostic_record(9005,{'success':True},full_flow_observed=True)
        self.assertTrue(record['diagnostic_full_flow_observed'])
        self.assertTrue(record['geometry_truth_policy_input'])
        self.assertFalse(record['full_task_passed'])
        self.assertFalse(record['hardware_deployment_allowed'])

    def test_delayed_multiple_exposures_use_their_own_snapshot_cup(self):
        m=self.module()
        import mujoco
        from executor.contract import GRIPPER_CONTEXT,VISUAL_PROFILE
        robot=''.join(f'<body name="b{i}"><joint name="j{i}" type="slide" axis="1 0 0" armature="1"/>'
            '<geom type="sphere" size=".01" mass="1"/>' for i in range(9))+'</body>'*9
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+robot+'<body name="cup"><freejoint/>'+
            '<geom type="cylinder" size=".0375 .0475" mass=".05"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model)
        config=dict(gripper_command_context=GRIPPER_CONTEXT,visual_observation_profile=VISUAL_PROFILE,
            state_dim=8,physical_state_dim=8,policy_input_dim=15)
        stats=dict(qpos_mean=np.zeros(8),qpos_std=np.ones(8),geometry_mean=np.zeros(6),geometry_std=np.ones(6))
        scenario=dict(seed=42,jitter=False,observation_delay_steps=1,diagnostic_observation_mode='cup_truth_diagnostic_v1')
        history=m.DiagnosticHistory(config,stats,scenario,np.random.default_rng(42),np.r_[np.zeros(7),.095])
        class Renderer:
            class Scene:flags={}
            scene=Scene()
            def update_scene(self,*args,**kwargs):pass
            def render(self):return np.full((2,2,3),127,np.uint8)
        def observer(rgb,state):
            np.testing.assert_array_equal(rgb,np.full((2,2,3),127,np.uint8))
            return np.arange(6,dtype=np.float32),dict(tcp_base_m=[0,0,0],cup_bottom_base_m=[8,8,8])
        for stamp,x in [(0.,.1),(.05,.2)]:
            data.time=stamp;data.qpos[9:12]=[x,0,.0475];mujoco.mj_forward(model,data)
            history.capture_eligible(data)
        data.time=.1;data.qpos[9]=.9;mujoco.mj_forward(model,data)
        result=history.sample(model,data,Renderer(),observer)
        rows=list(history.buffer.items)
        self.assertEqual(len(rows),2)
        np.testing.assert_allclose(rows[0]['observation'][9:12],[.1,0,0],atol=1e-7)
        np.testing.assert_allclose(rows[1]['observation'][9:12],[.2,0,0],atol=1e-7)
        np.testing.assert_array_equal(result['features'][3:],np.arange(6,dtype=np.float32)[3:])
        self.assertEqual(result['record']['diagnostic_truth_exposure_s'],.05)
        self.assertFalse(result['record']['runtime_qualified'])

    def test_diagnostic_history_cannot_be_used_as_normal_runtime_history(self):
        m=self.module()
        with self.assertRaises(ValueError):
            m.DiagnosticHistory({}, {},dict(seed=42,jitter=False),np.random.default_rng(42),np.zeros(8))


if __name__=='__main__':unittest.main()
