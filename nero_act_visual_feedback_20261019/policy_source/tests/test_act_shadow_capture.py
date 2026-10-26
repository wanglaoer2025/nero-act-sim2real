import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.act.run_act_shadow import JOINT_NAMES,load_capture
from src.act.run_act_real import load_runtime


class ShadowCaptureTests(unittest.TestCase):
    def test_observation_only_handoff_is_rejected_before_live_model_or_hardware_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'policy.ckpt'
            torch.save({'hardware_deployment_allowed':False},checkpoint)
            config=root/'config.yaml';config.write_text(f'checkpoint: {checkpoint}\ndevice: cpu\n')
            with self.assertRaisesRegex(ValueError,'run_act_shadow'):load_runtime(config)

    def write_capture(self,path,**updates):
        values={'joint_names':np.array(JOINT_NAMES[::-1]),
                'state':np.tile(np.arange(8)[::-1],(2,1)),
                'rgb':np.zeros((2,16,16,3),dtype=np.uint8),
                'sample_monotonic_s':np.array([10.,10.05])}
        values.update(updates);np.savez(path,**values)

    def test_saved_named_feedback_is_reordered_without_rotating_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observations.npz';rgb=np.zeros((2,16,16,3),dtype=np.uint8);rgb[:,0,0]=[255,40,10]
            self.write_capture(path,rgb=rgb)
            states,loaded,_=load_capture(path)
            np.testing.assert_array_equal(states[0],np.arange(8))
            np.testing.assert_array_equal(loaded,rgb)

    def test_duplicate_names_or_nonmonotonic_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observations.npz'
            self.write_capture(path,joint_names=np.array(['joint1']*8))
            with self.assertRaises(ValueError):load_capture(path)
            self.write_capture(path,sample_monotonic_s=np.array([10.,10.]))
            with self.assertRaises(ValueError):load_capture(path)


if __name__=='__main__':unittest.main()
