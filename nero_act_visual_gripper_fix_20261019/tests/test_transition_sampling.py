import hashlib,importlib.util,tempfile,unittest
from pathlib import Path
import h5py,numpy as np

class TransitionSamplingTests(unittest.TestCase):
    def test_off_grid_critical_inputs_are_present_and_causal_wait_filters_remain(self):
        self.assertIsNotNone(importlib.util.find_spec('training.transition_sampling'),'critical first inputs must not disappear between ten-step samples')
        from training.transition_sampling import critical_windows,critical_validation_windows
        from training.train import complete_windows
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'episode_0.hdf5';stages=['approach']*7+['close']*18+['lift']*26+['carry']*4+['release']*25
            actions=np.zeros((80,8),np.float32);actions[:,7]=.095;widths=np.full((80,1),.095,np.float32)
            actions[7:10,7]=[.0948,.0946,.0944];actions[10:25,7]=.067;widths[10:25,0]=.075
            actions[25:55,7]=.067;widths[25:55,0]=.075;actions[55:,7]=np.linspace(.067,.095,25)
            widths[55:,0]=.075
            with h5py.File(path,'w') as f:
                f.attrs['task']='complete_pick_carry_place_retreat'
                f['action']=actions;f['observations/gripper_state']=widths
                f['simulation/stage']=np.array([s.encode() for s in stages])
            before=hashlib.sha256(path.read_bytes()).hexdigest();base=complete_windows([path],20,command_context=True)
            self.assertNotIn((0,25),base)
            selected=critical_windows([path],20,command_context=True)
            for t in [7,25,26,27,28,29,30,55]:self.assertIn((0,t),selected)
            self.assertGreaterEqual(selected.count((0,25)),6)
            self.assertNotIn((0,10),selected,'stationary contact wait must not be reintroduced')
            self.assertTrue(all(selected.count(sample)>=base.count(sample) for sample in set(base)))
            val,records=critical_validation_windows([path],20,20,command_context=True)
            self.assertEqual(len(set(val)),len(val));self.assertIn((0,25),val)
            self.assertIn(dict(episode_id=0,start_ts=25),records)
            self.assertEqual(before,hashlib.sha256(path.read_bytes()).hexdigest())
