import importlib.util,tempfile,unittest
from pathlib import Path
import h5py,numpy as np

class ObservationRefreshTests(unittest.TestCase):
    def test_stale_dense_input_is_rejected_and_physical_equivalence_is_required(self):
        self.assertIsNotNone(importlib.util.find_spec('simulation.refresh_observations'))
        from simulation.refresh_observations import require_fresh_inputs,require_identical_physics
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.hdf5';target=Path(tmp)/'episode_0.hdf5';n=30
            with h5py.File(source,'w') as f:
                f.attrs.update(scene_config='{}',sim=True,task='complete_pick_carry_place_retreat')
                f['action']=np.zeros((n,8));f['simulation/qpos']=np.zeros((n,16));f['simulation/qvel']=np.zeros((n,15))
                f['simulation/stage']=np.array([b'approach']*7+[b'lift']*23)
                f['observations/gripper_state']=np.full((n,1),.095)
                f['observations/images/camera_0']=np.zeros((n,2,2,3),dtype=np.uint8)
            import shutil,hashlib
            shutil.copy2(source,target)
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            with h5py.File(target,'a') as f:
                f.attrs.update(fresh_critical_rgb_hz=20,observation_source_path=str(source),observation_source_sha256=sha)
                f['observations/fresh_rgb_mask']=np.arange(n)%10==0
            self.assertTrue(require_identical_physics(source,target))
            with self.assertRaises(ValueError):require_fresh_inputs([target])
            with h5py.File(target,'a') as f:f['observations/fresh_rgb_mask'][:]=True
            self.assertEqual(require_fresh_inputs([target]),[source])
            with h5py.File(target,'a') as f:f['action'][0,0]=.01
            with self.assertRaises(ValueError):require_identical_physics(source,target)

    def test_derivation_requires_the_previously_verified_source_hash_before_copy(self):
        import inspect
        from simulation.refresh_observations import derive_episode
        self.assertIn('expected_source_sha256',inspect.signature(derive_episode).parameters)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.hdf5';target=Path(tmp)/'derived.hdf5';source.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError,'source'):
                derive_episode(source,target,expected_source_sha256='0'*64)
            self.assertFalse(target.exists())
