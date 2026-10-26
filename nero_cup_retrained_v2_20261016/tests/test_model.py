import unittest,json,pickle,hashlib
from pathlib import Path
import numpy as np,h5py,torch
ROOT=Path(__file__).resolve().parents[1]
class ModelTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.bundle=torch.load(ROOT/'model/policy.ckpt',map_location='cpu',weights_only=False);cls.c=cls.bundle['config']
  with (ROOT/'model/dataset_stats.pkl').open('rb') as f:cls.stats=pickle.load(f)
 def test_entire20_step_horizon_supervised(self):
  self.assertEqual(self.c['chunk_size'],20);self.assertEqual(self.c['loss_prefix_steps'],20);self.assertEqual(self.c['execution_query_stride'],10);self.assertEqual(self.c['state_dim'],8);self.assertEqual(self.c['action_dim'],8)
 def test_anchor_matches_statistics(self):
  self.assertTrue(self.c['joint_residual_anchor']);np.testing.assert_allclose(self.c['joint_anchor_scale'],self.stats['qpos_std'][:7]/self.stats['action_std'][:7],rtol=1e-6);np.testing.assert_allclose(self.c['joint_anchor_offset'],(self.stats['qpos_mean'][:7]-self.stats['action_mean'][:7])/self.stats['action_std'][:7],rtol=1e-6,atol=1e-7)
 def test_train_only_stats(self):
  states=[];actions=[]
  split=json.loads((ROOT/'model/split.json').read_text());self.assertEqual(split,{'train':list(range(6)),'val':[6,7]})
  for i in split['train']:
   with h5py.File(ROOT/f'training_data/episode_{i}.hdf5','r') as f:states.append(np.concatenate([f['observations/qpos'][:].astype(np.float32),f['observations/gripper_state'][:].astype(np.float32)],axis=1));actions.append(f['action'][:].astype(np.float32))
  for key,rows in [('qpos',states),('action',actions)]:
   x=np.concatenate(rows);np.testing.assert_array_equal(self.stats[key+'_mean'],x.mean(0));np.testing.assert_array_equal(self.stats[key+'_std'],np.maximum(x.std(0),1e-2))
 def test_no_oracle_in_policy_input(self):
  self.assertFalse(self.c.get('coordinate_conditioning',False));self.assertFalse(self.c.get('task_displacement_features',False));self.assertEqual(self.c['state_dim'],8)
 def test_no_hardware_grant(self):self.assertFalse(self.bundle['hardware_deployment_allowed'])
 def test_saved_real_observation_inference(self):
  p=json.loads((ROOT/'preview/report.json').read_text());self.assertEqual(p['raw_chunk_shape'],[20,8]);self.assertLess(p['first_arm_gap_rad'],.01);self.assertEqual(p['checkpoint_sha256'],hashlib.sha256((ROOT/'model/policy.ckpt').read_bytes()).hexdigest());self.assertFalse(p['motion_sent'])
 def test_action_units_and_actual_training_slew(self):
  for i in range(8):
   with h5py.File(ROOT/f'training_data/episode_{i}.hdf5','r') as f:a=f['action'][:];self.assertLessEqual(float(np.max(abs(np.diff(a[:,:7],axis=0)))),.002001);self.assertLessEqual(float(np.max(abs(np.diff(a[:,7])))),.000501)
 def test_physical_replay_and_collision_audit(self):
  for name in ['replay_verification.json','data_audit.json']:
   report=json.loads((ROOT/'evidence'/name).read_text());self.assertTrue(report['passed']);self.assertEqual(len(report['episodes']),8)
 def test_frozen_evaluation_matches_weights(self):
  e=json.loads((ROOT/'evaluation/summary.json').read_text());self.assertTrue(e['complete']);self.assertEqual(e['total'],16);self.assertEqual(e['checkpoint_sha256'],hashlib.sha256((ROOT/'model/policy.ckpt').read_bytes()).hexdigest());self.assertEqual(e['stats_sha256'],hashlib.sha256((ROOT/'model/dataset_stats.pkl').read_bytes()).hexdigest());self.assertEqual(e['passed'],sum(r['success'] for r in e['results']))
 def test_success_requires_real_simulation_hold(self):
  e=json.loads((ROOT/'evaluation/summary.json').read_text())
  for r in e['results']:
   if not r['success']:continue
   self.assertTrue(r['continuous_post_hold_verified']);self.assertEqual(r['post_completion_hold_points'],100);z=np.load(ROOT/'evaluation'/f"{r['scenario']}.npz");t=z['states_targets_cup_tcp_contacts'];mask=z['completion_hold_mode'].astype(bool);post=t[mask];self.assertEqual(len(post),100);self.assertTrue(np.all(post[:,22]==1));np.testing.assert_allclose(post[:,8:16],np.broadcast_to(post[0,8:16],post[:,8:16].shape),atol=1e-8);self.assertLessEqual(int(z['query_steps'].max()),r['completion_step'])
 def test_hold_timing_repair_preserves_original_motion(self):
  fix=json.loads((ROOT/'evidence/hold_timing_fix.json').read_text());self.assertTrue(fix['model_sha256_unchanged']);self.assertTrue(fix['thresholds_unchanged']);self.assertTrue(fix['reach_deadline_unchanged'])
  self.assertEqual(fix['original_score'],[9,16]);self.assertEqual(fix['corrected_score'],[13,16]);self.assertEqual(len(fix['comparisons']),4)
  for c in fix['comparisons']:
   self.assertEqual(c['original_prefix_max_difference'],0);self.assertEqual(c['corrected_hold_points'],100)
   old=np.load(ROOT/'evidence/evaluation_initial_window'/f"{c['scenario']}.npz")['states_targets_cup_tcp_contacts'];new=np.load(ROOT/'evaluation'/f"{c['scenario']}.npz")['states_targets_cup_tcp_contacts'];np.testing.assert_array_equal(old,new[:len(old)])
if __name__=='__main__':unittest.main()
