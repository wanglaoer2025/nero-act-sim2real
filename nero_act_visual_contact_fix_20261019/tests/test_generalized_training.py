import copy,importlib.util,json
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

class GeneralizedTrainingTests(unittest.TestCase):
    def test_rebased_context_model_preserves_physical_predictions_with_new_statistics(self):
        self.assertIsNotNone(importlib.util.find_spec('training.rebase_initializer'))
        from training.rebase_initializer import rebase_initializer,paired_config
        from executor.model import BoundedActPolicy,make_policy_observation,physical_actions
        config=json.loads((ROOT/'initializers/run13/training_config.json').read_text())
        config.update(device='cpu',pretrained_backbone=False)
        old={k:np.array(v,np.float32) for k,v in config['constraint_loss_stats'].items()}
        new={k:v.copy() for k,v in old.items()}
        new['qpos_mean']+=np.array([.01]*7+[.001],np.float32)
        new['qpos_std']*=1.2;new['action_mean']+=.002;new['action_std']*=.8
        torch.set_num_threads(1);torch.manual_seed(41)
        a=BoundedActPolicy(config).eval()
        new_config=paired_config(config,new)
        b=BoundedActPolicy(new_config).eval()
        b.load_state_dict(rebase_initializer(a.state_dict(),old,new,config))
        image=torch.rand(2,1,3,90,160)
        states=np.array([config['real_initial_state'],config['real_initial_state']],np.float32)
        states[1,7]=.074
        qa=np.array([make_policy_observation(s,old,config,.065) for s in states])
        qb=np.array([make_policy_observation(s,new,new_config,.065) for s in states])
        with torch.no_grad():
            pa=physical_actions(a(torch.from_numpy(qa),image),old,bounded_gripper=True)
            pb=physical_actions(b(torch.from_numpy(qb),image),new,bounded_gripper=True)
        torch.testing.assert_close(pa,pb,atol=2e-6,rtol=2e-5)

    def test_manifest_requires_complete_verified_unique_groups_and36_12_split(self):
        self.assertIsNotNone(importlib.util.find_spec('training.generalized'))
        from training.generalized import validate_manifest
        rows=[dict(id=i,path=f'/tmp/episode_{i}.hdf5',status='valid',full_task_verified=True,
                   teacher_success=True,split='train' if i<36 else 'val',source_group=f'scene{i}',
                   replay=dict(passed=True),prefix_contract_audit=dict(passed=True)) for i in range(48)]
        paths=validate_manifest(dict(episodes=rows,rejected=[]),dict(train=list(range(36)),val=list(range(36,48))),check_files=False)
        self.assertEqual(len(paths),48)
        for field,value in [('full_task_verified',False),('source_group','scene0')]:
            changed=copy.deepcopy(rows);changed[-1][field]=value
            with self.assertRaises(ValueError):validate_manifest(dict(episodes=changed,rejected=[]),dict(train=list(range(36)),val=list(range(36,48))),check_files=False)

    def test_same_hdf5_cannot_be_relabeled_as_multiple_scene_groups(self):
        from training.generalized import validate_manifest
        rows=[dict(id=i,path='/tmp/one_episode.hdf5',status='valid',full_task_verified=True,
                   teacher_success=True,split='train' if i<36 else 'val',source_group=f'scene{i}',
                   replay=dict(passed=True),prefix_contract_audit=dict(passed=True)) for i in range(48)]
        with self.assertRaises(ValueError):
            validate_manifest(dict(episodes=rows,rejected=[]),dict(train=list(range(36)),val=list(range(36,48))),check_files=False)
