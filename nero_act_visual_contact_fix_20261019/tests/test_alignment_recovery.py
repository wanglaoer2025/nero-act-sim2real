import copy
import unittest
from pathlib import Path

import test_recovery_training as fixtures


class AlignmentRecoveryTests(unittest.TestCase):
    def test_verified_descend_corrections_are_accepted_as_partial_only(self):
        from training.recoveries import validate_corrections
        m = fixtures.RecoveryTrainingTests().rows()
        for row in m['episodes']:
            row['task'] = 'descend_to_retreat_correction'
        self.assertEqual(len(validate_corrections(m, [], check_files=False)), 3)
        m['episodes'][0]['full_task_verified'] = True
        with self.assertRaises(ValueError):
            validate_corrections(m, [], check_files=False)

    def test_new_descend_correction_preserves_the_original_goal_orientation(self):
        from training.recoveries import validate_recovery_scene
        original = dict(ik_goal_rotation_link7=[[1,0,0],[0,1,0],[0,0,1]])
        snap = dict(qpos=[0.]*7+[.0475,-.0475])
        config = dict(original, continuation='descend_to_retreat',
            joint_speed_limit_rad_s=.02, teacher_command_lead_limit_rad=.015,
            gripper_speed_limit_m_s=.004, teacher_gripper_feedback_lead_m=.008,
            dwell_seconds=dict(close=8,release=8),
            initial_robot_state=dict(qpos_rad=[0.]*7,gripper_total_width_m=.095))
        validate_recovery_scene(config, original, snap)
        config['ik_goal_rotation_link7'] = [[-1,0,0],[0,-1,0],[0,0,1]]
        with self.assertRaises(ValueError):
            validate_recovery_scene(config, original, snap)

    def test_mixed_batches_keep_old_lift_and_new_descend_provenance_and_splits(self):
        from training.generalized import combined_inputs
        base = [dict(id=i,path=f'/tmp/episode_{i}.hdf5',status='valid',full_task_verified=True,
            teacher_success=True,split='train' if i<36 else 'val',source_group=f'base{i}',
            replay=dict(passed=True),prefix_contract_audit=dict(passed=True)) for i in range(48)]
        a = fixtures.RecoveryTrainingTests().rows();b = copy.deepcopy(a)
        b['source_evaluation'] = '/tmp/eval_new'
        b['source_checkpoint_sha256'] = 'c'*64
        for row in b['episodes']:
            row.update(id=row['id']+1000,path=f"/tmp/episode_{row['id']+1000}.hdf5",
                task='descend_to_retreat_correction',origin=row['origin'].replace('eval/','eval_new/'),
                source_group='new_'+row['source_group'])
        manifest, split, paths = combined_inputs(dict(episodes=base),
            dict(train=list(range(36)),val=list(range(36,48))),[a,b],check_files=False)
        self.assertEqual(len(paths), 54)
        self.assertEqual(split['val'][-2:], [1003,2003])
        self.assertNotIn(Path('/tmp/episode_2003.hdf5'),
            [p for p,r in zip(paths,manifest['episodes']) if r['id'] in split['train']])
        b['episodes'][0]['source_group'] = a['episodes'][0]['source_group']
        with self.assertRaises(ValueError):
            combined_inputs(dict(episodes=base),dict(train=list(range(36)),val=list(range(36,48))),
                [a,b],check_files=False)

    def test_renaming_groups_cannot_split_the_same_origin_across_batches(self):
        from training.generalized import combined_inputs
        base = [dict(id=i,path=f'/tmp/episode_{i}.hdf5',status='valid',full_task_verified=True,
            teacher_success=True,split='train' if i<36 else 'val',source_group=f'base{i}',
            replay=dict(passed=True),prefix_contract_audit=dict(passed=True)) for i in range(48)]
        a = fixtures.RecoveryTrainingTests().rows();b = copy.deepcopy(a)
        for i,row in enumerate(b['episodes']):
            row.update(id=2001+i,path=f'/tmp/episode_{2001+i}.hdf5',
                source_group='renamed_'+row['source_group'],origin=f'/tmp/eval/case{4+i}')
        b['episodes'][0]['origin'] = '/tmp/eval/../eval/case2'
        with self.assertRaises(ValueError):
            combined_inputs(dict(episodes=base),dict(train=list(range(36)),val=list(range(36,48))),
                [a,b],check_files=False)

    def test_descend_recovery_inputs_are_dense_for_the_whole_descent(self):
        from training.transition_sampling import critical_inputs
        import numpy as np
        stages = np.array(['descend']*35+['close']*25+['lift']*25)
        starts = {t for t,stage,offset in critical_inputs(stages,dense_descend=True) if stage=='descend'}
        self.assertEqual(starts,set(range(35)))
        self.assertFalse(any(stage=='descend' for _,stage,_ in critical_inputs(stages)))
