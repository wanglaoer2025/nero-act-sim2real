import copy,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import h5py
import numpy as np

class RecoveryTrainingTests(unittest.TestCase):
    def rows(self):
        return dict(complete=True,rejected=[],source_checkpoint_sha256='a'*64,
            source_evaluation='/tmp/eval',source_protocol_sha256='b'*64,
            episodes=[dict(id=1001+i,path=f'/tmp/episode_{1001+i}.hdf5',status='valid',
                split='train' if i<2 else 'val',source_group='site' if i<2 else 'cup_other',
                task='lift_to_retreat_correction',teacher_success=True,full_task_verified=False,
                correction_verified=True,origin=f'/tmp/eval/case{i}',query_step=101,
                replay=dict(passed=True),prefix_contract_audit=dict(passed=True)) for i in range(3)])

    def test_partial_corrections_have_separate_ids_and_grouped_splits(self):
        self.assertIsNotNone(importlib.util.find_spec("training.recoveries"), "verified partial-correction adapter is required")
        from training.recoveries import validate_corrections
        m=self.rows();paths=validate_corrections(m,[],check_files=False)
        self.assertEqual([p.name for p in paths],['episode_1001.hdf5','episode_1002.hdf5','episode_1003.hdf5'])
        for field,value in [('id',1001),('source_group','site'),('task','complete_pick_carry_place_retreat'),
                ('full_task_verified',True),('correction_verified',False),('path','/tmp/episode_1001.hdf5')]:
            changed=copy.deepcopy(m);changed['episodes'][-1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_corrections(changed,[],check_files=False)
        changed=copy.deepcopy(m);changed['episodes'][-1]['origin']=changed['episodes'][0]['origin']
        with self.assertRaises(ValueError):validate_corrections(changed,[],check_files=False)
        with self.assertRaises(ValueError):
            validate_corrections(m,[dict(id=0,path='/tmp/episode_0.hdf5',source_group='site')],check_files=False)

    def test_hdf_role_and_exact_failed_snapshot_provenance_are_verified(self):
        self.assertIsNotNone(importlib.util.find_spec("training.recoveries"), "verified partial-correction adapter is required")
        from training.recoveries import validate_corrections
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);origin=root/'eval/case0';origin.mkdir(parents=True)
            snap=dict(qpos=[0.]*7+[.0375,-.0375]+[0.]*7,qvel=[0.]*15,previous_target=[0.]*7+[.067])
            (origin/'physical_snapshots.json').write_text(json.dumps(dict(query_steps=[101],snapshots=[snap])))
            original=dict(table_top_z_m=0.,finger_force_limit_n=1.,arm_kp=[100]*7)
            (origin/'scene_configuration.json').write_text(json.dumps(original))
            (root/'eval/protocol.json').write_text(json.dumps(dict(checkpoint_sha256='a'*64)))
            path=root/'episode_1001.hdf5'
            source=dict(origin=str(origin),query_step=101,split='train',source_group='site',role='partial_not_full')
            config=dict(original,continuation='lift_to_retreat',initial_simulation_snapshot=snap,correction_source=source,
                joint_speed_limit_rad_s=.02,teacher_command_lead_limit_rad=.015,
                gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
                dwell_seconds=dict(close=8,release=8),
                initial_robot_state=dict(qpos_rad=snap['qpos'][:7],gripper_total_width_m=.075))
            with h5py.File(path,'w') as f:
                f.attrs.update(task='lift_to_retreat_correction',sim=True,teacher_success=True,
                    full_task_verified=False,same_cycle_observation_before_action=True,scene_config=json.dumps(config))
                f['simulation/qpos']=np.array([snap['qpos']]);f['simulation/qvel']=np.array([snap['qvel']])
            m=self.rows();m['episodes']=m['episodes'][:1];row=m['episodes'][0]
            row.update(path=str(path),origin=str(origin))
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            m.update(source_evaluation=str(root/'eval'),source_protocol_sha256=sha(root/'eval/protocol.json'))
            row.update(sha256=sha(path),snapshot_sha256=sha(origin/'physical_snapshots.json'),
                scene_sha256=sha(origin/'scene_configuration.json'))
            self.assertEqual(validate_corrections(m,[]),[path])
            for key,value in [('table_top_z_m',.001),('finger_force_limit_n',2.),('arm_kp',[120]*7)]:
                with h5py.File(path,'a') as f:
                    bad=copy.deepcopy(config);bad[key]=value;f.attrs['scene_config']=json.dumps(bad)
                row['sha256']=sha(path)
                with self.subTest(physical_key=key),self.assertRaises(ValueError):validate_corrections(m,[])
            # A manifest SHA alone cannot legitimize a modified snapshot or role.
            with h5py.File(path,'a') as f:
                bad=copy.deepcopy(config);bad['initial_simulation_snapshot']['qpos'][0]=.1
                f.attrs['scene_config']=json.dumps(bad)
            row['sha256']=sha(path)
            with self.assertRaises(ValueError):validate_corrections(m,[])

    def test_combined_split_retains_base_validation_and_uses_only_training_corrections_for_stats(self):
        import training.generalized as module
        self.assertTrue(hasattr(module,'combined_inputs'),'optional corrections must reach the actual training adapter')
        rows=[dict(id=i,path=f'/tmp/episode_{i}.hdf5',status='valid',full_task_verified=True,
            teacher_success=True,split='train' if i<36 else 'val',source_group=f'base{i}',
            replay=dict(passed=True),prefix_contract_audit=dict(passed=True)) for i in range(48)]
        manifest,split,paths=module.combined_inputs(dict(episodes=rows),
            dict(train=list(range(36)),val=list(range(36,48))),self.rows(),check_files=False)
        self.assertEqual(len(paths),51)
        self.assertEqual(split['train'],list(range(36))+[1001,1002])
        self.assertEqual(split['val'],list(range(36,48))+[1003])
        train_paths=[p for p,r in zip(paths,manifest['episodes']) if r['id'] in split['train']]
        self.assertNotIn(Path('/tmp/episode_1003.hdf5'),train_paths)
