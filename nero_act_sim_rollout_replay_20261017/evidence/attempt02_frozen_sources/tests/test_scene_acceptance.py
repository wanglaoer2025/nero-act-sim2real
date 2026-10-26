import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np


class SceneAcceptanceTests(unittest.TestCase):
    def test_explicit_current_user_acceptance_changes_only_cup_scene_not_actions_or_limits(self):
        from executor.scene_acceptance import accepted_scene
        scene={'pick_xy_m':[0.,0.],'place_xy_m':[.2,0.], 'table_top_z_m':0.,'cup':{'height_m':.095},'joint_speed_limit_rad_s':.04}
        manifest={'source_rollout_sha256':'trace'}
        acceptance={'user_statement':'有一点偏差也没关系','source_xy_m':[0.,.0096],
                    'maximum_accepted_source_offset_m':.01,'trace_sha256':'trace',
                    'authorized_scene_sha256':hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest()}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'acceptance.json';p.write_text(json.dumps(acceptance))
            result=accepted_scene(scene,manifest,p)
            self.assertEqual(result['pick_xy_m'],[0.,.0096])
            expected=copy.deepcopy(scene);expected['pick_xy_m']=[0.,.0096]
            self.assertEqual(result,expected);self.assertEqual(scene['pick_xy_m'],[0.,0.])
            for change in [{'user_statement':''},{'source_xy_m':[0.,.02]}, {'trace_sha256':'other'}, {'maximum_accepted_source_offset_m':.1}]:
                p.write_text(json.dumps(dict(acceptance,**change)))
                with self.assertRaises((ValueError,PermissionError)):accepted_scene(scene,manifest,p)

    def test_acceptance_does_not_override_live_5mm_stability_or_robot_start(self):
        from executor.execution_guards import require_current_scene
        scene=dict(pick_xy_m=[0,.0096],place_xy_m=[.2,0],table_top_z_m=0,
            initial_robot_state=dict(qpos_rad=[0]*7,gripper_total_width_m=.095))
        geometry=dict(capture_wall_s=100,geometry_fit_passed=True,table_z_base_mm=0,
            targets=dict(source_bottom_mm=[0,9.6,0],destination_mm=[200,0,0]))
        state=np.r_[np.zeros(7),.095];require_current_scene(scene,state,geometry,now=101)
        geometry['targets']['source_bottom_mm']=[0,15.6,0]
        with self.assertRaises(ValueError):require_current_scene(scene,state,geometry,now=101)

    def test_accepted_offset_needs_both_matching_physics_and_full_collision_proofs(self):
        from executor.scene_acceptance import execution_scene,scene_hash
        scene={'pick_xy_m':[0.,0.]};manifest={'source_rollout_sha256':'trace'}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'evidence';out.mkdir();(out/'accepted_offset_collision').mkdir()
            accepted=dict(scene,pick_xy_m=[0.,.009])
            (out/'source_offset_acceptance.json').write_text(json.dumps(dict(user_statement='有一点偏差也没关系',source_xy_m=[0.,.009],maximum_accepted_source_offset_m=.01,trace_sha256='trace',authorized_scene_sha256=scene_hash(scene))))
            proof=dict(success=True,hardware_connected=False,execution_scene_sha256=scene_hash(accepted),trace_sha256='trace')
            (out/'accepted_offset_replay_physics.json').write_text(json.dumps(proof))
            with self.assertRaises(FileNotFoundError):execution_scene(root,scene,manifest,require_evidence=True)
            path=out/'accepted_offset_collision/report.json';path.write_text(json.dumps(dict(proof,trace_sha256='other')))
            with self.assertRaises(PermissionError):execution_scene(root,scene,manifest,require_evidence=True)
            path.write_text(json.dumps(proof))
            self.assertEqual(execution_scene(root,scene,manifest,require_evidence=True),accepted)

    def test_fresh_retry_authorization_must_bind_existing_scene_acceptance_without_rewriting_it(self):
        from executor.scene_acceptance import execution_scene,scene_hash
        scene={'pick_xy_m':[0.,0.]};manifest={'source_rollout_sha256':'trace'}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'evidence';out.mkdir();path=out/'source_offset_acceptance.json'
            path.write_text(json.dumps(dict(user_statement='有一点偏差也没关系',source_xy_m=[0.,.009],maximum_accepted_source_offset_m=.01,trace_sha256='trace',authorized_scene_sha256=scene_hash(scene),parent_authorization_sha256='old_attempt')))
            before=path.read_bytes();bound=hashlib.sha256(before).hexdigest()
            with self.assertRaises(PermissionError):execution_scene(root,scene,manifest,receipt=dict(source_acceptance_sha256='wrong'))
            self.assertEqual(execution_scene(root,scene,manifest,receipt=dict(source_acceptance_sha256=bound))['pick_xy_m'],[0.,.009])
            self.assertEqual(path.read_bytes(),before)
