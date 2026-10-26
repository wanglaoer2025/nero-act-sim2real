import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'nero_act_new_calibration_20261023'


def load_protocol(test):
    path = ROOT/'protocol.py'
    test.assertTrue(path.is_file(), 'Missing audited new-camera scene and sampling protocol')
    spec = importlib.util.spec_from_file_location('newcamera_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProtocolTests(unittest.TestCase):
    def test_new_calibration_groups_and_exact_dynamic_recovery_are_preserved(self):
        module = load_protocol(self)
        scene = json.loads((SOURCE/'scene_relocalized_after_return.json').read_text())
        snapshots = json.loads((SOURCE/'after_return_validation/hypothetical_95mm_start/physical_snapshots.json').read_text())['snapshots']
        original = copy.deepcopy(scene)
        rows = module.plan_scenes(scene, snapshots)
        self.assertEqual(scene, original)
        self.assertEqual(len(rows), 10)
        train = [r for r in rows if r['split']=='train']
        val = [r for r in rows if r['split']=='val']
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertFalse({r['source_group'] for r in train}&{r['source_group'] for r in val})
        for row in rows:
            self.assertEqual(row['scene']['camera_calibration'], original['camera_calibration'])
            self.assertFalse(row['scene']['hardware_deployment_allowed'])
        recovery = [r for r in rows if r['role']=='recovery']
        self.assertEqual([r['parent_query'] for r in recovery], [100,273,450])
        for row in recovery:
            self.assertEqual(row['scene']['initial_simulation_snapshot'], snapshots[row['parent_query']])
            self.assertEqual(row['scene']['initial_robot_state']['qpos_rad'], snapshots[row['parent_query']]['qpos'][:7])

    def test_fixed_batches_cover_every_stage_and_current_scene_without_validation(self):
        module = load_protocol(self)
        pools = {f'{kind}:{stage}': [j*20+k for k in range(5)]
                 for j,(kind,stage) in enumerate((kind,stage) for kind in ['center','other'] for stage in module.STAGES)}
        batches = module.balanced_batches(pools, steps=3)
        self.assertEqual(batches.shape, (3,32))
        self.assertTrue(np.array_equal(batches,module.balanced_batches(pools,steps=3)))
        allowed = {x for values in pools.values() for x in values}
        self.assertTrue(set(batches.ravel()) <= allowed)
        for row in batches:
            for key, values in pools.items():
                self.assertEqual(sum(int(x in values) for x in row), 2, key)
        pools['other:lift'] = []
        with self.assertRaises(ValueError):
            module.balanced_batches(pools,steps=3)

    def test_initial_reference_requires_completed_preparation_and_fresh_actual_feedback(self):
        module = load_protocol(self)
        state=np.array([0,0,0,1.5,0,.2,0,.095])
        observed=dict(state=state.tolist(),receipt_monotonic_s=10.1)
        preparation=dict(accepted=True,result_status=4,error_code=0,
            target=state.tolist(),finished_monotonic_s=10.,arm_held_verified=True)
        np.testing.assert_array_equal(module.initial_reference(observed,preparation,now=10.2),state)
        bad=copy.deepcopy(observed);bad['state'][7]=.09942
        with self.assertRaises(ValueError):module.initial_reference(bad,preparation,now=10.2)
        self.assertEqual(bad['state'][7],.09942)
        for changes in [dict(accepted=False),dict(result_status=6),dict(finished_monotonic_s=10.15),dict(arm_held_verified=False)]:
            with self.assertRaises(ValueError):
                module.initial_reference(observed,dict(preparation,**changes),now=10.2)
        with self.assertRaises(ValueError):module.initial_reference(observed,preparation,now=10.5)

    def test_context_initialization_does_not_replace_original_physical_action_limits(self):
        module=load_protocol(self)
        state=[3.0,0,0,1.5,0,.2,0,.095]
        observed=dict(state=state,receipt_monotonic_s=10.1)
        preparation=dict(accepted=True,result_status=4,error_code=0,target=state,
            finished_monotonic_s=10.,arm_held_verified=True)
        reference=module.initial_reference(observed,preparation,now=10.2)
        sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
        from executor.contract import FullTaskContract
        guard=FullTaskContract(dict(task_scope='complete_pick_carry_place_retreat',
            chunk_size=20,state_dim=8,action_dim=8))
        with self.assertRaisesRegex(ValueError,'physical arm/gripper limits'):
            guard.validate_prefix(np.tile(state,(20,1)),dict(state=state,age_s=0),reference)


if __name__ == '__main__':
    unittest.main()
