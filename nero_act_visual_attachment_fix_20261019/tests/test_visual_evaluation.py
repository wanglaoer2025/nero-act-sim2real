import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    import importlib.util
    assert importlib.util.find_spec('simulation.visual_evaluation') is not None, 'Visual15D full-physics evaluation missing'
    import simulation.visual_evaluation as m
    return m


def test_partial_grasp_is_not_complete():
    m = api()
    result = dict(success=True, max_lift_m=.01, maximum_horizontal_transfer_m=0,
                  stable_released_frames=0, simulated_events=[dict(kind='GRASP')])
    assert m.full_task_passed(result) is False


def test_complete_requires_all_confirmations_and_five_second_settle():
    m = api()
    r = dict(success=True, error=None, max_lift_m=.19, maximum_horizontal_transfer_m=.15,
        stable_released_frames=100, unexpected_arm_cup_contact_frames=0, execution_contract_failures=0,
        premature_release_before_support=False, simulated_events=[dict(kind=x) for x in ['GRASP','CARRY','RELEASE','RETREAT']],
        active_policy_points=3500)
    assert m.full_task_passed(r)
    for update in [dict(stable_released_frames=99),dict(active_policy_points=3901),
                   dict(unexpected_arm_cup_contact_frames=1),dict(max_lift_m=.08),dict(error='stopped')]:
        assert not m.full_task_passed(dict(r, **update))


def test_independent_ids_match_but_incomplete_parameter_predeclaration_is_explicit():
    m = api()
    protocol = json.loads((ROOT/'evidence/baseline_development_protocol.json').read_text())
    center = json.loads((ROOT/'scene/center.json').read_text())
    cases, provenance = m.build_cases(protocol, center)
    assert len(cases) == 16 and [c['name'] for c in cases] == [r['id'] for r in protocol['cases']]
    assert cases[0]['scene']['pick_xy_m'] != center['pick_xy_m']
    assert provenance['all_parameters_predeclared_before_training'] is False
    assert provenance['hardware_deployment_allowed'] is False


def test_observation_failure_cannot_fallback_to_scene_positions():
    m = api()
    center = json.loads((ROOT/'scene/center.json').read_text())
    state = np.r_[center['initial_robot_state']['qpos_rad'], .095]
    with pytest.raises(ValueError, match='visible|unique'):
        m.image_features(np.zeros((180,320,3),np.uint8), state, center)


def test_changed_geometry_invalidates_evaluation(tmp_path):
    m = api()
    path = tmp_path/'scene.json'; path.write_text('{}')
    binding = m.file_hashes([path]); path.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match='changed'): m.require_files(binding)


def test_wrong_visual_condition_is_not_silently_ignored():
    m = api()
    from tests.test_visual_policy import config, statistics
    import torch
    from executor.model import BoundedActPolicy, make_policy_observation
    torch.manual_seed(42); torch.set_num_threads(1)
    c=config(); policy=BoundedActPolicy(c).eval(); stats=statistics(c)
    state=np.r_[np.zeros(7), .095]; image=torch.rand(1,1,3,90,160)
    first=make_policy_observation(state,stats,c,.095,geometry_features_m=np.zeros(6))
    second=make_policy_observation(state,stats,c,.095,geometry_features_m=np.ones(6)*.02)
    with torch.no_grad():
        a=policy(torch.from_numpy(first)[None],image); b=policy(torch.from_numpy(second)[None],image)
    assert torch.max(torch.abs(a-b))>0


def test_incomplete_or_duplicate_cases_cannot_finalize_full_suite():
    m=api()
    records=[dict(name=f'case{i}',passed=False) for i in range(16)]
    assert m.aggregate_cases(records,[r['name'] for r in records])['passed']==0
    for bad in [records[:-1],records[:-1]+[records[0]]]:
        with pytest.raises(ValueError,match='16|case|unique'):
            m.aggregate_cases(bad,[f'case{i}' for i in range(16)])


def test_delayed_observation_uses_old_paired_physics_and_reset_requires_warmup():
    m=api()
    assert hasattr(m,'ExposureHistory'), 'Actual delayed observation history missing'
    history=m.ExposureHistory(1)
    first=dict(stamp=1.,state=np.zeros(8),previous=.09)
    second=dict(stamp=1.05,state=np.ones(8),previous=.08)
    history.append(first);history.append(second)
    chosen=history.take()
    assert chosen['stamp']==1. and chosen['previous']==.09
    np.testing.assert_array_equal(chosen['state'],np.zeros(8))
    history.reset();history.append(dict(stamp=5.,state=np.ones(8),previous=.095))
    with pytest.raises(ValueError,match='warmup'):history.take()


def test_suite_and_cases_bind_xml_all_assets_and_statistics(tmp_path):
    m=api();root=tmp_path/'source';run=tmp_path/'run'
    for directory in [root/'simulation/assets',root/'evaluation',run/'checkpoints']:
        directory.mkdir(parents=True,exist_ok=True)
    files=[root/'simulation/scene_template.xml',root/'simulation/assets/finger.obj',
           root/'simulation/assets/cup.stl',root/'evaluation/protocol.json',
           run/'checkpoints/policy_best_val.ckpt',run/'checkpoints/dataset_stats.pkl']
    for f in files:f.write_bytes(b'original')
    assert hasattr(m,'evaluation_dependencies'), 'Shared physical dependency binding missing'
    bindings=m.file_hashes(m.evaluation_dependencies(root,run))
    assert all(str(f.resolve()) in bindings for f in files)
    files[1].write_bytes(b'changed_geometry')
    with pytest.raises(RuntimeError,match='changed'):m.require_files(bindings)
