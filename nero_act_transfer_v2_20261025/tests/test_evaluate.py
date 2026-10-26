def good_result():
    return dict(success=True,error=None,max_lift_m=.2,maximum_horizontal_transfer_m=.18,
                stable_released_frames=101,unexpected_arm_cup_contact_frames=0,
                execution_contract_failures=0,premature_release_before_support=False,
                active_policy_points=3000,simulated_events=[{'kind':x} for x in ['GRASP','CARRY','RELEASE','RETREAT']],
                constraint_attachment_used=False,teacher_free=True,full_task_from_fresh_start=True)


def test_grasp_only_or_forced_attachment_cannot_pass_complete_task():
    from evaluate import full_task_passed
    r=good_result();assert full_task_passed(r)
    r['simulated_events']=r['simulated_events'][:1]
    assert not full_task_passed(r)
    r=good_result();r['constraint_attachment_used']=True
    assert not full_task_passed(r)
    r=good_result();r['active_policy_points']=3901
    assert not full_task_passed(r)


def test_development_cases_do_not_consume_final_test_positions():
    from evaluate import development_cases
    from protocol import plan_positions
    cases=development_cases()
    assert len(cases)==7
    final={tuple(x) for x in plan_positions()['positions_mm']['test']}
    assert not final.intersection(tuple(r['offset_mm']) for r in cases)


def test_protected_runner_import_does_not_require_robot_sdk():
    import sys
    from evaluate import load_runner
    runner=load_runner()
    assert callable(runner)
    assert 'pyAgxArm' not in sys.modules and 'rclpy' not in sys.modules


def test_rollout_owns_new_case_directory(tmp_path,monkeypatch):
    import evaluate,json,h5py
    source=tmp_path/'source.h5'
    with h5py.File(source,'w') as f:
        f.attrs['scene_config']=json.dumps({'pick_xy_m':[.4,0],
            'initial_robot_state':{'qpos_rad':[0]*7}})
    calls=[]
    def runner(scene,policy,stats,scenario,output,**kwargs):
        output.mkdir(exist_ok=False)
        calls.append(output.name)
        return good_result()
    monkeypatch.setattr(evaluate,'load_runner',lambda:runner)
    dev=[]
    for i in range(30):
        p=tmp_path/f'dev{i}.h5'
        with h5py.File(p,'w') as f:f.attrs['scene_config']=json.dumps({'pick_xy_m':[.4,0]})
        dev.append(dict(path=str(p),split='dev'))
    report=evaluate.evaluate_development(None,None,{'source_scene':str(source),'episodes':dev},tmp_path/'eval')
    assert len(calls)==44 and report['passed']==44
