import json,time,threading
from pathlib import Path
import h5py,numpy as np,pytest,torch
from test_data import episode


def test_simulation_nacks_and_unproven_initial_context_are_rejected(tmp_path):
    from data_audit import audit_episode
    p=tmp_path/'nack.h5';episode(p)
    with h5py.File(p,'r+') as f:f['action_acknowledged'][0]=False
    assert not audit_episode(p)['main15_ready']
    p=tmp_path/'initial.h5';episode(p)
    with h5py.File(p,'r+') as f:del f.attrs['initial_previous_source']
    assert not audit_episode(p)['main15_ready']


def test_second_prefix_point_uses_same_previous_and_reports_location(tmp_path):
    from data_audit import audit_episode
    p=tmp_path/'two.h5';episode(p)
    with h5py.File(p,'r+') as f:f['action'][:,7]=.095-.0008*np.arange(1,25)
    result=audit_episode(p)
    assert not result['main15_ready']
    assert result['invalid_prefix']['query_frame']==0 and result['invalid_prefix']['point']==1


def test_spatial_admission_checks_actual_scene_and_hdf_metadata(tmp_path):
    from train import spatial_errors
    from protocol import plan_positions
    p=tmp_path/'one.h5';episode(p)
    row=dict(path=str(p),split='train',position_group='train_00')
    base={'pick_xy_m':[.4,0]};protocol=plan_positions()
    with h5py.File(p,'r+') as f:
        f.attrs.update(split='train',position_group='train_00',
                       scene_config=json.dumps({'pick_xy_m':[.4,0]}))
    assert spatial_errors(row,base,protocol)
    with h5py.File(p,'r+') as f:
        f.attrs['scene_config']=json.dumps({'pick_xy_m':[.37,-.03]})
    assert not spatial_errors(row,base,protocol)
    row['position_group']='train_fake'
    assert spatial_errors(row,base,protocol)


def test_complete_predicate_requires_fresh_teacher_free_provenance():
    from evaluate import full_task_passed
    from test_evaluate import good_result
    for extra in [dict(resumed_partial_not_full=True),dict(teacher_free=False),dict(full_task_from_fresh_start=False)]:
        assert not full_task_passed(dict(good_result(),**extra))
    r=good_result();del r['teacher_free'];assert not full_task_passed(r)


def test_validation_ignores_unexecuted_gripper_and_selection_keeps_eligibility():
    from train import deployment_error,select_candidate
    row=dict(observation=torch.zeros(15),image=torch.zeros(1,3,4,4),
             target=torch.zeros(20,8),padding=torch.zeros(20,dtype=torch.bool))
    class P:
        device='cpu'
        def eval(self):return self
        def __call__(self,obs,image):
            out=torch.zeros(len(obs),20,8);out[:,2:,7]=100
            return out
    assert deployment_error(P(),[row])==0
    records=[dict(passed=6,adaptation_gate_passed=True,deployment_error=2,phase_progress=[1]*4),
             dict(passed=6,adaptation_gate_passed=False,deployment_error=0,phase_progress=[1]*4)]
    assert select_candidate(records)==records[0]


def test_trace_close_is_bounded_when_queue_full_and_writer_stalled():
    from runtime import AsyncTrace
    gate=threading.Event();entered=threading.Event()
    def writer(_):entered.set();gate.wait(2)
    trace=AsyncTrace(writer,capacity=1)
    trace.append(1);assert entered.wait(1);trace.append(2)
    started=time.monotonic()
    try:
        with pytest.raises(RuntimeError):trace.close(timeout_s=.03)
        assert time.monotonic()-started<.2 and trace.closed
    finally:gate.set();trace.thread.join(timeout=2)


def test_development_suite_keeps_all_thirty_and_seven_historical_cases(tmp_path):
    from evaluate import development_suite
    paths=[]
    for i in range(30):
        p=tmp_path/f'dev_{i}.h5'
        with h5py.File(p,'w') as f:f.attrs['scene_config']=json.dumps({'pick_xy_m':[.4,0]})
        paths.append(dict(path=str(p),split='dev'))
    cases=development_suite(dict(episodes=paths))
    assert sum(c['role']=='full_development' for c in cases)==30
    assert sum(c['role']=='historical_regression' for c in cases)==7
    assert len(cases)==44


def test_final_positions_exclude_every_selected_development_case():
    from protocol import validate_final_positions,plan_positions
    root=Path(__file__).resolve().parents[1]
    final=json.loads((root/'final_position_protocol.json').read_text())
    historical=json.loads((root/'historical_development_cases.json').read_text())
    assert validate_final_positions(final,plan_positions(),historical)
    final['positions_mm'][0]=[7,-12]
    with pytest.raises(ValueError):validate_final_positions(final,plan_positions(),historical)
