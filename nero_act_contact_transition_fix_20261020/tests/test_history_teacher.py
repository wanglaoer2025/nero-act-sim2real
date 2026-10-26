import copy
import importlib.util
import numpy as np
import pytest


def test_partial_teacher_keeps_the_original_monitor_instead_of_erasing_failure():
    import simulation.teacher as teacher
    assert hasattr(teacher,'resume_teacher_supervision'), 'Full partial-teacher monitor restoration missing'
    from simulation.outcome import OutcomeMonitor
    monitor=OutcomeMonitor([0,0],[.2,0],0)
    old=OutcomeMonitor([0,0],[.2,0],0);old.had_lifted=True;old.carried=True
    old.premature_release=True;old.max_lift=.2
    config=dict(continuation='release_to_retreat',initial_simulation_snapshot={'saved':True},runtime_physics_offset=143)
    teacher.resume_teacher_supervision(monitor,dict(monitor=vars(old)),config)
    assert monitor.premature_release and monitor.had_lifted and monitor.max_lift==.2
    assert teacher.teacher_physics_index(config,2)==145
    with pytest.raises(ValueError):teacher.resume_teacher_supervision(monitor,dict(monitor={}),config)
    with pytest.raises(ValueError):teacher.resume_teacher_supervision(monitor,dict(monitor=vars(old)),{})


def test_interruption_capture_has_no_invented_ACT_failure_or_future_frames():
    assert importlib.util.find_spec('simulation.history_teacher'), 'Teacher-context generator missing'
    from simulation.history_teacher import role_for_stage
    assert role_for_stage('close')=='pre_close'
    assert role_for_stage('lift')=='hold_to_lift'
    assert role_for_stage('release')=='support_release'
    assert role_for_stage('retreat')=='release_retreat'
    assert role_for_stage('approach') is None


def test_recovery_batch_requires_two_actual_nominal_contexts_before_generation(tmp_path):
    import training.history_recoveries as m
    assert hasattr(m,'prepare_recoveries'), 'Recovery generation entry missing'
    with pytest.raises(ValueError,match='nominal|contexts'):
        m.prepare_recoveries(tmp_path/'missing_data',tmp_path/'missing_contexts',tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_teacher_interruption_keeps_first_liftoff_transform_after_landing():
    from types import SimpleNamespace
    import simulation.history_teacher as teacher
    from simulation.supported_attachment import SupportedAttachment
    assert hasattr(teacher,'update_teacher_attachment'), 'Teacher interruption omitted airborne transform tracking'
    a=SupportedAttachment(radius=.0375,half_height=.0475)
    contact=SimpleNamespace(state='holding')
    def pose(z):return dict(position=np.array([0,0,z]),rotation=np.eye(3))
    teacher.update_teacher_attachment(a,pose(.1),np.array([0,0,.0475]),np.eye(3),supported=True,contact=contact)
    teacher.update_teacher_attachment(a,pose(.11),np.array([0,0,.0575]),np.eye(3),supported=False,contact=contact)
    center=a.center.copy()
    teacher.update_teacher_attachment(a,pose(.2),np.array([.01,0,.0475]),np.eye(3),supported=True,contact=contact)
    assert a.airborne is True
    np.testing.assert_array_equal(a.center,center)


def test_context_records_preserve_actual_float64_hold_commands(tmp_path):
    import inspect,h5py
    from simulation.collect import save_records
    assert 'preserve_action_precision' in inspect.signature(save_records).parameters, 'Exact ACT hold precision is lost on save'
    action=np.r_[np.zeros(7),np.nextafter(np.float64(np.float32(.07128669)),np.inf)]
    q=np.zeros(16);q[12]=1
    rows=[dict(qpos=q,qvel=np.zeros(15),action=action.copy(),timestamp=0.,stage='history_context',
               rgb=np.zeros((90,160,3),np.uint8),cup_xyz=np.zeros(3),torque=np.zeros(7))]
    result=dict(task='grasp_to_retreat_correction',success=True,full_task_verified=False)
    save_records(tmp_path/'exact.hdf5',rows,{},result,preserve_action_precision=True)
    with h5py.File(tmp_path/'exact.hdf5') as f:np.testing.assert_array_equal(f['action'][0],action)
    save_records(tmp_path/'legacy.hdf5',rows,{},result)
    with h5py.File(tmp_path/'legacy.hdf5') as f:assert f['action'].dtype==np.dtype('float32')
