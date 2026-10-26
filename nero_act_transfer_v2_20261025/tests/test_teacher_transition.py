from pathlib import Path
import json
import sys
import h5py


def test_physical_teacher_exits_close_after_measured_stable_contact():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
    from simulation.teacher import run_teacher
    source=Path('/home/user/wq/nero_act_new_handeye_policy_repair_20261023/data/physical/episode_9000.hdf5')
    with h5py.File(source) as f:scene=json.loads(f.attrs['scene_config'])
    result=run_teacher(scene)
    assert result['full_task_verified'],result['error']
    assert result.get('teacher_transition_profile')=='measured_event_v2'
    close=[r for r in result['trace'] if r['stage']=='close']
    assert len(close)<140
    assert all(r['bilateral'] for r in close[-10:])
