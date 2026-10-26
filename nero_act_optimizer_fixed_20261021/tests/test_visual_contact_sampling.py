from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.test_visual_dataset import episode


def test_stable_close_history_targets_are_retained_without_handwritten_closure(tmp_path):
    from training.visual import visual_windows
    p=episode(tmp_path/'episode_0.hdf5')
    with h5py.File(p,'r+') as f:
        f['simulation/stage']=np.array(['close']*25,dtype='S8')
        f['action'][:,7]=.065
        original=f['action'][:].copy()
    samples=visual_windows([p])
    assert {t for _,t in samples}==set(range(25)), 'Stable3..12mm close frames must not be discarded'
    assert all(samples.count((0,t))==3 for t in range(25))
    with h5py.File(p) as f:np.testing.assert_array_equal(f['action'][:],original)


def test_sampling_excludes_invalid_rgb_and_preserves_full_task_background(tmp_path):
    from training.visual import visual_windows
    p=episode(tmp_path/'episode_0.hdf5')
    with h5py.File(p,'r+') as f:
        f['simulation/stage']=np.array(['approach']*10+['close']*15,dtype='S8')
        f['action'][:,7]=.065;f['observations/visual_valid'][15]=False
    samples=visual_windows([p])
    assert (0,0) in samples and (0,10) in samples and (0,24) in samples
    assert (0,15) not in samples


def test_data_producer_and_trainer_use_the_same_close_windows(tmp_path):
    import training.legacy_visual_data as producer
    assert hasattr(producer,'visual_sampling_windows'), 'Data producer still uses filtered legacy sampler'
    from training.contact_sampling import visual_sampling_windows
    from training.visual import visual_windows
    p=episode(tmp_path/'episode_0.hdf5')
    with h5py.File(p,'r+') as f:
        f['simulation/stage']=np.array(['close']*25,dtype='S8');f['action'][:,7]=.065
    assert producer.visual_sampling_windows is visual_sampling_windows
    assert visual_windows([p])==visual_sampling_windows([p],20)


def test_fresh_resolved_cases_require_exact_pretraining_source_binding():
    import simulation.visual_evaluation as m
    assert hasattr(m,'resolved_case_provenance'), 'Exact numeric independent-case binding missing'
    with pytest.raises(ValueError):m.resolved_case_provenance(dict(resolved_cases=[{}]*16))


def test_stationary_force_building_is_kept_but_repeated_dwell_is_collapsed(tmp_path):
    from training.contact_sampling import visual_sampling_windows
    p=tmp_path/'episode_0.hdf5';n=180
    stages=np.array(['close']*150+['lift']*30,dtype='S8')
    a=np.zeros((n,8),np.float32);a[:,7]=.066
    a[:60,7]=np.linspace(.09,.066,60)
    a[150:,0]=np.arange(1,31)*.001
    w=np.full(n,.074,np.float32);w[:40]=np.linspace(.09,.074,40)
    with h5py.File(p,'w') as f:
        f['simulation/stage']=stages;f['action']=a
        f['observations/gripper_state']=w[:,None];f['observations/qpos']=np.zeros((n,7))
    samples=visual_sampling_windows([p],20)
    close={t for _,t in samples if t<150}
    assert set(range(60))<=close, 'Retain force-building target changes beyond3mm gap'
    assert 149 in close, 'Retain the representative that transitions to lifting'
    assert len(close & set(range(60,149)))<=1, 'Stationary wait must not dominate lift input'
    assert (0,150) in samples and samples.count((0,150))==6
    with h5py.File(p) as f:np.testing.assert_array_equal(f['action'][:],a)


def test_each_assessment_hashes_the_protocol_actually_selected():
    import json,hashlib
    import simulation.visual_evaluation as m
    root=Path(m.ROOT)
    for path in [root/'evaluation/protocol.json',root/'evidence/baseline_development_protocol.json']:
        cases,proof=m.build_cases(json.loads(path.read_text()),json.loads((root/'scene/center.json').read_text()),protocol_path=path)
        assert len(cases)==16
        assert proof['original_protocol_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
