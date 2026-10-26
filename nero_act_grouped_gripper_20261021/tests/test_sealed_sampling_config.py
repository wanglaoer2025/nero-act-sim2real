import json
import pytest


@pytest.mark.parametrize('stage',['pilot','formal'])
@pytest.mark.parametrize('override',[None,'none'])
def test_caller_cannot_replace_the_sealed_sampling_profile(tmp_path,monkeypatch,stage,override):
    import training.history as module
    import executor.history_qualification as qualification
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'_require_seal',lambda path:{})
    (tmp_path/'config').mkdir();(tmp_path/'config/history_experiment.json').write_text(json.dumps(
        dict(transition_sampling_profile='recovery_first5_quarter_v1')))
    def unexpected_gate(*args):raise AssertionError('Unqualified config reached downstream formal gate')
    monkeypatch.setattr(qualification,'require_formal_gate',unexpected_gate)
    config=dict(variant='B',stage=stage)
    if override is not None:config['transition_sampling_profile']=override
    with pytest.raises(ValueError,match='sealed'):
        module.train_history(tmp_path/'data',tmp_path/'initializer',config,tmp_path/'output')
    assert not (tmp_path/'output').exists()
