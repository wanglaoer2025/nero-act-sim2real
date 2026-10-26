import importlib
import importlib.util
import pytest


def api():
    assert importlib.util.find_spec('executor.history_qualification'), 'Pilot/development qualification graph missing'
    return importlib.import_module('executor.history_qualification')


def test_formal_training_cannot_start_without_completed_pilot_mechanisms(tmp_path):
    m=api()
    with pytest.raises(ValueError,match='pilot|mechanism'):m.require_formal_gate(tmp_path)


def test_independent_cases_cannot_start_without_development_qualification(tmp_path):
    m=api()
    with pytest.raises(ValueError,match='candidate|development'):
        m.require_independent_candidate(tmp_path,tmp_path/'training/formal_B')


def test_mechanism_summary_cannot_be_substituted_for_another_checkpoint():
    m=api()
    summary=dict(complete=True,checkpoint_sha256='old',source_seal_sha256='seal')
    with pytest.raises(ValueError):m.require_mechanism_binding(summary,'new','seal')


def test_direct_development_summary_uses_unique_bound_case_results(tmp_path):
    import json
    from tests.test_history_evaluation import complete_result
    m=api();checkpoint=tmp_path/'policy.ckpt';checkpoint.write_bytes(b'model')
    path=tmp_path/'summary.json';protocol=tmp_path/'protocol.json'
    names=[f'case_{i}' for i in range(16)]
    results=[dict(complete_result(),scenario={'name':name}) for name in names]
    summary=dict(complete=True,finalized=True,total=16,passed=16,results=results,
        provenance={'assessment_role':'paired_baseline_development_not_independent'})
    protocol.write_text(json.dumps({'dependencies':{str(checkpoint.resolve()):m.sha(checkpoint)}}))
    assert m.development_count(path,summary,checkpoint,names)[0]==16
    summary['results'][-1]['scenario']['name']=names[0]
    with pytest.raises(ValueError):m.development_count(path,summary,checkpoint,names)


@pytest.mark.parametrize('stage',['pilot','development'])
def test_development_requires_successful_bound_prerequisite(tmp_path,stage):
    m=api()
    with pytest.raises(ValueError,match='decision|nominal|prerequisite'):
        m.require_development_gate(tmp_path,tmp_path/'training/run',pilot=stage=='pilot')


@pytest.mark.parametrize('failure',['failed','wrong_checkpoint','valid'])
def test_nominal_gate_checks_actual_result_and_checkpoint(tmp_path,failure):
    import json
    from tests.test_history_evaluation import complete_result
    m=api();run=tmp_path/'training/formal_B';(run/'checkpoints').mkdir(parents=True)
    checkpoint=run/'checkpoints/policy_best_val.ckpt';checkpoint.write_bytes(b'weights')
    (run/'config.json').write_text(json.dumps({'variant':'B'}))
    directory=tmp_path/'results/nominal_B';directory.mkdir(parents=True)
    summary=dict(complete=True,finalized=True,total=1,results=[dict(complete_result(),success=failure!='failed')],
        provenance={'assessment_role':'nominal_development_not_final_test'})
    (directory/'summary.json').write_text(json.dumps(summary))
    (directory/'protocol.json').write_text(json.dumps({'dependencies':{str(checkpoint.resolve()):
        'different' if failure=='wrong_checkpoint' else m.sha(checkpoint)}}))
    if failure=='valid':assert len(m.require_development_gate(tmp_path,run))==2
    else:
        with pytest.raises(ValueError):m.require_development_gate(tmp_path,run)


def test_failed_pilot_decision_cannot_enter_development(tmp_path):
    import json
    m=api();(tmp_path/'evidence').mkdir()
    (tmp_path/'evidence/pre_optimization_freeze.json').write_text('{}')
    (tmp_path/'evidence/pilot_decision.json').write_text(json.dumps({'complete':True,'gate':'stop'}))
    with pytest.raises(ValueError,match='decision'):
        m.require_development_gate(tmp_path,tmp_path/'training/pilot_A',pilot=True)
