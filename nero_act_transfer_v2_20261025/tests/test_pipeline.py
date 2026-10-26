def test_pipeline_waits_for_all_qualified_data_and_does_not_retry_failed_pilot():
    from pipeline import next_stage
    assert next_stage({'complete':False,'episodes':[],'rejected':[]},None,None)=='collecting'
    assert next_stage({'complete':False,'episodes':[],'rejected':[{}]},None,None)=='blocked_data'
    complete={'complete':True,'episodes':[{}]*150,'rejected':[]}
    assert next_stage(complete,None,None)=='pilot'
    assert next_stage(complete,{'complete':True,'mechanism_passed':False},None)=='blocked_pilot'
    assert next_stage(complete,{'complete':True,'mechanism_passed':True},None)=='formal'
    assert next_stage(complete,{'complete':True,'mechanism_passed':True},{'complete':True})=='awaiting_real_data_or_development_result'
def test_pipeline_can_wait_for_the_versioned_requalified_manifest(tmp_path):
    from pipeline import parse_arguments
    path=tmp_path/'qualified.json'
    assert parse_arguments(['--manifest',str(path)]).manifest==path


def test_limited_pilot_must_be_revalidated_without_another_training_budget():
    from pipeline import next_stage
    complete={'complete':True,'episodes':[{}]*150,'rejected':[]}
    limited={'complete':True,'mechanism_passed':True,'admission_scope':'mechanism_pilot_only'}
    assert next_stage(complete,limited,None)=='revalidate_pilot'
    assert next_stage(complete,limited,None,{'complete':True,'mechanism_passed':False,'admission_allowed':False})=='blocked_pilot_revalidation'
    assert next_stage(complete,limited,None,{'complete':True,'mechanism_passed':True,'admission_allowed':True})=='formal'
    assert next_stage(complete,limited,None,{'complete':True,'mechanism_passed':True,'admission_allowed':False})=='blocked_pilot_revalidation'

