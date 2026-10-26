def test_missing_rendered_frames_cannot_be_called_complete_visibility():
    from visibility_audit import summarize_visibility
    result=summarize_visibility(100,[dict(frame=i,stage='approach',valid=True) for i in range(30)])
    assert not result['complete_visibility'] and result['audited_frames']==30


def test_one_rejected_release_frame_invalidates_complete_visibility():
    from visibility_audit import summarize_visibility
    rows=[dict(frame=i,stage='release',valid=i!=5,reason='occluded' if i==5 else None) for i in range(10)]
    result=summarize_visibility(10,rows)
    assert not result['complete_visibility']
    assert result['rejected_frames']==1 and result['failure_reasons']=={'occluded':1}
