import copy
import pytest
from tests.test_additional_contact_recovery import extended


def two_additional():
    data,split=extended();data['additional_failure_ids']=[2018,2019]
    data['episodes'].append(dict(data['episodes'][-1],id=2019,role='approach',context_sha256='safe_approach_failure'))
    return data,split


def test_approach_recovery_is_explicit_training_only_and_preserves_original_roles():
    from training.history_recoveries import validate_history_manifest
    data,split=two_additional()
    groups=validate_history_manifest(data,split,check_files=False,additional_failure_ids=[2018,2019])
    assert len(groups['train_groups'])==37 and len(groups['val_groups'])==12
    for wrong in [[],[2018]]:
        with pytest.raises(ValueError):validate_history_manifest(data,split,check_files=False,additional_failure_ids=wrong)
    data['episodes'][-1]['role']='pre_close'
    with pytest.raises(ValueError):validate_history_manifest(data,split,check_files=False,additional_failure_ids=[2018,2019])


def test_next_extension_preserves_all67_records_and_renders_only_new_failure():
    from training.append_history_recovery import validate_parent_extension
    parent,split=extended();data,split=two_additional()
    base=dict(complete=True,full_demonstrations=48,partial_recoveries=19,episodes=parent['base_episodes']+parent['episodes'])
    assert [r['id'] for r in validate_parent_extension(base,parent,data)]==[2019]
    changed=copy.deepcopy(data);changed['episodes'][-2]['sha256']='rewritten2018'
    with pytest.raises(ValueError):validate_parent_extension(base,parent,changed)
