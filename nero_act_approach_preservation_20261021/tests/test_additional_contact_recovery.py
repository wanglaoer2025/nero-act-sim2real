import copy
import pytest
from tests.test_history_data import manifest


def extended():
    data,split=manifest()
    data['additional_failure_ids']=[2018]
    data['episodes'].append(dict(data['episodes'][-2],id=2018,context_sha256='new_failure_context'))
    return data,split


def test_additional_failure_must_be_explicitly_declared_and_training_only():
    from training.history_recoveries import validate_history_manifest
    data,split=extended()
    groups=validate_history_manifest(data,split,check_files=False,additional_failure_ids=[2018])
    assert len(groups['train_groups'])==37 and len(groups['val_groups'])==12
    with pytest.raises(ValueError):validate_history_manifest(data,split,check_files=False)
    data['episodes'][-1]['split']='val'
    with pytest.raises(ValueError):validate_history_manifest(data,split,check_files=False,additional_failure_ids=[2018])


def test_new_failure_cannot_duplicate_a_context_or_use_failed_actions_as_labels():
    from training.history_recoveries import validate_history_manifest
    data,split=extended();data['episodes'][-1]['context_sha256']='context'
    with pytest.raises(ValueError):validate_history_manifest(data,split,check_files=False,additional_failure_ids=[2018])
    data,split=extended();data['episodes'][-1]['label_source']='failed_ACT_actions'
    with pytest.raises(ValueError):validate_history_manifest(data,split,check_files=False,additional_failure_ids=[2018])


def test_extension_reuses_originals_and_rejects_rewritten_parent_records():
    from training.append_history_recovery import validate_parent_extension
    recoveries,split=manifest();base=dict(complete=True,full_demonstrations=48,partial_recoveries=18,
        episodes=recoveries['base_episodes']+recoveries['episodes'])
    extra,split=extended()
    assert [r['id'] for r in validate_parent_extension(base,recoveries,extra)]==[2018]
    changed=copy.deepcopy(extra);changed['episodes'][0]['sha256']='rewritten'
    with pytest.raises(ValueError):validate_parent_extension(base,recoveries,changed)
    changed=copy.deepcopy(base);changed['episodes'].pop(0)
    with pytest.raises(ValueError):validate_parent_extension(changed,recoveries,extra)
