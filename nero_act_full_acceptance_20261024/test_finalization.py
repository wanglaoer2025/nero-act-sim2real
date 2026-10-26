import copy
import json
from pathlib import Path

import pytest

from finalize_existing import finalize_summary

ROOT=Path(__file__).resolve().parent/'run_01'


def test_finalize_all_saved_cases_without_changing_incomplete_raw_summary():
    raw=json.loads((ROOT/'summary.json').read_text())
    before=copy.deepcopy(raw)
    expected=json.loads((ROOT/'protocol.json').read_text())['expected_case_ids']
    result=finalize_summary(raw,expected)
    assert raw==before and raw['complete'] is False
    assert result['complete'] and result['total']==7 and result['passed']==1
    assert not result['full_suite_passed'] and not result['in_memory_policy_comparison_completed']


@pytest.mark.parametrize('mode',['missing','duplicate'])
def test_finalization_rejects_incomplete_or_duplicate_saved_cases(mode):
    raw=json.loads((ROOT/'summary.json').read_text())
    expected=json.loads((ROOT/'protocol.json').read_text())['expected_case_ids']
    if mode=='missing':raw['records'].pop()
    else:raw['records'][-1]=raw['records'][0]
    with pytest.raises(ValueError):finalize_summary(raw,expected)
