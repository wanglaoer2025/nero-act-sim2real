import numpy as np
import pytest


def test_diagnostic_interventions_preserve_current_and_never_enter_worker():
    from simulation.history_intervention import isolated_interventions
    original=np.arange(75,dtype=np.float32).reshape(5,15)
    values=isolated_interventions(original)
    assert set(values)=={'unaltered','past_masked','past_reversed'}
    for value in values.values():np.testing.assert_array_equal(value[-1],original[-1])
    np.testing.assert_array_equal(values['past_masked'][:4],0)
    np.testing.assert_array_equal(values['past_reversed'][:4],original[:4][::-1])
    np.testing.assert_array_equal(original,np.arange(75,dtype=np.float32).reshape(5,15))
    with pytest.raises(ValueError):isolated_interventions(original[:4])
