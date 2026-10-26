import numpy as np
from supervision import remove_wait_aliases


def test_only_conflicting_close_query_starts_are_excluded():
    observed=np.zeros((5,15));actions=np.zeros((5,8));actions[3,0]=.002
    rgb=np.zeros((5,2,2,3),np.uint8);stages=np.array(['close','close','approach','lift','release'])
    observed[1,7]=.001
    valid=np.ones(5,bool)
    mask,dropped=remove_wait_aliases(observed,actions,rgb,stages,valid,3)
    assert dropped==[0] and mask.tolist()==[False,True,True,True,True]
    assert valid.all() and actions[0,0]==0
