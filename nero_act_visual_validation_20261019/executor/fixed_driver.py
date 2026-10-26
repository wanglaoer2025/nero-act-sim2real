"""Explicit fixed-trial driver entry; ordinary driver retains its general gate."""
from executor.driver import main
from executor.fixed_permission import require_fixed_trial_permission

if __name__=='__main__':main(permission_check=require_fixed_trial_permission)
