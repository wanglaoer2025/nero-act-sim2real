"""Explicit fixed-trial gate entry; all original watchdog/contact logic is reused."""
from executor.gate import main
from executor.fixed_permission import require_fixed_trial_permission

if __name__=='__main__':main(permission_check=require_fixed_trial_permission)
