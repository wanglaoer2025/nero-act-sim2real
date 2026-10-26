"""Independent empty-gripper recovery to this air model's known opening."""
import json
import time
import numpy as np

from full_core import FullGate
from restore_start import confirm_return
from trial_contract import air_trial_allowed


def gripper_rows(state, profile):
    q = np.asarray(state, dtype=float)
    target = profile['initial_state'][7]
    if (q.shape != (8,) or not np.isfinite(q).all()
        or np.any(q[:7] < profile['arm_lower_rad']) or np.any(q[:7] > profile['arm_upper_rad'])
        or not profile['gripper_lower_m'] <= q[7] <= profile['gripper_upper_m']
        or abs(target-q[7]) > .01):
        raise ValueError('Gripper recovery requires current air envelope and at most 10 mm change')
    count = max(20, int(np.ceil(abs(target-q[7])/.00025)))
    return [[*q[:7].tolist(), float(q[7]+(target-q[7])*i/count)] for i in range(1,count+1)]


def gripper_result(start, target, states, action_success):
    q = np.asarray(states, dtype=float)
    if q.ndim != 2 or q.shape[1] != 8 or len(q) < 5 or not np.isfinite(q).all():
        raise ValueError('Insufficient finite actual gripper recovery feedback')
    drift = float(np.max(abs(q[:,:7]-np.asarray(start[:7]))))
    error = float(np.max(abs(q[-5:,7]-target)))
    corridor = bool(np.all(q[:,7] >= min(start[7],target)-.001)
                    and np.all(q[:,7] <= max(start[7],target)+.001))
    return dict(success=bool(action_success and drift <= .0007 and error <= .0005 and corridor),
                maximum_arm_drift_rad=drift, last5_width_error_m=error,
                actual_feedback_required=True, action_success=bool(action_success))


def run_gripper_return(args, output, report, session_class):
    meta = json.loads((args.model_dir/'runtime_metadata.json').read_text())
    summary = json.loads((args.model_dir/'safety_summary.json').read_text())
    if not air_trial_allowed(meta) or summary.get('trial_profile') != meta['trial_profile']:
        raise ValueError('Unrecognized bounded air profile')
    gate = FullGate(summary, True)
    report.update(mode='gripper_start_execute' if args.execute else 'gripper_start_plan',
                  scope='Separate known empty-gripper recovery, not ACT inference',
                  gripper_command_sent=False)
    s = session_class(output,report)
    try:
        start = s.inventory()
        if not report['execution_readiness_checks_passed']:
            raise RuntimeError(report.get('execution_blocker'))
        gate.state(start)
        rows = gripper_rows(start,meta['trial_profile']); target = rows[-1][7]
        report.update(gripper_start_m=start[7],gripper_target_m=target,duration_s=len(rows)/20)
        if abs(start[7]-target) <= .0005:
            report.update(success=True,already_at_gripper_start=True)
            return
        groups = [g for g,j in report['groups'].items() if len(j)==7 and set(j)=={f'joint{i}' for i in range(1,8)}]
        if len(groups)!=1: raise ValueError('One exact seven-axis arm group required')
        for row in rows: gate.state(row)
        s.validate_block(rows,groups[0]);s.display_chunk(rows)
        (output/'gripper_start_trajectory.json').write_text(json.dumps(rows,indent=2)+'\n')
        report.update(plan_complete=True,planned_monotonic_s=time.monotonic())
        if not args.execute:
            report['success']=True
            return
        clients=s.clients()
        confirm_return(s,'GRIPPER_START','仅恢复空夹爪至训练开口，七轴保持；确认无接触及急停后输入 GRIPPER_START：')
        s.ready(start);begin=len(s.trace);sent=time.monotonic()
        report['execution_goal_submitted']=True
        futures=s.send_block(clients,rows);report['gripper_command_sent']=True
        while not all(f.done() for f in futures):
            s.spin(.005);q=s.actual()
            if max(abs(a-b) for a,b in zip(q[:7],start[:7])) > .0007:
                raise RuntimeError('Arm drift during gripper recovery')
            if not min(start[7],target)-.001 <= q[7] <= max(start[7],target)+.001:
                raise RuntimeError('Gripper recovery left endpoint corridor')
            if time.monotonic()-sent > len(rows)/20+3:
                raise TimeoutError('Gripper recovery timed out')
        results=[s.wait(f) for f in futures];s.spin(.5)
        physical=gripper_result(start,target,[x['position'] for x in s.trace[begin:]],
                                all(r.status==4 and r.result.error_code==0 for r in results))
        report['physical_result']=physical
        if not physical['success']: raise RuntimeError('Actual gripper recovery failed')
        report['success']=True
    finally:
        if report.get('execution_goal_submitted') and not report.get('success'):s.cancel_all()
        (output/'actual_feedback.json').write_text(json.dumps(s.trace,indent=2)+'\n')
        s.node.destroy_node();s.ros.shutdown()
