"""Separate, confirmed high-air preparation for the existing complete air demo.

Default plans only. ROS remains the sole CAN sender. No policy limits change.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

AIR = Path('/home/user/wq/nero_aligned_air_sequence_20261016_onsite')


def preparation_rows(start, target, limits):
    a, b, limits = np.asarray(start, float), np.asarray(target, float), np.asarray(limits, float)
    if (a.shape != (8,) or b.shape != (8,) or limits.shape != (7, 2)
        or not np.isfinite(np.r_[a, b, limits.ravel()]).all()
        or np.any(limits[:, 0] >= limits[:, 1])
        or np.any(np.abs(b[:7]-a[:7]) > .05)
        or np.any(np.minimum(a[:7], b[:7]) < limits[:, 0])
        or np.any(np.maximum(a[:7], b[:7]) > limits[:, 1])
        or not 0 <= a[7] <= b[7] <= .095):
        raise ValueError('Preparation requires a small high-air arm transfer and opening an empty gripper')
    duration = max(4., 1.875 * float(np.max(abs(b-a))) / .005)
    if duration > 30:
        raise ValueError('Preparation exceeds 30-second bound')
    count = int(np.ceil(duration/.05))
    u = np.arange(1, count+1, dtype=float)/count
    fraction = 10*u**3-15*u**4+6*u**5
    return (a+(b-a)*fraction[:, None]).tolist()


def preparation_result(start, target, states, action_success):
    a, b, q = np.asarray(start), np.asarray(target), np.asarray(states)
    if q.ndim != 2 or q.shape[1] != 8 or len(q) < 5 or not np.isfinite(q).all():
        raise ValueError('Insufficient finite actual preparation feedback')
    corridor = bool(np.all(q[:, :7] >= np.minimum(a[:7], b[:7])-.002)
                    and np.all(q[:, :7] <= np.maximum(a[:7], b[:7])+.002)
                    and np.all(q[:, 7] >= a[7]-.001)
                    and np.all(q[:, 7] <= b[7]+.001))
    arm_error = float(np.max(abs(q[-5:, :7]-b[:7])))
    grip_error = float(np.max(abs(q[-5:, 7]-b[7])))
    return dict(success=bool(action_success and corridor and arm_error <= .0007 and grip_error <= .0005),
                actual_feedback_required=True, action_success=bool(action_success),
                corridor_passed=corridor, last5_arm_error_rad=arm_error,
                last5_gripper_error_m=grip_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(AIR))
    from full_runtime import FullSession, verify_distribution
    from restore_start import confirm_return
    from full_core import FullGate
    from trial_contract import air_trial_allowed
    report = dict(mode='prepare_execute' if args.execute else 'prepare_plan', success=False,
                  execution_goal_submitted=False, scope='Independent noncontact empty-gripper preparation',
                  enable_disable_reset_requested=False, automatic_return=False)
    session = None
    try:
        verify_distribution()
        meta = json.loads((AIR/'model/runtime_metadata.json').read_text())
        summary = json.loads((AIR/'model/safety_summary.json').read_text())
        if not air_trial_allowed(meta):
            raise ValueError('Expected unchanged bounded air model')
        target = meta['trial_profile']['initial_state']
        gate = FullGate(summary, True)
        if gate.startup_issues(target):
            raise ValueError('Target fails unchanged ACT startup checks')
        session = FullSession(args.output, report)
        start = session.inventory()
        if not {'arm_controller', 'gripper_controller'} <= {c['name'] for c in report['controllers'] if c['state']=='active'}:
            raise RuntimeError('Both trajectory controllers must be active')
        rows = preparation_rows(start, target, summary['settings']['joint_limits_rad'])
        # Empty group requests whole-robot collision checks, including both fingers.
        session.validate_block([start]+rows, '')
        session.display_chunk(rows)
        report.update(plan_complete=True, planned_monotonic_s=time.monotonic(),
                      target=target, duration_s=len(rows)/20, checked_points=len(rows)+1)
        (args.output/'preparation_rows.json').write_text(json.dumps([start]+rows, indent=2))
        if args.execute:
            clients = session.clients()
            confirm_return(session, 'PREPARE_AIR',
                '高位空载准备：七轴微调至已知起点，空夹爪缓慢张开至60.118mm；确认扫掠区及急停后输入 PREPARE_AIR：')
            session.ready(start)
            begin = len(session.trace); sent = time.monotonic()
            report['execution_goal_submitted'] = True
            futures = session.send_block(clients, rows)
            while not all(f.done() for f in futures):
                session.spin(.01)
                q = session.actual()
                if (any(not min(start[i], target[i])-.002 <= q[i] <= max(start[i], target[i])+.002 for i in range(7))
                    or not start[7]-.001 <= q[7] <= target[7]+.001):
                    raise RuntimeError('Actual preparation left endpoint corridor')
                expected = rows[min(len(rows)-1, int((time.monotonic()-sent)/.05))]
                if max(abs(q[i]-expected[i]) for i in range(7)) > .004 or abs(q[7]-expected[7]) > .004:
                    raise RuntimeError('Actual preparation tracking error exceeds 4mrad / 4mm')
                if time.monotonic()-sent > report['duration_s']+5:
                    raise TimeoutError('Preparation execution timed out')
            results = [session.wait(f) for f in futures]
            session.spin(1.)
            result = preparation_result(start, target, [t['position'] for t in session.trace[begin:]],
                all(r.status == 4 and r.result.error_code == 0 for r in results))
            report['physical_result'] = result
            if not result['success']:
                raise RuntimeError('Physical preparation acceptance failed')
            if gate.startup_issues(session.actual()):
                raise RuntimeError('Actual final preparation fails unchanged ACT startup checks')
        report['success'] = True
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if session:
            if report['execution_goal_submitted'] and not report['success']:
                session.cancel_all()
            (args.output/'actual_feedback.json').write_text(json.dumps(session.trace, indent=2))
            session.node.destroy_node(); session.ros.try_shutdown()
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k:report.get(k) for k in ('mode','success','duration_s','checked_points','physical_result','error')}))
        print('Report:', args.output/'report.json')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
