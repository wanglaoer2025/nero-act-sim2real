"""Training-reference-only projection. No optimizer updates or hardware paths."""
import math

import torch


def project_rows(original, trained, assess):
    if (original.ndim != 2 or original.shape[0] != 7 or trained.shape != original.shape
            or not torch.isfinite(original).all() or not torch.isfinite(trained).all()):
        raise ValueError('Finite paired seven-row arm parameters required')
    def allowed(result):
        return (result.get('samples') == 423
                and all(math.isfinite(result.get(k, math.inf)) for k in
                        ['maximum_arm_prefix_change_rad', 'rms_arm_prefix_change_rad'])
                and 0 <= result['maximum_arm_prefix_change_rad'] <= .0003
                and 0 <= result['rms_arm_prefix_change_rad'] <= .0001)
    if not allowed(assess(original)):
        raise ValueError('The unmodified reference fails original approach bounds')
    trials = []
    for power in range(13):
        factor = 2.**(-power)
        proposed = original+(trained-original)*factor
        result = assess(proposed)
        trials.append(dict(factor=factor, assessment=result))
        if allowed(result):
            return proposed, dict(factor=factor, assessment=result, trials=trials,
                                  reference_role='training423_only', optimizer_updates=0)
    raise ValueError('No bounded nonzero update satisfies original approach limits')


def load_projected():
    import json
    from arm_tuning import ROOT, load_tuned
    from training.integrity import require_unchanged
    run = ROOT/'training/projected_A'
    proof = json.loads((run/'integrity.json').read_text())
    if (proof.get('complete') is not True or proof.get('actual_optimizer_updates') != 1000
            or proof.get('additional_optimizer_updates') != 0
            or proof.get('hardware_deployment_allowed') is not False):
        raise ValueError('Sealed offline projection proof required')
    require_unchanged(proof['source_sha256'])
    require_unchanged(proof['checkpoint_sha256'])
    payload = torch.load(run/'policy_last.ckpt', map_location='cpu', weights_only=False)
    if payload.get('profile') != 'projected_increment_arm_rows_v1':
        raise ValueError('Wrong projected model profile')
    policy, stats = load_tuned()
    head = policy.base.model.action_head
    with torch.no_grad():
        head.weight[:7].copy_(payload['arm_weight'].to(head.weight))
        head.bias[:7].copy_(payload['arm_bias'].to(head.bias))
    return policy.eval(), stats


def main():
    import json
    from pathlib import Path
    from arm_tuning import ROOT, INCREMENT, load_tuned
    from pipeline import load_policy
    from training.integrity import snapshot, require_unchanged
    from training.approach_preservation import ApproachPreservation
    torch.set_num_threads(2)
    policy, stats = load_tuned()
    baseline, _ = load_policy()
    head, old = policy.base.model.action_head, baseline.base.model.action_head
    original = torch.cat([old.weight[:7], old.bias[:7, None]], 1).detach().clone()
    trained = torch.cat([head.weight[:7], head.bias[:7, None]], 1).detach().clone()
    keeper = ApproachPreservation(policy.config['approach_preservation'], stats, 'cuda')
    def assess(rows):
        with torch.no_grad():
            head.weight[:7].copy_(rows[:, :-1])
            head.bias[:7].copy_(rows[:, -1])
        return keeper.assess(policy)
    chosen, report = project_rows(original, trained, assess)
    output = ROOT/'training/projected_A'
    output.mkdir(exist_ok=False)
    torch.save(dict(profile='projected_increment_arm_rows_v1', arm_weight=chosen[:, :-1].cpu(),
                    arm_bias=chosen[:, -1].cpu(), actual_optimizer_updates=1000,
                    additional_optimizer_updates=0), output/'policy_last.ckpt')
    report.update(hardware_deployment_allowed=False, full_task_passed=False)
    (output/'projection.json').write_text(json.dumps(report, indent=2)+'\n')
    files = [Path(__file__), ROOT/'tests/test_arm_projection.py', ROOT/'training/pilot_A/policy_last.ckpt',
             ROOT/'training/pilot_A/integrity.json', output/'projection.json',
             INCREMENT/'training/pilot_A/policy_last.ckpt', INCREMENT/'training/pilot_A/integrity.json']
    hashes = snapshot(files)
    proof = dict(complete=True, actual_optimizer_updates=1000, additional_optimizer_updates=0,
                 source_sha256=hashes, checkpoint_sha256=snapshot([output/'policy_last.ckpt']),
                 hardware_deployment_allowed=False)
    (output/'integrity.json').write_text(json.dumps(proof, indent=2)+'\n')
    require_unchanged(hashes)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
