"""Zero optimizer updates: bind an existing200-step pilot to qualified full data.

This proves numerical mechanisms only. It cannot certify a physical task or
replace any real-geometry, contact, collision or controller protection.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = 'nero_act_transfer_v2_full_data_pilot_revalidation'
CORE_FILES = ['policy.py', 'dataset.py', 'contract.py', 'train_protocol.json']
GROUPS = [f'train_{i:02d}' for i in range(12)]
STAGES = ['approach', 'descend', 'close', 'lift', 'carry', 'place', 'release', 'retreat']
GRADIENT_PREFIXES = ['model.encoder_action_proj', 'model.latent_proj',
    'model.transformer.decoder.layers.0', 'model.transformer.decoder.layers.1',
    'model.backbones.0.0.body.layer4', 'gripper_head.2']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def admission_errors(proof_path, manifest_path, binding_path, *, root=ROOT):
    """The current full-data preflight is independently required by train.py."""
    try:
        proof = json.loads(Path(proof_path).read_text())
        paired = (proof.get('data_manifest_sha256') == sha(manifest_path)
                  and proof.get('implementation_binding_sha256') == sha(binding_path))
        if proof.get('schema') != SCHEMA:
            valid = (proof.get('complete') is True and proof.get('updates') == 200
                and proof.get('formal') is False and proof.get('mechanism_passed') is True
                and proof.get('full_main_data_admission') is True and paired)
            return [] if valid else ['qualified200_step_pilot_missing_or_changed']
        if not (proof.get('complete') is True and proof.get('optimizer_updates') == 0
                and proof.get('mechanism_passed') is True
                and proof.get('full_main_data_admission') is True and paired):
            return ['full_data_revalidation_incomplete_or_changed']
        for key in ['source_pilot_completion', 'source_checkpoint', 'source_implementation', 'observer_equivalence']:
            item = proof[key]
            if sha(item['path']) != item['sha256']:
                return ['revalidation_source_changed:' + key]
        source = json.loads(Path(proof['source_pilot_completion']['path']).read_text())
        if not (source['complete'] is True and source['updates'] == 200 and source['formal'] is False
                and source['mechanism_passed'] is True
                and source['implementation_binding_sha256'] == proof['source_implementation']['sha256']):
            return ['original200_step_pilot_not_proven']
        old = json.loads(Path(proof['source_implementation']['path']).read_text())['sha256']
        if any(sha(Path(root) / name) != old[name] for name in CORE_FILES):
            return ['training_core_changed_since200_step_pilot']
        equivalence=json.loads(Path(proof['observer_equivalence']['path']).read_text())
        if (equivalence.get('compatible') is not True
                or equivalence.get('new_observer_sha256')!=sha(Path(root)/'observer.py')):
            return ['old_success_observations_not_equivalent']
        records = proof['records']
        keys = [(r['position_group'], r['stage']) for r in records]
        if len(keys) != 36 or set(keys) != set(itertools.product(GROUPS, ['close', 'carry', 'release'])):
            return ['all12_groups_three_mechanisms_required']
        for record in records:
            delta = record['net_target_change_m']
            good = (delta < -.0001 if record['stage'] == 'close' else
                    delta > .0001 if record['stage'] == 'release' else abs(delta) < .001)
            if not (record['passed'] is True and record['queries'] == 10 and math.isfinite(delta) and good):
                return ['full_data_mechanism_failed']
        cases = []
        for batch in proof['gradient_batches']:
            cases.extend(batch['cases'])
            if any(not math.isfinite(batch['norms'][p]) or batch['norms'][p] <= 0 for p in GRADIENT_PREFIXES):
                return ['fresh_full_data_gradient_failed']
        expected = {g + '/' + s for g, s in itertools.product(GROUPS, STAGES)}
        if len(cases) != 96 or set(cases) != expected:
            return ['all12_groups_eight_stage_gradient_coverage_required']
        return []
    except (OSError, KeyError, TypeError, ValueError):
        return ['pilot_proof_unreadable_or_unbound']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from torch.utils.data import default_collate
    from dataset import V2Dataset, fit_statistics
    from policy import ACTPolicy
    from train import preflight, requery_probes
    report = preflight(args.manifest, mode='check')
    (args.output / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['allowed'] or not report['full_main_data_ready']:
        return 2
    source_path = ROOT / 'training/pilot/completion.json'
    source = json.loads(source_path.read_text())
    binding_path = ROOT / 'evidence/implementation_manifest.json'
    old_binding = ROOT / 'evidence/implementation_manifest_before_cup_geometry.json'
    checkpoint_path = ROOT / 'training/pilot/pilot.ckpt'
    result = dict(schema=SCHEMA, complete=False, optimizer_updates=0, full_main_data_admission=True,
        full_task_passed=False, hardware_deployment_allowed=False, hardware_commands_sent=False,
        source_pilot_scope=source['admission_scope'], data_manifest_sha256=sha(args.manifest),
        implementation_binding_sha256=sha(binding_path), records=[], gradient_batches=[])
    for key, path in [('source_pilot_completion', source_path), ('source_checkpoint', checkpoint_path),
                      ('source_implementation', old_binding),
                      ('observer_equivalence', ROOT / 'evidence/contour_observer_equivalence.json')]:
        result[key] = dict(path=str(path), sha256=sha(path))
    old_core = json.loads(old_binding.read_text())['sha256']
    if (source['updates'] != 200 or source['mechanism_passed'] is not True
            or source['implementation_binding_sha256'] != sha(old_binding)
            or any(sha(ROOT / name) != old_core[name] for name in CORE_FILES)):
        raise ValueError('Original200 numerical pilot/core no longer matches')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['updates'] != 200 or checkpoint['mode'] != 'pilot':
        raise ValueError('Actual checkpoint is not the original200-step pilot')
    manifest = json.loads(args.manifest.read_text())
    representatives = {g: next(r['path'] for r in sorted(manifest['episodes'], key=lambda r: r['path'])
                               if r['position_group'] == g) for g in GROUPS}
    policy = ACTPolicy(checkpoint['stats']).cuda().eval()
    policy.load_state_dict(checkpoint['model'])
    if policy.config != checkpoint['config']:
        raise ValueError('Pilot configuration changed')
    for group, path in representatives.items():
        checks = requery_probes(policy, V2Dataset([path], checkpoint['stats']))
        for row in checks['records']:
            result['records'].append(dict(**row, queries=10, position_group=group))
        print(json.dumps(dict(group=group, mechanism_passed=checks['passed'], optimizer_updates=0)), flush=True)
    del policy
    stats = fit_statistics([r['path'] for r in manifest['episodes'] if r['split'] == 'train'])
    dataset = V2Dataset(list(representatives.values()), stats)
    chosen = []
    for i, (group, path) in enumerate(representatives.items()):
        for stage in STAGES:
            candidates = [j for j, (p, _) in enumerate(dataset.samples) if p == i and dataset.stages[j] == stage]
            if not candidates:
                raise ValueError('Full-stage representative missing:' + group + '/' + stage)
            chosen.append((group + '/' + stage, candidates[len(candidates) // 2]))
    fresh = ACTPolicy(stats).cuda().train()
    for offset in range(0, len(chosen), 32):
        chunk = chosen[offset:offset + 32]
        batch = default_collate([dataset[i] for _, i in chunk])
        tensors = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        fresh.zero_grad(set_to_none=True)
        losses = fresh(tensors['observation'], tensors['image'], tensors['target'], tensors['padding'])
        if not torch.isfinite(losses['loss']):
            raise ValueError('Nonfinite fresh full-data training loss')
        losses['loss'].backward()
        norms = {prefix: sum(float(p.grad.norm()) for n, p in fresh.named_parameters()
                              if n.startswith(prefix) and p.grad is not None)
                 for prefix in GRADIENT_PREFIXES}
        result['gradient_batches'].append(dict(cases=[name for name, _ in chunk], norms=norms))
    result['mechanism_passed'] = all(r['passed'] for r in result['records'])
    result['complete'] = True
    proof = args.output / 'completion.json'
    proof.write_text(json.dumps(result, indent=2) + '\n')
    errors = admission_errors(proof, args.manifest, binding_path)
    for name, digest in json.loads(binding_path.read_text())['sha256'].items():
        if sha(ROOT / name) != digest:
            errors.append('implementation_changed_during_revalidation:' + name)
    (args.output / 'admission.json').write_text(json.dumps(dict(allowed=not errors, blockers=errors,
        optimizer_updates=0, hardware_commands_sent=False), indent=2) + '\n')
    print(json.dumps(dict(allowed=not errors, blockers=errors, optimizer_updates=0)), flush=True)
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
