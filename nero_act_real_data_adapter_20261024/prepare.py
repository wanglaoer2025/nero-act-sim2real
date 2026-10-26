"""Prepare an offline future-state corpus; never starts training or hardware."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py

from adapter import PROFILE, SEMANTICS, convert_episode, file_snapshot, fit_statistics


def assign_splits(project, manifest_path, split_path):
    project = Path(project).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    split = json.loads(Path(split_path).read_text())
    train, val = split['train'], split['val']
    if (any(isinstance(i, bool) or not isinstance(i, int) for i in train + val)
            or len(set(train)) != len(train) or len(set(val)) != len(val)
            or set(train) & set(val)):
        raise ValueError('Duplicate or overlapping train/val episode identifiers')
    selected = {}
    ids = set()
    for row in manifest['episodes']:
        if row['status'] != 'valid':
            continue
        raw = Path(row['path'])
        path = (raw if raw.is_absolute() else project / raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        episode_id = int(raw.stem.removeprefix('episode_'))
        if path in selected or episode_id in ids:
            raise ValueError('Duplicate source alias or episode in selection')
        ids.add(episode_id)
        selected[path] = episode_id
    if ids != set(train) | set(val):
        raise ValueError('Split must cover exactly the existing valid selection')
    return {str(path): ('train' if i in train else 'val') for path, i in selected.items()}


def prepare_data(project, audit_path, output, lead_steps=2):
    project, output = Path(project).resolve(), Path(output).resolve()
    audit_path = Path(audit_path).resolve()
    audit = json.loads(audit_path.read_text())
    if Path(audit['resolved_project']).resolve() != project:
        raise ValueError('Source audit belongs to another project')
    specifications = [
        ('act_dual_fixed/manifest_compatible38.json', 'act_dual_fixed/splits/compatible38_day5_split.json'),
        ('act_dual_fixed_destination/manifest_fixed_destination_9.json',
         'act_dual_fixed_destination/splits/fixed_destination_9_split.json'),
    ]
    assignments, provenance = {}, []
    for manifest_name, split_name in specifications:
        manifest, split = project / 'data' / manifest_name, project / 'data' / split_name
        selected = assign_splits(project, manifest, split)
        if set(assignments) & set(selected):
            raise ValueError('Duplicate source across dataset selections')
        assignments.update(selected)
        for path in (manifest, split):
            provenance.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    sources = [Path(row['path']).resolve() for row in audit['episodes']]
    if len(set(sources)) != len(sources) or not set(assignments) <= set(map(str, sources)):
        raise ValueError('Audit sources must be unique and cover the selected corpus')
    for row, path in zip(audit['episodes'], sources):
        if not path.is_relative_to(project / 'data'):
            raise ValueError('Source must be inside the audited project data directory')
        if file_snapshot(path) != {'size': row['file_bytes'], 'mtime_ns': row['mtime_ns']}:
            raise ValueError('Source changed since the supplied audit: ' + str(path))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'state.json').write_text(json.dumps({'complete': False, 'training_started': False}) + '\n')
    entries = []
    for source, audited in zip(sources, audit['episodes']):
        destination = output / 'episodes' / source.relative_to(project / 'data')
        row = convert_episode(source, destination, lead_steps, assignments.get(str(source), 'reserve'))
        with h5py.File(destination, 'r') as f:
            if f.attrs['source_numeric_sha256'] != audited['numeric_arrays_sha256']:
                raise ValueError('Source numeric values changed since audit')
        entries.append(row)
    stats = fit_statistics([r['path'] for r in entries if r['split'] == 'train'])
    stats_document = {k: v.tolist() for k, v in stats.items()}
    stats_document.update(profile=PROFILE, split='train',
                          source_paths=[r['source_path'] for r in entries if r['split'] == 'train'])
    (output / 'statistics.json').write_text(json.dumps(stats_document, indent=2) + '\n')
    manifest = dict(complete=True, profile=PROFILE, action_semantics=SEMANTICS,
                    original_audit_path=str(audit_path),
                    original_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                    selection_provenance=provenance, lead_steps=lead_steps,
                    split_counts=dict(Counter(r['split'] for r in entries)),
                    evaluation_scope='legacy_episode_holdout_not_position_generalization',
                    independent_position_holdout=False,
                    physical_task_acceptance='not_independently_reviewed',
                    current_visual15_compatible=False, deployment_authorized=False,
                    training_started=False, episodes=entries)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    config = dict(profile=PROFILE, state_dim=8, policy_input_dim=8, action_dim=8,
                  camera_names=['camera_0'], image_size=[180, 320], chunk_size=20,
                  query_stride=4, additional_joint_lookahead_steps=0,
                  statistics='statistics.json', require_source_guard=True,
                  model_initialization='not_selected', training_started=False,
                  current_visual15_checkpoint_compatible=False)
    (output / 'training_view.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'state.json').write_text(json.dumps({'complete': True, 'training_started': False}) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/home/user/projects/nero'))
    parser.add_argument('--audit', type=Path, default=Path(__file__).parents[1] /
                        'docs/evidence/nero-act-review-20261024/user-specified-data-audit.json')
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrites')
    parser.add_argument('--lead-steps', type=int, default=2, help='Future feedback lead; not measured control latency')
    args = parser.parse_args()
    result = prepare_data(args.project, args.audit, args.output, args.lead_steps)
    print(json.dumps({'episodes': len(result['episodes']), 'splits': result['split_counts'],
                      'manifest': str(args.output / 'manifest.json'), 'training_started': False}))


if __name__ == '__main__':
    main()
