"""A current scoped experimental receipt cannot override a failed model evaluation."""
import hashlib
import math
from pathlib import Path
from executor.evaluation_integrity import evaluation_sources,SUPERVISED_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'complete_pick_carry_place_retreat'
CASES = {'nominal', *[f'independent_{i}' for i in range(8)],
         'cadence_11', 'delay_100ms', 'delay_500ms', 'force_0', 'force_1'}


def require_experimental_permission(metadata, evaluation, receipt, *, now):
    rows = evaluation.get('results', [])
    allowed = (
        metadata.get('task_scope') == evaluation.get('task_scope') == receipt.get('scope') == SCOPE
        and metadata.get('action_chunk_shape') == [20, 8]
        and metadata.get('checkpoint_sha256') == evaluation.get('checkpoint_sha256') == receipt.get('checkpoint_sha256')
        and all(bool(metadata.get(key)) and metadata[key] == evaluation.get(key) == receipt.get(key)
                for key in ['stats_sha256', 'scene_sha256', 'training_config_sha256'])
        and receipt.get('confirmation') == 'ACT_FULL' and receipt.get('consumed') is True
        and receipt.get('experimental_full_task_trial') is True
        and bool(metadata.get('scene_sha256')) and metadata['scene_sha256'] == receipt.get('scene_sha256')
        and all(receipt.get(key) is True for key in ['operator_at_estop', 'empty_cup',
                                                   'swept_volume_clear', 'sole_control_confirmed'])
        and math.isfinite(now) and math.isfinite(receipt.get('authorized_wall_s', float('nan')))
        and 0 <= now-receipt['authorized_wall_s'] <= 1800
        and evaluation.get('teacher_free') is True and evaluation.get('complete') is True
        and evaluation.get('finalized') is True
        and evaluation.get('supervised_contact_protocol') == SUPERVISED_PROTOCOL
        and set(evaluation.get('source_sha256', {})) == {
            str(p.relative_to(ROOT)) for p in evaluation_sources()}
        and evaluation.get('total') == evaluation.get('passed') == len(rows) == 14
        and {r.get('scenario', {}).get('name') for r in rows} == CASES
        and all(r.get('success') is True and r.get('error') is None
                and r.get('physical_success') is True and r.get('teacher_free') is True
                and r.get('premature_release_before_support') is False
                and r.get('constraint_attachment_used') is False
                and r.get('supervision_failures') == 0
                and r.get('execution_contract_failures') == 0
                and r.get('unexpected_arm_cup_contact_frames') == 0 for r in rows))
    if not allowed:
        raise PermissionError('Current model, complete14/14 evaluation and one scoped ACT_FULL receipt are required')
    for name, expected in evaluation.get('source_sha256', {}).items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise PermissionError(f'Frozen evaluation source no longer matches: {name}')
