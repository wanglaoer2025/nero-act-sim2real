"""One supervised migration-feasibility attempt; no robustness or acceptance claim."""
import hashlib
import math
import json
from pathlib import Path
from executor.evaluation_integrity import evaluation_sources,SUPERVISED_PROTOCOL
from executor.replay_trace import verify_replay
from executor.scene_acceptance import execution_scene

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'complete_pick_carry_place_retreat'
CASES = {'nominal'}


def require_experimental_permission(metadata, evaluation, receipt, *, now):
    rows = evaluation.get('results', [])
    allowed = (
        metadata.get('task_scope') == evaluation.get('task_scope') == receipt.get('scope') == SCOPE
        and metadata.get('action_chunk_shape') == [20, 8]
        and metadata.get('checkpoint_sha256') == evaluation.get('checkpoint_sha256') == receipt.get('checkpoint_sha256')
        and all(bool(metadata.get(key)) and metadata[key] == evaluation.get(key) == receipt.get(key)
                for key in ['stats_sha256', 'scene_sha256', 'training_config_sha256'])
        and receipt.get('confirmation') == 'ACT_REPLAY' and receipt.get('consumed') is True
        and receipt.get('experimental_full_task_trial') is True
        and receipt.get('feasibility_only') is True and receipt.get('single_trial') is True
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
        and evaluation.get('total') == evaluation.get('passed') == len(rows) == 1
        and {r.get('scenario', {}).get('name') for r in rows} == CASES
        and all(r.get('success') is True and r.get('error') is None
                and r.get('physical_success') is True and r.get('teacher_free') is True
                and r.get('premature_release_before_support') is False
                and r.get('constraint_attachment_used') is False
                and r.get('supervision_failures') == 0
                and r.get('execution_contract_failures') == 0
                and r.get('unexpected_arm_cup_contact_frames') == 0 for r in rows))
    if not allowed:
        raise PermissionError('Current model, fresh complete nominal evaluation and one scoped ACT_REPLAY receipt are required')
    manifest,_=verify_replay(ROOT,metadata,receipt)
    execution_scene(ROOT,json.loads((ROOT/'model/current_run27/scene_config.json').read_text()),manifest,
                    require_evidence=True,receipt=receipt)
    for name, expected in evaluation.get('source_sha256', {}).items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise PermissionError(f'Frozen evaluation source no longer matches: {name}')
