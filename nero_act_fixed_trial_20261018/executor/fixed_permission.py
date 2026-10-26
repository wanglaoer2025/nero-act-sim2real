"""Explicit one-candidate fixed-scene exception; never changes the general14/14 gate."""
import hashlib
import json
import math
from pathlib import Path

from executor.evaluation_integrity import evaluation_sources, verify_frozen_artifacts

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'nero_act_fixed_scene_heads_20261018'
MODEL=SOURCE/'model/run27_fixed_heads'
EVALUATION=SOURCE/'evaluation/run27_current_scene14/summary.json'
PROPOSAL=ROOT/'fixed_trial/proposal.json'


def require_fixed_trial_permission(metadata,evaluation,receipt,*,now):
    proposal=json.loads(PROPOSAL.read_text())
    approved=json.loads(EVALUATION.read_text())
    keys=['checkpoint_sha256','stats_sha256','scene_sha256','training_config_sha256']
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    allowed=(canonical(receipt)==canonical(json.loads((ROOT/'fixed_trial/receipt.json').read_text()))
        and metadata.get('task_scope')=='complete_pick_carry_place_retreat'
        and metadata.get('action_chunk_shape')==[20,8]
        and receipt.get('scope')=='one_current_fixed_scene_supervised_ACT_trial'
        and receipt.get('confirmation')=='ACT_FIXED_TRIAL' and receipt.get('consumed') is True
        and all(receipt.get(k) is True for k in ['operator_at_estop','empty_cup','swept_volume_clear',
            'sole_control_confirmed','blue_card_unchanged','accepted_incomplete_evaluation',
            'known_failure_modes_acknowledged','one_physical_trial_only'])
        and receipt.get('general_deployment_accepted') is False
        and receipt.get('speed_percent')==5 and receipt.get('gripper_force_limit_n')==1.
        and receipt.get('proposal_sha256')==hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()
        and receipt.get('evaluation_sha256')==proposal['evaluation_sha256']==hashlib.sha256(EVALUATION.read_bytes()).hexdigest()
        and all(metadata.get(k)==evaluation.get(k)==receipt.get(k)==proposal[k] for k in keys)
        and evaluation==approved and approved['complete'] is True and approved['finalized'] is True
        and approved['passed']==9 and approved['total']==len(approved['results'])==14
        and sum(r['success'] for r in approved['results'])==9
        and all(r.get('teacher_free') is True and r.get('unexpected_arm_cup_contact_frames')==0
            and r.get('execution_contract_failures')==0 and r.get('premature_release_before_support') is False
            and r.get('constraint_attachment_used') is False for r in approved['results'])
        and math.isfinite(now) and math.isfinite(receipt.get('authorized_wall_s',float('nan')))
        and 0<=now-receipt['authorized_wall_s']<=1800)
    required={'nominal','cadence_11','delay_100ms','force_0','force_1'}
    successful={r['scenario']['name'] for r in approved['results']
        if r['success'] and r['physical_success'] and r['error'] is None and r['supervision_failures']==0}
    if not allowed or not required<=successful:
        raise PermissionError('Only the explicitly acknowledged exact fixed-scene candidate and current ACT_FIXED_TRIAL receipt are allowed')
    verify_frozen_artifacts(MODEL/'policy.ckpt',evaluation,source_root=SOURCE)
    names={str(p.relative_to(ROOT)) for p in evaluation_sources()}
    if names!=set(evaluation['source_sha256']):
        raise PermissionError('Fixed trial must retain the original evaluated task and protection sources')
    for name,expected in evaluation['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise PermissionError('Evaluated ACT task/protection changed: '+name)
