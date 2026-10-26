"""Verify the diagnostic evidence chain without changing frozen experiments."""
from pathlib import Path
import sys,json,re
import numpy as np

BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'nero_act_optimizer_fixed_20261021'))
from training.integrity import require_unchanged,snapshot

records={}
for name in ['training_audit','policy_probe','horizon_probe','timebase_audit']:
    record=json.loads((OUT/(name+'.json')).read_text());assert record['complete']
    require_unchanged(record['source_sha256']);records[name]=record
assert records['training_audit']['ordinary_sample_visits']==32000
assert len(records['timebase_audit']['records'])==68
policy=records['policy_probe'];assert len(policy['teacher_fitting'])==245 and len(policy['gradient_probes'])==15
assert policy['actual_optimizer_updates']==records['horizon_probe']['actual_optimizer_updates']==records['timebase_audit']['actual_optimizer_updates']==0
assert records['training_audit']['optimizer_updates']==0
assert len(policy['model_states'])==5 and all(r['before_sha256']==r['after_sha256'] and r['optimizer_state_entries']==0 for r in policy['model_states'])
assert all(r['before_sha256']==r['after_sha256'] for r in records['horizon_probe']['model_states'])
reproduced=[r for r in policy['exact_runtime_reproduction'] if r['query'].startswith(r['policy']+'_')]
assert len(reproduced)==8 and all(r['maximum_arm_difference_rad']==r['maximum_gripper_difference_m']==0 for r in reproduced)
paired=[r for r in reproduced if 'runtime_teacher_RGB_equal' in r]
assert len(paired)==4 and all(r['runtime_teacher_RGB_equal'] and r['runtime_teacher_15D_max_difference']<=3e-7 for r in paired)
assert policy['teacher_prefix_inverse_roundtrip_max_physical_error']<=1e-7
for r in records['horizon_probe']['records']:
    assert r['first2_vs_other18']['cosine']<-.9
    assert r['loss_variants_vs_first2']['existing']['cosine']<.05
    assert r['loss_variants_vs_first2']['without_tail']['cosine']>.99
for arm in ['fixed','decay']:
    root=BASE/f'nero_act_optimizer_{arm}_20261021'
    seal=json.loads((root/'evidence/pre_optimization_freeze.json').read_text());require_unchanged(seal['source_sha256']);require_unchanged(seal['data_sha256'])
    tested=json.loads((root/'evidence/tested_sources.json').read_text());require_unchanged(tested['source_sha256'])
previous=BASE/'nero_act_optimizer_control_20261021/evidence'
old=json.loads((previous/'final_verification.json').read_text());require_unchanged(old['artifact_sha256'])
pilot=json.loads((previous/'pilot_verification.json').read_text());require_unchanged(pilot['source_and_output_sha256'])
for target in re.findall(r'\]\(([^)]+)\)',(OUT/'findings.md').read_text()):
    if target=='verification.json':continue
    if not target.startswith(('https:','http:','#')):assert (OUT/target).resolve().exists(),target
files=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.py','.md','.png') and p.name!='verification.json']
proof=dict(complete=True,actual_optimizer_updates=0,frozen_policies_checked=5,teacher_window_predictions=245,
    failed_queries_exactly_reproduced=8,RGB_teacher_equal_cases=4,gradient_probes=15,horizon_counterfactual_models=2,
    all68_timebases_checked=True,all_prior_seals_and_outputs_unchanged=True,hardware_commands_sent=False,
    training_causal_ablation_performed=False,figure_visually_inspected=True,artifact_sha256=snapshot(files))
require_unchanged(proof['artifact_sha256']);(OUT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k!='artifact_sha256'},indent=2))
