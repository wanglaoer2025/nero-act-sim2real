"""Fresh provenance and bounded-scope checks after both evaluations and documentation."""
from pathlib import Path
import sys,json,hashlib,re
import numpy as np

BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'nero_act_optimizer_fixed_20261021'))
from training.integrity import require_unchanged,snapshot
from training.optimizer_control import require_optimizer_diagnostics

pilot=json.loads((OUT/'pilot_verification.json').read_text())
require_unchanged(pilot['source_and_output_sha256'])
contact=json.loads((OUT/'nominal_contact_diagnostics.json').read_text());require_unchanged(contact['source_sha256'])
assert contact['complete'] and len(contact['rows'])==4
assert pilot['actual_updates_total']==2000 and pilot['nominal_candidate'] is None
assert all(r['approach_gate_passed'] and r['mechanism']['passed']==4 for r in pilot['reports'].values())
sources=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.md','.py','.png') and p.name!='final_verification.json']
sources += [BASE/'nero_act_sim_to_real_full_process_20261018.md',OUT.parent/'approved_scope.md']
for arm in ['fixed','decay']:
    root=BASE/f'nero_act_optimizer_{arm}_20261021';settings=json.loads((root/'config/history_experiment.json').read_text())
    require_optimizer_diagnostics(settings['optimizer_diagnostics_path'],settings['excluded_ambiguous_windows'])
    tested=json.loads((root/'evidence/tested_sources.json').read_text());require_unchanged(tested['source_sha256'])
    assert tested['complete'] and tested['tests_passed']==421 and tested['tests_skipped']==5
    seal=json.loads((root/'evidence/pre_optimization_freeze.json').read_text())
    require_unchanged(seal['source_sha256']);require_unchanged(seal['data_sha256'])
    assert not (root/'results/nominal_after').exists()
    for folder in ['executor','simulation','calibration','vision','policy_source']:
        for path in (root/folder).rglob('*.py'):
            parent=BASE/'nero_act_approach_preservation_20261021'/path.relative_to(root)
            assert path.read_bytes()==parent.read_bytes()
    mechanism=json.loads((root/'results/mechanism/summary.json').read_text())
    for row in mechanism['records']:
        folder=root/'results/mechanism'/row['name'];result=json.loads((folder/'result.json').read_text())
        assert result==row['result'] and result['resumed_partial_not_full'] and result['teacher_free'] and result['real_hardware_tested'] is False
        with np.load(folder/'rollout.npz',allow_pickle=False) as f:assert len(f['states'])==result['frames']
    sources += [p for p in (root/'results/mechanism').rglob('*') if p.is_file()]
doc=(OUT/'results.md').read_text()
for target in re.findall(r'\]\(([^)]+)\)',doc):
    if not target.startswith(('http:','https:','#')):assert (OUT/target).resolve().exists(),target
proof=dict(complete=True,review_date='2026-10-21',verification_complete_not_task_success=True,
    actual_updates_total=2000,actual_formal_epochs=0,tests_per_arm=dict(passed=421,skipped=5),
    approach_gate_both_passed=True,mechanism_per_arm=dict(passed=4,total=6),nominal_full_rollout_started=False,
    independent16_evaluated=False,hardware_commands_sent=False,core_runtime_matches_parent=True,
    final_review_findings=0,document_links_valid=True,artifact_sha256=snapshot(sources))
require_unchanged(proof['artifact_sha256'])
(OUT/'final_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k!='artifact_sha256'},indent=2))
