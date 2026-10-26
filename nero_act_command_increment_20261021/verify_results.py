"""Verify the bounded experiment without reinterpreting failed outcomes as success."""
from pathlib import Path
import json,hashlib
import torch,numpy as np
from pipeline import ROOT,PARENT,require_checkpoint,snapshot,require_unchanged
from model import PROFILE
from run_qualification import verify_outcomes


def main():
    run=ROOT/'training/pilot_A';checkpoint=run/'policy_last.ckpt'
    proof=json.loads((run/'integrity.json').read_text());require_checkpoint(proof,checkpoint)
    seal=json.loads((run/'seal.json').read_text())
    for name in ['source_sha256','data_sha256','cache_sha256']:require_unchanged(seal[name])
    assert seal['optimizer_updates_before_seal']==0 and proof['frozen_base_before']==proof['frozen_base_after']==seal['frozen_base_sha256']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['profile']==PROFILE and payload['actual_updates']==1000 and payload['formal_epochs']==0
    assert set(payload['head'])=={'qpos_mean','qpos_std','network.0.weight','network.0.bias','network.2.weight','network.2.bias'}
    trace=json.loads((run/'optimizer_trace.json').read_text());assert len(trace)==1000 and [r['update'] for r in trace]==list(range(1,1001))
    assert all(r['lr']==.0003 and np.isfinite([r['loss'],r['gradient_norm']]).all() for r in trace)
    optimizer=torch.load(run/'optimizer_last.pt',map_location='cpu',weights_only=False);assert len(optimizer['state'])==4
    assert all(int(s['step'])==1000 for s in optimizer['state'].values())
    summary=json.loads((ROOT/'results/qualification/summary.json').read_text());require_unchanged(summary['source_sha256'])
    verify_outcomes(summary,json.loads((ROOT/'results/qualification/input_gate.json').read_text()))
    expected={'nominal_pre_close','nominal_hold_to_lift','heldout_pre_close','heldout_hold_to_lift','heldout_support_release','heldout_release_retreat'}
    assert len(summary['records'])==6 and {r['name'] for r in summary['records']}==expected
    for record in summary['records']:
        folder=ROOT/'results/mechanism_original'/record['name'];result=json.loads((folder/'result.json').read_text())
        assert result==record['result'] and result['teacher_free'] and not result['real_hardware_tested'] and result['resumed_partial_not_full']
        assert result['active_policy_points']<=3900 and result['maximum_ACT_prefix_points']==2
        with np.load(folder/'rollout.npz',allow_pickle=False) as f:assert len(f['states'])==result['frames']
    assert summary['original_six_pass_count']==sum(r['passed'] for r in summary['records'])
    assert summary['nominal_evaluated']==bool(summary['exact_and_preservation_gate_passed'] and summary['original_six_passed'])
    if summary['nominal_evaluated']:
        assert summary['nominal']['result']==json.loads((ROOT/'results/nominal_after/nominal_development/result.json').read_text())
    assert not summary['hardware_commands_sent']
    parent=json.loads((PARENT/'evidence/final_verification.json').read_text());require_unchanged(parent['artifact_sha256'])
    tests=json.loads((ROOT/'evidence/tests_verified.json').read_text());require_unchanged(tests['source_sha256']);assert tests['complete'] and tests['failed']==0
    files=[p for p in ROOT.glob('*.md')]+[p for p in (ROOT/'evidence').glob('*') if p.is_file()]
    files += [p for p in (ROOT/'results').rglob('*') if p.is_file()]
    files += [p for p in run.glob('*') if p.is_file()]
    files=[p for p in files if p.name not in ['final_verification.json','final_verification.log']]
    report=dict(complete=True,verification_not_task_success=True,actual_updates=1000,formal_epochs=0,frozen_base_unchanged=True,only_new_head_trained=True,original_six_pass_count=summary['original_six_pass_count'],full_task_passed=summary['full_task_passed'],hardware_commands_sent=False,artifact_sha256=snapshot(files))
    require_unchanged(report['artifact_sha256']);(ROOT/'evidence/final_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2))

if __name__=='__main__':main()
