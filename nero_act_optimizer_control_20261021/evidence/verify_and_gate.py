"""Verify both bounded pilots and decide qualification without relaxing the approved gate."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np
import torch

BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'nero_act_optimizer_fixed_20261021'))
from training.integrity import require_unchanged,require_finalized,snapshot
from training.history import initialize_history
from training.optimizer_control import arm_rate


def main():
    init=BASE/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
    source=torch.load(init,map_location='cpu',weights_only=False)
    old=pickle.loads((init.parent/'dataset_stats.pkl').read_bytes())
    reports={};bindings=[]
    for arm in ['fixed','decay']:
        root=BASE/f'nero_act_optimizer_{arm}_20261021';run=root/'training/pilot_A'
        seal=json.loads((root/'evidence/pre_optimization_freeze.json').read_text())
        require_unchanged(seal['source_sha256']);require_unchanged(seal['data_sha256'])
        tested=json.loads((root/'evidence/tested_sources.json').read_text());require_unchanged(tested['source_sha256'])
        checkpoint=run/'checkpoints/policy_last.ckpt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False);config=payload['config']
        integrity=require_finalized(checkpoint,config,allowed_budgets=('updates',));proof=json.loads(integrity.read_text())
        assert proof['actual_updates']==1000 and proof['actual_epochs']==0
        stats=pickle.loads((run/'checkpoints/dataset_stats.pkl').read_bytes())
        _,initial=initialize_history(source,old,stats,config,variant='A')
        changed=[k for k,v in payload['model'].items() if not torch.equal(v,initial[k])]
        expected={'model.action_head.weight','model.action_head.bias','gripper_adapter.0.weight','gripper_adapter.0.bias','gripper_adapter.2.weight','gripper_adapter.2.bias'}
        assert set(changed)<=expected and set(changed)&{'model.action_head.weight','model.action_head.bias'}
        for key in ['model.action_head.weight','model.action_head.bias']:assert torch.equal(initial[key][7:],payload['model'][key][7:])
        frozen=json.loads((run/'artifacts/frozen_parameters_verified.json').read_text());assert frozen['before']==frozen['after']
        trace=[json.loads(line) for line in (run/'artifacts/optimizer_trace.jsonl').read_text().splitlines()]
        assert len(trace)==1000 and all(r['update']==i and r['arm_lr']==arm_rate(config['optimizer_control'],i) and r['gripper_lr']==.0003 for i,r in enumerate(trace,1))
        measured=[r for r in trace if 'maximum_arm_update_rad' in r]
        assert [r['update'] for r in measured]==[1,2,25,100,500,1000]
        before=json.loads((run/'artifacts/approach_before.json').read_text());after=json.loads((run/'artifacts/approach_after.json').read_text())
        assert before['samples']==after['samples']==423 and before['maximum_arm_prefix_change_rad']<=1e-6 and before['maximum_gripper_prefix_change_m']<=1e-7
        approach_ok=after['rms_arm_prefix_change_rad']<=.0001 and after['maximum_arm_prefix_change_rad']<=.0003
        mechanism_path=root/'results/mechanism/summary.json'
        mechanism=None
        if mechanism_path.exists():
            m=json.loads(mechanism_path.read_text());assert m['complete'] and m['checkpoint_sha256']==snapshot([checkpoint])[str(checkpoint.resolve())]
            assert m['source_seal_sha256']==snapshot([root/'evidence/pre_optimization_freeze.json'])[str((root/'evidence/pre_optimization_freeze.json').resolve())]
            assert len(m['records'])==6
            mechanism=dict(passed=sum(r['passed'] for r in m['records']),total=6,records=[dict(name=r['name'],passed=r['passed'],new_points=r['result'].get('new_active_points'),error=r['result'].get('error')) for r in m['records']])
            bindings += list((root/'results/mechanism').rglob('*.json'))
        reports[arm]=dict(actual_updates=proof['actual_updates'],actual_epochs=proof['actual_epochs'],tests_passed=tested['tests_passed'],tests_skipped=tested['tests_skipped'],
            checkpoint_sha256=snapshot([checkpoint])[str(checkpoint.resolve())],changed_tensors=changed,frozen_tensor_digests=len(frozen['before']),
            approach_before=before,approach_after=after,approach_gate_passed=approach_ok,mechanism=mechanism,
            full_nominal_gate_passed=bool(approach_ok and mechanism and mechanism['passed']==6),measured_updates=measured,
            first_arm_lr=trace[0]['arm_lr'],last_arm_lr=trace[-1]['arm_lr'],batch_order_sha256=config['batch_order_sha256'],sample_ids_sha256=config['sample_ids_sha256'])
        bindings += [integrity,checkpoint,root/'evidence/pre_optimization_freeze.json',root/'evidence/tested_sources.json']
        bindings += list((run/'artifacts').glob('*.json'))+[run/'artifacts/optimizer_trace.jsonl']
    assert reports['fixed']['batch_order_sha256']==reports['decay']['batch_order_sha256'] and reports['fixed']['sample_ids_sha256']==reports['decay']['sample_ids_sha256']
    candidates=[name for name,r in reports.items() if r['full_nominal_gate_passed']]
    selected=min(candidates,key=lambda name:(reports[name]['approach_after']['rms_arm_prefix_change_rad'],name!='fixed')) if candidates else None
    result=dict(complete=True,verification_complete_not_task_success=True,actual_updates_total=sum(r['actual_updates'] for r in reports.values()),
        reports=reports,nominal_candidate=selected,selection_rule='lowest_replay_RMS_then_fixed_after_6of6',formal_training_started=False,independent16_evaluated=False,
        hardware_commands_sent=False,source_and_output_sha256=snapshot(bindings+[Path(__file__)]))
    (OUT/'pilot_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['reports','source_and_output_sha256']},indent=2))
    for arm,r in reports.items():print(json.dumps(dict(arm=arm,replay=r['approach_after'],approach_gate=r['approach_gate_passed'],mechanism=r['mechanism'])))


if __name__=='__main__':main()
