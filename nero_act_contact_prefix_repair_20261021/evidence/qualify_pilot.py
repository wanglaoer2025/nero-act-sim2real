"""Evaluate the fixed1000 pilot on original failed inputs and unchanged simulation cases."""
from pathlib import Path
import sys,json,copy
import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulation.history_mechanism import load_pilot,run_mechanism
from simulation.supervised_rollout import rollout
from simulation.visual_evaluation import full_task_passed
from training.history_dataset import HistoryReferenceDataset
from training.approach_preservation import ApproachPreservation
from training.integrity import snapshot,require_unchanged
from executor.model import physical_actions


def qualified(records,replay):
    return bool(len(records)==3 and all(r['prefix_error_m']<=.0001 and r['next_minus_previous_m']<=-.0001
        and (not r['require_holding'] or r['minimum_prefix_margin_m']>=.003) for r in records)
        and replay['rms_arm_prefix_change_rad']<=.0001 and replay['maximum_arm_prefix_change_rad']<=.0003)


def main():
    # A below-contact input must keep closing; it need not already satisfy holding.
    safe=[dict(prefix_error_m=0.,next_minus_previous_m=-.0004,require_holding=i!=1,
        minimum_prefix_margin_m=.0034 if i!=1 else .0025) for i in range(3)]
    fixed=dict(rms_arm_prefix_change_rad=0.,maximum_arm_prefix_change_rad=0.)
    assert qualified(safe,fixed)
    failed=copy.deepcopy(safe);failed[0]['next_minus_previous_m']=.000025
    assert not qualified(failed,fixed)
    train=ROOT/'training/pilot_A';output=ROOT/'results/qualification';output.mkdir(exist_ok=False)
    captures=ROOT.parent/'nero_act_optimizer_decay_20261021/training_data/recoveries_v5'
    old_data=ROOT.parent/'nero_act_optimizer_decay_20261021/training_data/filtered_v1'
    manifest=json.loads((old_data/'manifest.json').read_text());rows={r['id']:r for r in manifest['episodes']}
    paths=[Path(rows[i]['path']) for i in [2017,2018]]
    inputs=[Path(__file__),ROOT/'approved_scope.md',ROOT/'evidence/pre_optimization_freeze.json',
        train/'checkpoints/policy_last.ckpt',train/'checkpoints/dataset_stats.pkl',old_data/'manifest.json',captures/'manifest.json']+paths
    hashes=snapshot(inputs)
    policy,stats=load_pilot(train);dataset=HistoryReferenceDataset(paths,stats,policy.config,weighted=False)
    keeper=ApproachPreservation(policy.config['approach_preservation'],stats,'cuda');replay=keeper.assess(policy)
    records=[]
    for eid,t,holding in [(2017,5,True),(2018,5,False),(2018,10,True)]:
        index=next(i for i,(name,start,view) in enumerate(dataset.sample_ids) if name==f'episode_{eid}' and start==t)
        sample=dataset[index];fi,j=dataset.samples[index];q=sample['current'][None].cuda();image=sample['image'][None].cuda()
        with h5py.File(paths[fi],'r') as f:width=float(f['observations/gripper_state'][t,0])
        previous=width+float(sample['current'][8])*.01
        with torch.no_grad():prediction=physical_actions(policy(q,image),stats,bounded_gripper=True)[0].cpu().numpy()
        truth=sample['actions'].numpy()*stats['action_std']+stats['action_mean']
        records.append(dict(sample=f'episode_{eid}:{t}',prefix_error_m=float(abs(prediction[:2,7]-truth[:2,7]).max()),
            next_minus_previous_m=float(prediction[1,7]-previous),minimum_prefix_margin_m=float(width-prediction[:2,7].max()),
            require_holding=holding,prediction_prefix_m=prediction[:2,7].tolist(),teacher_prefix_m=truth[:2,7].tolist(),
            measured_width_m=width,previous_command_m=previous,original_failed_input=True))
    exact=qualified(records,replay)
    gate=dict(complete=True,failed_inputs=records,replay_assessment=replay,exact_and_preservation_gate_passed=exact,
        thresholds=dict(prefix_error_m=.0001,closing_progress_m=.0001,holding_margin_m=.003,
            arm_rms_rad=.0001,arm_max_rad=.0003),source_sha256=hashes,hardware_commands_sent=False)
    (output/'input_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps(dict(exact_and_preservation_gate_passed=exact,records=records,replay=replay)),flush=True)
    del policy,keeper;torch.cuda.empty_cache()
    # Always retain the original six as a regression diagnosis; never substitute new snapshots.
    six=run_mechanism(train,captures,ROOT/'results/mechanism_original')
    all_six=six['mechanism_and_recoveries_passed'];nominal=None
    if exact and all_six:
        scene_path=ROOT.parent/'nero_act_approach_preservation_20261021/training_data/approach_complete_v1/scene_config.json'
        scene=json.loads(scene_path.read_text());hashes.update(snapshot([scene_path]))
        policy,stats=load_pilot(train)
        case=dict(name='nominal_development',seed=191899,query_stride=2,servo_delay_steps=0,observation_age_s=.05,
            jitter=False,force=False,scene=copy.deepcopy(scene))
        result=rollout(copy.deepcopy(scene),policy,stats,case,ROOT/'results/nominal_after/nominal_development',runtime_profile='history_common_v1')
        nominal=dict(result=result,full_task_passed=full_task_passed(result))
    require_unchanged(hashes)
    summary=dict(complete=True,actual_optimizer_updates=1000,formal_training_epochs=0,
        exact_and_preservation_gate_passed=exact,original_six_passed=all_six,original_six_pass_count=sum(r['passed'] for r in six['records']),
        nominal_evaluated=nominal is not None,nominal=nominal,full_task_passed=bool(nominal and nominal['full_task_passed']),
        formal_qualification=False,independent16_evaluated=False,hardware_commands_sent=False,source_sha256=hashes)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
