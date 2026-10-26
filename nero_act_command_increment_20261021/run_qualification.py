"""Unchanged original failure inputs, six snapshots and conditional high-start full task."""
from pathlib import Path
import json,copy,importlib.util
import torch,numpy as np,h5py
from pipeline import ROOT,PARENT,load_policy,validated_replay,snapshot,require_unchanged
from training.history_dataset import HistoryReferenceDataset
from training.approach_preservation import ApproachPreservation
from training.history_recoveries import validate_history_manifest,sha
from executor.model import physical_actions
from simulation.history_mechanism import mechanism_passed
from simulation.supervised_rollout import rollout
from simulation.visual_evaluation import full_task_passed

legacy_spec=importlib.util.spec_from_file_location('frozen_qualification',PARENT/'evidence/qualify_pilot.py')
legacy=importlib.util.module_from_spec(legacy_spec);legacy_spec.loader.exec_module(legacy)


def verify_outcomes(summary,gate):
    exact=legacy.qualified(gate['failed_inputs'],gate['replay_assessment'])
    if (gate['replay_assessment']['samples']!=423 or gate['exact_and_preservation_gate_passed']!=exact
            or summary['exact_and_preservation_gate_passed']!=exact):
        raise ValueError('Input gate summary differs from original numeric thresholds')
    expected={'nominal_pre_close','nominal_hold_to_lift','heldout_pre_close','heldout_hold_to_lift','heldout_support_release','heldout_release_retreat'}
    records=summary['records']
    if len(records)!=6 or {r['name'] for r in records}!=expected:raise ValueError('Original six cases required')
    passed=[mechanism_passed(r['result'],r['role']) for r in records]
    if any(r['passed']!=actual for r,actual in zip(records,passed)):raise ValueError('Case flags differ from raw results')
    all_six=all(passed);nominal=summary['nominal'];evaluated=nominal is not None
    if (summary['original_six_pass_count']!=sum(passed) or summary['original_six_passed']!=all_six
            or summary['nominal_evaluated']!=evaluated or evaluated!=bool(exact and all_six)):
        raise ValueError('Full task was skipped or run outside original gate')
    full=bool(nominal and full_task_passed(nominal['result']))
    if summary['full_task_passed']!=full or (nominal and nominal['full_task_passed']!=full):
        raise ValueError('Full-task success must be computed from an actual complete result')


def main():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    output=ROOT/'results/qualification';output.mkdir(exist_ok=False)
    captures=ROOT.parent/'nero_act_optimizer_decay_20261021/training_data/recoveries_v5'
    old_data=ROOT.parent/'nero_act_optimizer_decay_20261021/training_data/filtered_v1'
    manifest=json.loads((old_data/'manifest.json').read_text());paths=[Path(next(r['path'] for r in manifest['episodes'] if r['id']==i)) for i in [2017,2018]]
    policy,stats=load_policy();replay_hashes=validated_replay(policy.config['approach_preservation']['replay_path'])
    keeper=ApproachPreservation(policy.config['approach_preservation'],stats,'cuda');replay=keeper.assess(policy)
    if replay['samples']!=423:raise ValueError('Original423 reference required')
    dataset=HistoryReferenceDataset(paths,stats,policy.config,weighted=False);records=[]
    for eid,t,holding in [(2017,5,True),(2018,5,False),(2018,10,True)]:
        index=next(i for i,(n,s,v) in enumerate(dataset.sample_ids) if n==f'episode_{eid}' and s==t);sample=dataset[index];fi,_=dataset.samples[index]
        with h5py.File(paths[fi],'r') as f:width=float(f['observations/gripper_state'][t,0])
        previous=width+float(sample['current'][8])*.01
        with torch.no_grad():pred=physical_actions(policy(sample['current'][None].cuda(),sample['image'][None].cuda()),stats,bounded_gripper=True)[0].cpu().numpy()
        target=sample['actions'].numpy()*stats['action_std']+stats['action_mean']
        records.append(dict(sample=f'episode_{eid}:{t}',prefix_error_m=float(abs(pred[:2,7]-target[:2,7]).max()),next_minus_previous_m=float(pred[1,7]-previous),minimum_prefix_margin_m=float(width-pred[:2,7].max()),require_holding=holding,prediction_prefix_m=pred[:2,7].tolist(),teacher_prefix_m=target[:2,7].tolist()))
    exact=legacy.qualified(records,replay)
    (output/'input_gate.json').write_text(json.dumps(dict(complete=True,failed_inputs=records,replay_assessment=replay,exact_and_preservation_gate_passed=exact),indent=2)+'\n')
    print(json.dumps(dict(stage='input_gate',passed=exact,records=records)),flush=True)
    recovery=json.loads((captures/'manifest.json').read_text());base={r['id']:r for r in recovery['base_episodes']}
    split=json.loads((ROOT.parent/'nero_act_generalized_pick_place_20261018/training_data/generalized_v1/split.json').read_text())
    validate_history_manifest(recovery,split,additional_failure_ids=[2018,2019])
    nominal=PARENT/'results/original_failure_capture';scene=json.loads((nominal/'scene_configuration.json').read_text());scenario=json.loads((nominal/'result.json').read_text())['scenario'];contexts=json.loads((nominal/'runtime_checkpoints.json').read_text())['named']
    cases=[(f'nominal_{role}',scene,scenario,contexts[role],'hold_to_lift',None) for role in ['pre_close','hold_to_lift']]
    inputs=[Path(__file__),PARENT/'evidence/qualify_pilot.py',ROOT/'training/pilot_A/policy_last.ckpt',ROOT/'training/pilot_A/integrity.json',captures/'manifest.json',old_data/'manifest.json']+paths
    inputs += [nominal/name for name in ['scene_configuration.json','result.json','runtime_checkpoints.json']]
    for row in [r for r in recovery['episodes'] if r['split']=='val']:
        parent=base[row['parent_episode_id']]
        if sha(parent['path'])!=parent['sha256']:raise ValueError('Heldout source changed')
        with h5py.File(parent['path'],'r') as f:heldout_scene=json.loads(f.attrs['scene_config'])
        heldout_scenario=dict(seed=heldout_scene['generalization']['seed'],query_stride=2,servo_delay_steps=heldout_scene.get('action_delay_steps',0),observation_age_s=.05,jitter=False,force=False)
        cases.append((f'heldout_{row["role"]}',heldout_scene,heldout_scenario,json.loads(Path(row['context_path']).read_text()),row['role'],parent['sha256']));inputs += [Path(row['context_path']),Path(parent['path'])]
    hashes=dict(snapshot(inputs),**replay_hashes);six=[]
    for name,scene,scenario,context,role,parent_sha in cases:
        result=rollout(copy.deepcopy(scene),policy,stats,copy.deepcopy(scenario),ROOT/'results/mechanism_original'/name,resume_context=context,runtime_profile='history_common_v1',resume_policy_transition=True,expected_parent_source_sha256=parent_sha,mechanism_role='hold_to_lift' if role=='hold_to_lift' else None)
        record=dict(name=name,role=role,passed=mechanism_passed(result,role),result=result);six.append(record)
        print(json.dumps(dict(name=name,passed=record['passed'],points=result.get('new_active_points'),error=result.get('error'))),flush=True)
        (output/'progress.json').write_text(json.dumps(dict(completed=len(six),total=6),indent=2)+'\n')
    all_six=len(six)==6 and all(r['passed'] for r in six);full=None
    if exact and all_six:
        scene_path=ROOT.parent/'nero_act_approach_preservation_20261021/training_data/approach_complete_v1/scene_config.json';scene=json.loads(scene_path.read_text());hashes.update(snapshot([scene_path]))
        case=dict(name='nominal_development',seed=191899,query_stride=2,servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False,scene=copy.deepcopy(scene))
        result=rollout(copy.deepcopy(scene),policy,stats,case,ROOT/'results/nominal_after/nominal_development',runtime_profile='history_common_v1');full=dict(result=result,full_task_passed=full_task_passed(result))
    require_unchanged(hashes)
    summary=dict(complete=True,actual_updates=1000,formal_epochs=0,exact_and_preservation_gate_passed=exact,original_six_pass_count=sum(r['passed'] for r in six),original_six_passed=all_six,records=six,nominal_evaluated=full is not None,nominal=full,full_task_passed=bool(full and full['full_task_passed']),independent_generalization_evaluated=False,hardware_commands_sent=False,source_sha256=hashes)
    verify_outcomes(summary,json.loads((output/'input_gate.json').read_text()))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['records','source_sha256','nominal']}),flush=True)

if __name__=='__main__':main()
