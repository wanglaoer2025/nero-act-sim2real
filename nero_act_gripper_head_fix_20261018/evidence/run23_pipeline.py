"""One bounded offline corrective training run and frozen ACT task evaluation."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
from training.generalized import combined_inputs
from simulation.refresh_observations import require_fresh_inputs

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'nero_act_transition_fix_20261018'
RUN='run23_gripper_head_only'
STATUS=ROOT/'evidence/run23_pipeline_status.json'
started=time.monotonic()
state=dict(run=RUN,stage='preflight',hardware_connected=False,hardware_deployment_allowed=False,steps=[])
def save():
    state['elapsed_s']=time.monotonic()-started
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATUS)
def run(stage,args):
    log=ROOT/'evidence'/f'run23_{stage}.log';state['stage']=stage
    with log.open('x') as stream:
        process=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        state['child_pid']=process.pid;save();code=process.wait()
    state['steps'].append(dict(stage=stage,exit_code=code,log=str(log)));save()
    if code:raise RuntimeError(f'{stage} failed ({code}); artifacts preserved')
# Claim the run before handlers can write; a second launch preserves prior evidence.
with STATUS.open('x') as stream:
    json.dump(state,stream,indent=2)
try:
    probe=json.loads((ROOT.parent/'nero_act_gripper_gradient_fix_20261018/evidence/actual_gradient_fix.json').read_text())
    if not probe['passed'] or probe['bounded_prediction_difference']>1e-6:raise RuntimeError('CVAE mechanism probe must pass first')
    tests=(ROOT/'evidence/full_suite.log').read_text()
    if '\nOK\n' not in tests or 'FAILED (' in tests:raise RuntimeError('Full regression must pass first')
    base=OLD/'training_data/critical_rgb_v1/generalized_v1'
    prior=OLD/'training_data/critical_rgb_v1/recoveries_v1/manifest.json'
    correction=ROOT.parent/'nero_act_gripper_gradient_fix_20261018/training_data/direct_grasp_fresh_v1/train_manifest.json'
    correction_val=ROOT.parent/'nero_act_gripper_gradient_fix_20261018/training_data/direct_grasp_fresh_v1/val_manifest.json'
    prior_grasp=ROOT.parent/'nero_act_grasp_state_recovery_20261018/training_data/grasp_fresh_v2/manifest.json'
    prior_descend=ROOT.parent/'nero_act_alignment_recovery_20261018/training_data/alignment_fresh_v1/manifest.json'
    batches=[json.loads(p.read_text()) for p in [correction,correction_val,prior_grasp,prior_descend,prior]]
    manifest,split,paths=combined_inputs(json.loads((base/'manifest.json').read_text()),
        json.loads((base/'split.json').read_text()),batches)
    assert len(paths)==72 and len(split['train'])==51 and len(split['val'])==21
    require_fresh_inputs(paths)
    audit=dict(episodes=len(paths),train=split['train'],val=split['val'],
        complete_demonstrations=48,prior_lift_recoveries=6,prior_descent_recoveries=12,prior_grasp_recoveries=4,new_direct_grasp_recoveries=2,
        complete_source_groups=len({r['source_group'] for r in manifest['episodes'][:48]}),
        paired_batches_validated=True,hardware_connected=False)
    (ROOT/'evidence/run23_data_preflight.json').write_text(json.dumps(audit,indent=2))
    definition=ROOT.parent/'nero_act_alignment_recovery_20261018/evidence/predeclared_independent_16_v4.json'
    state['independent_definition_sha256']=hashlib.sha256(definition.read_bytes()).hexdigest();save()
    run('training',['-u','-m','training.generalized','--data-dir',str(base),
        '--corrections',str(correction),'--additional-corrections',str(correction_val),
        '--additional-corrections',str(prior_grasp),'--additional-corrections',str(prior_descend),'--additional-corrections',str(prior),
        '--initializer',str(ROOT.parent/'nero_act_cvae_training_fix_20261018/model/run21_cvae_posterior/policy.ckpt'),
        '--independent-cases',str(definition),'--run',RUN,'--epochs','3',
        '--train-gripper-head-only','--dense-contact-inputs','--pose-objective','--cvae-kl-weight','.001','--prior-gripper-weight','.05'])
    run('frozen_arm_audit',['-u','-m','evidence.audit_frozen_arm'])
    run('export',['-u','-m','training.export','--checkpoint',str(ROOT/'training'/RUN/'checkpoints/policy_best_val.ckpt'),
        '--output',str(ROOT/'model'/RUN)])
    run('nominal',['-u','-m','simulation.evaluate_generalized','--model',str(ROOT/'model'/RUN),
        '--output',str(ROOT/'evaluation/run23_nominal'),'--nominal-only','--development-regression'])
    nominal=json.loads((ROOT/'evaluation/run23_nominal/summary.json').read_text())
    state['nominal']=dict(passed=nominal['passed'],total=nominal['total']);save()
    run('validation42',['-u','-m','evidence.compare_validation42'])
    validation=json.loads((ROOT/'evaluation/run23_validation42/summary.json').read_text())
    state['validation42']=dict(passed=validation['passed'],total=validation['total']);save()
    if nominal['passed']!=1:
        state['stage']='nominal_failed_no_deployment'
    else:
        run('development16',['-u','-m','simulation.evaluate_generalized','--model',str(ROOT/'model'/RUN),
            '--output',str(ROOT/'evaluation/run23_development16'),'--development-regression'])
        run('independent16',['-u','-m','simulation.evaluate_generalized','--model',str(ROOT/'model'/RUN),
            '--output',str(ROOT/'evaluation/run23_independent16'),'--fresh-independent'])
        for name in ['development16','independent16']:
            summary=json.loads((ROOT/f'evaluation/run23_{name}/summary.json').read_text())
            state[name]=dict(passed=summary['passed'],total=summary['total'])
        state['stage']='training_and_evaluation_finished'
except Exception as exc:
    state.update(stage='failed',error=str(exc));save();raise
finally:save()
