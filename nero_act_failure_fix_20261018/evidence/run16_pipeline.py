"""One bounded offline training+frozen-evaluation job; no hardware entrypoints."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN='run16_visual_learning'
STATUS=ROOT/'evidence/run16_pipeline_status.json'
started=time.monotonic()
state=dict(run=RUN,stage='starting',hardware_connected=False,hardware_deployment_allowed=False,steps=[])
def save():
    state['elapsed_s']=time.monotonic()-started
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATUS)
def run(stage,args):
    log=ROOT/'evidence'/f'run16_{stage}.log';state['stage']=stage
    with log.open('x') as f:
        process=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        state['child_pid']=process.pid;save();code=process.wait()
    state['steps'].append(dict(stage=stage,exit_code=code,log=str(log)));save()
    if code:raise RuntimeError(f'{stage} process failed ({code}); preserve all artifacts')
try:
    definition=ROOT/'evidence/predeclared_independent_16_v2.json'
    state['independent_definition_sha256']=hashlib.sha256(definition.read_bytes()).hexdigest()
    run('training',['-u','-m','training.generalized','--data-dir',str(ROOT.parent/'nero_act_generalized_pick_place_20261018/training_data/generalized_v1'),
        '--corrections',str(ROOT.parent/'nero_act_contact_recovery_20261018/training_data/recoveries_v1/manifest.json'),
        '--initializer',str(ROOT/'model/run14/policy.ckpt'),'--independent-cases',str(definition),
        '--run',RUN,'--epochs','20','--train-visual-backbone'])
    run('export',['-u','-m','training.export','--checkpoint',str(ROOT/'training'/RUN/'checkpoints/policy_best_l1.ckpt'),
        '--output',str(ROOT/'model'/RUN)])
    run('development16',['-u','-m','simulation.evaluate_generalized','--model',str(ROOT/'model'/RUN),
        '--output',str(ROOT/'evaluation/run16_development16'),'--development-regression'])
    run('independent16',['-u','-m','simulation.evaluate_generalized','--model',str(ROOT/'model'/RUN),
        '--output',str(ROOT/'evaluation/run16_independent16'),'--fresh-independent'])
    state['stage']='training_and_evaluation_finished'
    for name in ['development16','independent16']:
        summary=json.loads((ROOT/f'evaluation/run16_{name}/summary.json').read_text())
        state[name]=dict(passed=summary['passed'],total=summary['total'])
except Exception as exc:
    state.update(stage='failed',error=str(exc));save();raise
finally:save()
