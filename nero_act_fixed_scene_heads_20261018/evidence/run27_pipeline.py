"""Offline bounded sequence: complete data, discarded probe, train, freeze, evaluate14."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'evidence/run27_pipeline_status.json'


def main():
    state=dict(stage='waiting_for_declared_dataset',hardware_connected=False,
        hardware_motion_allowed=False,steps=[],started_wall_s=time.time())
    with STATUS.open('x') as stream:json.dump(state,stream,indent=2)
    def save():
        state['updated_wall_s']=time.time()
        temporary=STATUS.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(STATUS)
    def run(stage,arguments):
        state['stage']=stage;save()
        log=ROOT/'evidence'/f'run27_{stage}.log'
        with log.open('x') as stream:
            process=subprocess.Popen([sys.executable,'-u',*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid;save();code=process.wait()
        state['steps'].append(dict(stage=stage,exit_code=code,log=str(log)));save()
        if code:raise RuntimeError(f'{stage} failed: {code}; no automatic retry')
    try:
        deadline=time.monotonic()+2700
        while True:
            data=json.loads((ROOT/'training_data/fixed_v2/manifest.json').read_text())
            state['collected_episodes']=len(data['episodes']);save()
            if data['rejected']:raise RuntimeError('Declared dataset rejected a case; no training')
            if data['complete']:break
            if time.monotonic()>deadline:raise TimeoutError('Dataset completion deadline expired')
            time.sleep(10)
        run('learning_probe',['evidence/probe_fixed_heads.py'])
        run('training',['evidence/train_fixed_heads.py'])
        run('export',['-m','training.export','--checkpoint',str(ROOT/'training/run27_fixed_heads/checkpoints/policy_best_val.ckpt'),
            '--output',str(ROOT/'model/run27_fixed_heads')])
        run('current_scene14',['-m','simulation.evaluate','--checkpoint',str(ROOT/'model/run27_fixed_heads/policy.ckpt'),
            '--output',str(ROOT/'evaluation/run27_current_scene14'),'--seed-offset','270018','--supervised'])
        evaluation=json.loads((ROOT/'evaluation/run27_current_scene14/summary.json').read_text())
        assert evaluation['complete'] and evaluation['finalized'] and evaluation['total']==14
        state.update(stage='offline_validation_finished',passed=evaluation['passed'],total=14,
            execution_ready=False,fresh_real_shadow_and_onsite_authorization_still_required=True)
        save()
    except BaseException as exc:
        state.update(stage='stopped_no_hardware',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
