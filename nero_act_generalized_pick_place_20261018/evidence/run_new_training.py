"""One offline data->smoke->20epoch->export->evaluation pipeline; no retries or hardware."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
PYTHON=str(ROOT/'.venv/bin/python');data=ROOT/'training_data/generalized_v1'
status=ROOT/'evidence/training_pipeline_status.json'
record=dict(mode='offline_simulation_training',hardware_access=False,stage='waiting_for_48_verified_demonstrations',started_wall_s=time.time(),pid=os.getpid(),steps=[])
def save():
 temp=status.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(status)
def run(stage,command,logname):
 record['stage']=stage;save();print(stage,flush=True)
 with (ROOT/'evidence'/logname).open('x') as log:
  child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  record['child_pid']=child.pid;save();code=child.wait()
 record['steps'].append(dict(stage=stage,returncode=code,log=logname));save()
 if code:raise RuntimeError(f'{stage} failed with exit{code}; see{logname}')
save()
try:
 deadline=time.monotonic()+3600
 while True:
  m=json.loads((data/'manifest.json').read_text())
  if m.get('complete') and len(m['episodes'])==48:break
  if time.monotonic()>deadline:raise TimeoutError('Complete physical dataset not ready; optimization not started')
  time.sleep(3)
 record.update(complete_demonstrations=48,train_scenes=36,val_scenes=12,
  manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
 save()
 run('cuda_smoke_1epoch',[PYTHON,'-u','-m','training.generalized','--data-dir',str(data),'--run','smoke_context_generalization','--epochs','1'],'training_smoke.log')
 run('formal_training_20epochs',[PYTHON,'-u','-m','training.generalized','--data-dir',str(data),'--run','run14_position_generalization','--epochs','20'],'run14_training.log')
 model=ROOT/'model/run14_position_generalization'
 run('export_frozen_validation_selected_model',[PYTHON,'-u','-m','training.export','--checkpoint',str(ROOT/'training/run14_position_generalization/checkpoints/policy_best_val.ckpt'),'--output',str(model)],'run14_export.log')
 run('frozen16case_online_closed_loop_evaluation',[PYTHON,'-u','-m','simulation.evaluate_generalized','--model',str(model),'--output',str(ROOT/'evaluation/run14_generalization16')],'run14_evaluation.log')
 summary=json.loads((ROOT/'evaluation/run14_generalization16/summary.json').read_text())
 record.update(stage='training_and_evaluation_finished',simulation_cases_passed=summary['passed'],simulation_cases_total=summary['total'],hardware_authorized=False,hardware_task_verified=False,ended_wall_s=time.time());save()
except BaseException as exc:
 record.update(stage='stopped_with_error',error=f'{type(exc).__name__}: {exc}',ended_wall_s=time.time());save();raise
