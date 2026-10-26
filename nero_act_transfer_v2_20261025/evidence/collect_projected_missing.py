import os
os.environ['MUJOCO_GL']='egl'
import sys,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed,CancelledError
import h5py
ROOT=Path('/home/user/wq/nero_act_transfer_v2_20261025')
sys.path.insert(0,str(ROOT))
from collect_sim import collect_one,planned_jobs
from rebuild_data import write,sha

if __name__=='__main__':
 output=ROOT/'data/v2_projected_missing_01'
 first=json.loads((output/'first_full_route.json').read_text())
 assert first['accepted'] and first['result']['full_task_verified']
 assert sha(first['path'])==first['sha256']
 original=json.loads((ROOT/'data/v2_enclosed_01/manifest.json').read_text())
 with h5py.File(original['source_scene']) as f:base=json.loads(f.attrs['scene_config'])
 protocol=json.loads((ROOT/'position_protocol.json').read_text())
 oldnames={Path(r['path']).name for r in original['episodes']}
 jobs=[j for j in planned_jobs(base,output,protocol) if Path(j['output']).name not in oldnames]
 assert len(jobs)==30
 todo=[j for j in jobs if j['output']!=first['path']]
 assert len(todo)==29 and not any(Path(j['output']).exists() for j in todo)
 manifest=dict(schema='nero_act_transfer_v2_missing_30',episodes=[first],rejected=[],expected_total=30,
  complete=False,collection_running=True,observer_sha256=sha(ROOT/'observer.py'),hardware_commands_sent=False)
 assert not (output/'manifest.json').exists()
 write(output/'manifest.json',manifest)
 def state():
  write(ROOT/'evidence/pipeline_state.json',dict(stage='collecting_missing_full_trajectories',
   accepted=120+len(manifest['episodes']),expected_total=150,new_accepted=len(manifest['episodes']),
   new_expected_total=30,rejected=len(manifest['rejected']),pilot_updates=200,pilot_limit=200,
   formal_updates=0,full_task_passed=False,hardware_commands_sent=False))
 state()
 with ProcessPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(collect_one,j) for j in todo]
  for future in as_completed(futures):
   try:row=future.result()
   except CancelledError:continue
   except Exception as exc:row=dict(accepted=False,error=type(exc).__name__+': '+str(exc))
   manifest['episodes' if row['accepted'] else 'rejected'].append(row)
   if not row['accepted']:
    manifest['stopped_on_failure']=True
    for pending in futures:pending.cancel()
   write(output/'manifest.json',manifest);state()
   print(json.dumps(dict(accepted=len(manifest['episodes']),rejected=len(manifest['rejected']),
    last=row.get('position_group'),error=row.get('error'))),flush=True)
 manifest['collection_running']=False
 manifest['complete']=len(manifest['episodes'])==30 and not manifest['rejected']
 write(output/'manifest.json',manifest);state()
 raise SystemExit(0 if manifest['complete'] else 2)
