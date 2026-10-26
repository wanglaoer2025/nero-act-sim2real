"""Run one scoped attempt and restore a receive-only driver in finally; never retry."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
from executor.supervision import start_detached,install_cleanup_signals,ignore_cleanup_signals
from executor.bundle import verify_model_bundle
from executor.permission import require_experimental_permission
from executor.evaluation_integrity import verify_frozen_artifacts
model=ROOT/'model/frozen_run11'; evaluation=ROOT/'evidence/accepted_offset_frozen_nominal/summary.json'
receipt=ROOT/'evidence/act_replay_authorization.json'
geometry=ROOT/'evidence/authorized_scene/geometry.json'
report_path=ROOT/'evidence/authorized_session.json'
if report_path.exists():raise FileExistsError(report_path)
metadata=verify_model_bundle(model);ev=json.loads(evaluation.read_text());auth=json.loads(receipt.read_text())
require_experimental_permission(metadata,ev,auth,now=time.time())
verify_frozen_artifacts(model/'policy.ckpt',ev)
for name,h in json.loads((ROOT/'package_sha256.json').read_text()).items():
 if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Source mismatch: '+name)
if not 0<=time.time()-json.loads(geometry.read_text())['capture_wall_s']<=75:raise RuntimeError('Recapture scene before switching drivers')
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
record=dict(started_wall_s=time.time(),can_tx_before=tx(),scope='one_ACT_simulation_trajectory_replay_with_real_cup',no_automatic_retry=True)
handles=[];driver=gate=runtime=restored=None
old_record=json.loads((ROOT/'evidence/readonly_after_mock_sync.json').read_text())
old_pid=old_record['pid']
old_cmd=Path(f'/proc/{old_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
if '-m executor.driver --readonly' not in old_cmd:raise RuntimeError('Known receive-only driver identity changed')
if Path(f'/proc/{old_pid}/cwd').resolve()!=ROOT:raise RuntimeError('Known driver working directory differs')
def stop(p):
 if p is None or p.poll() is not None:return
 p.send_signal(signal.SIGINT)
 try:p.wait(timeout=8)
 except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=5)
def start(cmd,name):
 log=(ROOT/'evidence'/name).open('x');handles.append(log)
 return start_detached(cmd,cwd=ROOT,log=log)
base=['--model',str(model),'--evaluation',str(evaluation),'--receipt',str(receipt)]
install_cleanup_signals()
try:
 os.kill(old_pid,signal.SIGINT)
 deadline=time.monotonic()+8
 while Path(f'/proc/{old_pid}/cmdline').exists():
  try:
   if not Path(f'/proc/{old_pid}/cmdline').read_bytes():break
  except FileNotFoundError:break
  if time.monotonic()>deadline:raise RuntimeError('Known receive-only driver did not exit')
  time.sleep(.1)
 record['can_tx_after_readonly_stop']=tx()
 if tx()!=record['can_tx_before']:raise RuntimeError('Unexpected CAN TX before scoped driver startup')
 driver=start(['/usr/bin/python3','-u','-m','executor.driver','--execute',*base,'--output',str(ROOT/'evidence/execute_driver_start.json')],'execute_driver.log')
 deadline=time.monotonic()+15
 while not (ROOT/'evidence/execute_driver_start.json').exists():
  if driver.poll() is not None:raise RuntimeError('Scoped driver startup failed')
  if time.monotonic()>deadline:raise TimeoutError('Scoped driver startup record missing')
  time.sleep(.1)
 gate=start(['/usr/bin/python3','-u','-m','executor.gate','--execute',*base,'--ros-args','-p','arm_joint_names:=[joint1,joint2,joint3,joint4,joint5,joint6,joint7]','-p','status_topics:=[/arm_controller/follow_joint_trajectory/_action/status,/gripper_controller/follow_joint_trajectory/_action/status]','-p','arm_sync_tolerance_rad:=0.001','-p','gripper_sync_tolerance_m:=0.0005','-p','state_timeout_s:=0.15'],'execute_gate.log')
 time.sleep(1)
 if gate.poll() is not None:raise RuntimeError('Scoped gate startup failed')
 command=[str(ROOT.parent/'nero_act_full_pick_place_20261017/.venv/bin/python'),'-u',str(ROOT/'run.py'),'--execute',*base,'--geometry',str(geometry),'--output',str(ROOT/'evidence/real_attempt_01')]
 print('STARTING ONE ACT SIMULATION TRAJECTORY REAL CUP REPLAY',flush=True)
 runtime=subprocess.Popen(command,cwd=ROOT)
 while runtime.poll() is None:
  if driver.poll() is not None or gate.poll() is not None:raise RuntimeError('Driver or protection gate exited during attempt')
  time.sleep(.1)
 record['runtime_returncode']=runtime.returncode
except BaseException as exc:
 record['session_error']=f'{type(exc).__name__}: {exc}'
 print(record['session_error'],flush=True)
finally:
 ignore_cleanup_signals()
 stop(runtime);stop(gate);stop(driver)
 record['can_tx_before_restore']=tx()
 try:
  restored=start(['/usr/bin/python3','-u','-m','executor.driver','--readonly','--output',str(ROOT/'evidence/readonly_restored.json')],'readonly_restored.log')
  deadline=time.monotonic()+15
  while not (ROOT/'evidence/readonly_restored.json').exists():
   if restored.poll() is not None:raise RuntimeError('Read-only restoration process failed')
   if time.monotonic()>deadline:raise TimeoutError('Read-only restoration record missing')
   time.sleep(.1)
  time.sleep(.5)
  if restored.poll() is not None or os.getsid(restored.pid)!=restored.pid:
   raise RuntimeError('Restored receiver is dead or still tied to the task session')
  restored_profile=json.loads((ROOT/'evidence/readonly_restored.json').read_text())
  if (restored_profile['pid']!=restored.pid or restored_profile['runtime_send_blocked'] is not True
      or restored_profile['auto_enable'] is not False or restored_profile['control_enabled_at_start'] is not False
      or restored_profile['speed_percent_parameter']!=0):
   raise RuntimeError('Restored receiver profile is not blocked receive-only')
  record['readonly_restored']=True;record['readonly_pid']=restored.pid
  record['post_session_feedback_verification_pending']=True
 except BaseException as exc:record['restoration_error']=f'{type(exc).__name__}: {exc}'
 record.update(ended_wall_s=time.time(),can_tx_after=tx())
 report_path.write_text(json.dumps(record,indent=2))
 for h in handles:h.close()
 print(json.dumps(record),flush=True)
