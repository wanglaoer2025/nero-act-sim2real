"""Move only mock joints while the real driver is absent, then restore blocked receive-only."""
from pathlib import Path
import json,os,signal,subprocess,time,sys
from executor.supervision import start_detached
r=Path(__file__).resolve().parents[1];os.chdir(r)
old=r.parent/'nero_act_transfer_completion_20261017/evidence/readonly_restored.json'
source=json.loads(old.read_text());pid=source['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert '-m executor.driver --readonly' in cmd
assert Path(f'/proc/{pid}/cwd').resolve()==r.parent/'nero_act_transfer_completion_20261017'
tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text());before=tx()
record=dict(mode='mock_only_start_alignment',physical_motion=False,can_tx_before=before,readonly_pid_stopped=pid)
try:
 os.kill(pid,signal.SIGINT);end=time.monotonic()+8
 while Path(f'/proc/{pid}/cmdline').exists():
  try:
   if not Path(f'/proc/{pid}/cmdline').read_bytes():break
  except FileNotFoundError:break
  if time.monotonic()>end:raise TimeoutError('Read-only process still present')
  time.sleep(.1)
 assert tx()==before
 result=subprocess.run([str(r.parent/'nero_act_full_pick_place_20261017/.venv/bin/python'),'-u','-m','executor.virtual_bridge','--synchronize-virtual','--output',str(r/'evidence/mock_bridge_synchronized.json')],cwd=r)
 record['mock_returncode']=result.returncode
finally:
 log=(r/'evidence/readonly_after_mock_sync.log').open('x')
 receiver=start_detached(['/usr/bin/python3','-u','-m','executor.driver','--readonly','--output',str(r/'evidence/readonly_after_mock_sync.json')],cwd=r,log=log)
 end=time.monotonic()+15
 while not (r/'evidence/readonly_after_mock_sync.json').exists():
  if receiver.poll() is not None or time.monotonic()>end:raise RuntimeError('Failed to restore readonly receiver')
  time.sleep(.1)
 record.update(restored_pid=receiver.pid,can_tx_after=tx(),restore_detached=True)
 (r/'evidence/mock_preparation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record));log.close()
if record.get('mock_returncode')!=0 or tx()!=before:raise SystemExit(1)
