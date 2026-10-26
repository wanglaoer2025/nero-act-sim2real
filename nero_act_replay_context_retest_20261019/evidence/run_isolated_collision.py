"""Own and stop an execution-disabled MoveIt validator on isolated ROS domain43."""
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[1]
previous = ROOT.parent/'nero_act_replay_current_cup_retest_20261019'
record = json.loads((previous/'evidence/replay_session_01/readonly_restored.json').read_text())
assert record['runtime_send_blocked'] is True and record['mode'] == 'receive_only'
proc = Path('/proc')/str(record['pid'])
expected = ['/usr/bin/python3', '-u', '-m', 'executor.driver', '--readonly', '--output',
            str(previous/'evidence/replay_session_01/readonly_restored.json')]
assert proc.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0') == expected
env = dict(os.environ)
for item in proc.joinpath('environ').read_bytes().split(b'\0'):
    key, _, value = item.partition(b'=')
    if key.decode() in ['AMENT_PREFIX_PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH']:
        env[key.decode()] = value.decode()
env.update(ROS_DOMAIN_ID='43', PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLCONFIGDIR='/tmp/nero_collision_mpl')
prior = json.loads((previous/'evidence/isolated_collision_process.json').read_text())
command = prior['command']
assert 'allow_trajectory_execution:=false' in command
tx = lambda: int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
report = dict(command=command, ROS_DOMAIN_ID=43, allow_trajectory_execution=False,
              hardware_connected=False, can_tx_before=tx(), owned_validator_stopped=False)
child = None
try:
    with (ROOT/'evidence/isolated_collision_move_group.log').open('x') as log:
        child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, stdin=subprocess.DEVNULL)
        report['pid'] = child.pid
        with (ROOT/'evidence/accepted_offset_collision.log').open('x') as output:
            result = subprocess.run([str(ROOT/'.venv/bin/python'), '-u', '-m', 'evidence.probe_accepted_collisions'],
                env=env, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=180)
        report['probe_returncode'] = result.returncode
finally:
    if child is not None:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=12)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
        report['owned_validator_stopped'] = child.poll() is not None
    report['can_tx_after'] = tx()
    (ROOT/'evidence/isolated_collision_process.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
if report.get('probe_returncode') != 0 or report['can_tx_before'] != report['can_tx_after']:
    raise SystemExit(1)
