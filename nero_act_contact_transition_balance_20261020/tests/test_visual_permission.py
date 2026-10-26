import importlib.util
import json
import subprocess
import sys


def test_offline_entry_cannot_start_robot_or_use_old_receipt(tmp_path):
    assert importlib.util.find_spec('executor.visual_run') is not None, 'Offline15D readiness entry missing'
    command=[sys.executable,'-m','executor.visual_run','--model',str(tmp_path/'missing_model'),
             '--profile',str(tmp_path/'missing_profile'),'--output',str(tmp_path/'report')]
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    report=json.loads((tmp_path/'report/report.json').read_text())
    assert not report['motion_sent'] and not report['hardware_connected'] and not report['execution_ready']
    assert report['action_clients_created']==0
    denied=subprocess.run(command+['--execute','--receipt',str(tmp_path/'old_ACT_REPLAY.json')],capture_output=True,text=True)
    assert denied.returncode != 0 and 'offline' in denied.stderr.lower()
