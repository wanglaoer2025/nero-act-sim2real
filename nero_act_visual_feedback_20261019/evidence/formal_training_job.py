"""Sequential OFFLINE formal ACT candidates. No hardware imports or permissions."""
import json, os, time
from pathlib import Path
from training.visual import train_visual
root=Path(__file__).resolve().parents[1]
status_path=root/'evidence/formal_job_status.json'
outputs=[]
try:
    for width in [320,160]:
        output=root/f'training/visual_retained_v1_{width}'
        status_path.write_text(json.dumps(dict(stage='running_formal_candidate',pid=os.getpid(),active_resolution_width=width,active_output=str(output),completed_candidates=outputs,hardware_deployment_allowed=False,updated_unix_s=time.time()),indent=2))
        settings=json.loads((root/f'config/visual_training_{width}.json').read_text())
        result=train_visual(root/'training_data/visual_retained_v1',root.parent/'nero_act_current_cup_trial_20261019/model/current_cup_run27/policy.ckpt',settings,output)
        outputs.append(dict(width=width,output=str(output),result=result))
    status_path.write_text(json.dumps(dict(stage='all_declared_training_complete_evaluation_pending',pid=os.getpid(),completed_candidates=outputs,hardware_deployment_allowed=False,updated_unix_s=time.time()),indent=2))
except Exception as exc:
    status_path.write_text(json.dumps(dict(stage='failed',error=str(exc),pid=os.getpid(),completed_candidates=outputs,hardware_deployment_allowed=False,updated_unix_s=time.time()),indent=2))
    raise
