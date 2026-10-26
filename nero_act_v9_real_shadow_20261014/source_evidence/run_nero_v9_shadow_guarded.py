from pathlib import Path
import sys,json,datetime,socket
from unittest.mock import patch
p=Path('/home/user/wq/nero_act_4090_shadow_recovery_v9(2)/nero_act_4090_shadow_recovery_v9');root=Path('/home/user/wq/nero_act_v9_real_shadow_20261014')
pre=json.loads((root/'preflight.json').read_text());sys.path.insert(0,str(p/'nero_act_source'))
from src.act.robot_wrapper import NeroRobotWrapper
from src.act import run_act_shadow
import torch
assert torch.cuda.is_available(),'4090 CUDA unavailable; do not silently report GPU test'
def denied(*args,**kwargs):raise AssertionError('Hardware operation forbidden during recorded shadow')
original_socket=socket.socket
def guarded_socket(family=socket.AF_INET,*args,**kwargs):
 if family==socket.AF_CAN:raise AssertionError('CAN socket forbidden during shadow')
 return original_socket(family,*args,**kwargs)
coords=[str(x) for x in pre['coordinate_input_mm']]
sys.argv=['run_act_shadow','--checkpoint',str(p/'model/policy.ckpt'),'--stats',str(p/'model/dataset_stats.pkl'),'--safety-summary',str(p/'model/safety_summary.json'),'--capture-npz',pre['capture_source'],'--output',str(root/'shadow_recorded'),'--device','cuda','--sample-stride','10','--task-coordinates-mm',*coords]
record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command_argv':sys.argv,'GPU':torch.cuda.get_device_name(0),'hardware_guards_enabled':True,'robot_construction_allowed':False,'CAN_socket_allowed':False,'input_scope':'saved real recording paired with retained visual geometry; not a live control loop'}
(root/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
with patch.object(NeroRobotWrapper,'__init__',side_effect=denied),patch.object(NeroRobotWrapper,'connect',side_effect=denied),patch.object(socket,'socket',side_effect=guarded_socket):
 code=run_act_shadow.main()
record.update(exit_code=code,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),motion_sent=False)
(root/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
raise SystemExit(code)
