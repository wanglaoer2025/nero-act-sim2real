import json
from pathlib import Path
import torch
from simulation.history_teacher import collect_partial
ROOT=Path('/home/user/wq/nero_act_contact_recovery_20261020');torch.cuda.init()
scene=json.loads((ROOT/'evidence/contact_failure_capture/scene_configuration.json').read_text())
scene['generalization']=dict(index='nominal_development',source_group='nominal_diagnostic_train',split='train')
context=ROOT/'evidence/partial_contact_context.json';output=ROOT/'training_data/new_partial_contact/episode_2018.hdf5'
report=collect_partial(scene,context,'pre_close',output,parent_group='nominal_diagnostic_train',split='train')
(ROOT/'evidence/new_partial_contact_verified.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
