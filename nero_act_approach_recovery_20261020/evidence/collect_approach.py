import copy,json
from pathlib import Path
import torch
from simulation.history_teacher import collect_partial
ROOT=Path(__file__).resolve().parents[1]
torch.cuda.init()
scene=json.loads((ROOT/'evidence/approach_failure_capture_v2/scene_configuration.json').read_text())
scene['generalization']=dict(index='nominal_development',source_group='nominal_diagnostic_train',split='train')
p=ROOT/'training_data/new_approach/episode_2019.hdf5'
r=collect_partial(scene,ROOT/'evidence/safe_approach_context.json','approach',p,parent_group='nominal_diagnostic_train',split='train')
(ROOT/'evidence/new_approach_verified.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
