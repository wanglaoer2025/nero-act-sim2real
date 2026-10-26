"""Training-only representability check; no independent evaluation labels used."""
import json
from pathlib import Path
import h5py,numpy as np
from training.train import complete_windows
root=Path(__file__).resolve().parents[1];run=root/'training/run10_supervised_release'
split=json.loads((run/'split.json').read_text());paths=[root/'training_data_current_source'/f'episode_{i}.hdf5' for i in split['train']]
samples=complete_windows(paths,20);cache={}
for i,p in enumerate(paths):
 with h5py.File(p,'r') as f:cache[i]=(f['action'][:],np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]])
maximum=np.zeros(8);bad=np.zeros(8,int)
for i,t in samples:
 actions,states=cache[i];delta=abs(actions[t:t+10]-states[t]);maximum=np.maximum(maximum,delta.max(axis=0));bad+=(delta>np.r_[np.full(7,.0295),.0095]).sum(axis=0)
report=dict(split='train_only',weighted_windows=len(samples),maximum_prefix_distance=maximum.tolist(),outside_v2=bad.tolist(),augmentation_allowance=[.0002]*7+[.0001],passed=bool(not bad.any() and np.all(maximum+np.r_[np.full(7,.0002),.0001]<np.r_[np.full(7,.0295),.0095])))
(root/'evidence/run10_training_readout_audit.json').write_text(json.dumps(report,indent=2));print(report);assert report['passed']
