"""Existing scene-binding procedure, no optimization or robot connection."""
import copy,hashlib,json,shutil
from pathlib import Path
import cv2,numpy as np,torch
from executor.bundle import verify_model_bundle
from executor.policy_worker import FrozenAct
from training.export import export_model

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'model/source_run11';metadata=verify_model_bundle(source)
payload=torch.load(source/'policy.ckpt',map_location='cpu',weights_only=False)
original=copy.deepcopy(payload['config'])
geometry=json.loads((ROOT/'evidence/scene_capture_initial/report.json').read_text())
scene=copy.deepcopy(json.loads((source/'scene_config.json').read_text()))
scene['pick_xy_m']=np.median([r['cup_xyz'][:2] for r in geometry['observations']],axis=0).tolist()
scene['place_xy_m']=np.median([r['target_xyz'][:2] for r in geometry['observations']],axis=0).tolist()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
destination=ROOT/'deployment/current_scene';(destination/'checkpoints').mkdir(parents=True,exist_ok=False)
(destination/'artifacts').mkdir()
config=copy.deepcopy(original)
config.update(source_training_scene_config=copy.deepcopy(original['scene_config']),scene_config=scene,
    deployment_scene_binding=dict(mode='posttraining_static_scene_binding_no_optimization',
        source_checkpoint_sha256=metadata['checkpoint_sha256'],
        original_training_config_sha256=metadata['training_config_sha256'],
        geometry_sha256=sha(ROOT/'evidence/scene_capture_initial/report.json'),
        prototype_sha256=sha(ROOT/'evidence/current_cup_prototype/summary.json')))
bound=dict(payload,config=config,posttraining_scene_binding=True,
           additional_optimization_steps=0,source_checkpoint_sha256=metadata['checkpoint_sha256'])
checkpoint=destination/'checkpoints/policy_scene_bound.ckpt';torch.save(bound,checkpoint)
shutil.copy2(source/'dataset_stats.pkl',destination/'checkpoints/dataset_stats.pkl')
summary_source=Path(original['artifact_dir'])/'training_summary.json';summary=json.loads(summary_source.read_text())
summary.update(deployment_derivation='scene binding only; original training metrics retained',
               source_training_summary=str(summary_source),additional_optimization_steps=0)
(destination/'artifacts/training_summary.json').write_text(json.dumps(summary,indent=2))
exported=ROOT/'model/frozen_run11';new_metadata=export_model(checkpoint,exported)
new=torch.load(exported/'policy.ckpt',map_location='cpu',weights_only=False)
assert new['model'].keys()==payload['model'].keys()
assert all(torch.equal(new['model'][key],payload['model'][key]) for key in payload['model'])
assert sha(source/'dataset_stats.pkl')==sha(exported/'dataset_stats.pkl')
assert new['config']['source_training_scene_config']==original['scene_config']
old_policy,new_policy=FrozenAct(source),FrozenAct(exported);maximum=0.
for row in geometry['observations'][::3]:
    rgb=cv2.cvtColor(cv2.imread(row['frame']),cv2.COLOR_BGR2RGB)
    a=old_policy.predict(rgb,geometry['actual_state']);b=new_policy.predict(rgb,geometry['actual_state'])
    maximum=max(maximum,float(abs(a-b).max()))
assert maximum==0.
report=dict(mode='current_scene_binding_no_optimization',passed=True,additional_optimization_steps=0,
    original_training_preserved=True,all_trained_tensors_bit_identical=True,normalization_file_identical=True,
    actual_image_prediction_max_difference=maximum,source_checkpoint_sha256=metadata['checkpoint_sha256'],
    deployment_checkpoint_sha256=new_metadata['checkpoint_sha256'],deployment_scene=scene,
    execution_ready=False,hardware_motion_sent=False)
(ROOT/'evidence/scene_binding_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='deployment_scene'}),flush=True)
