from pathlib import Path
import json,hashlib,pickle,datetime
import numpy as np
import torch
p=Path('/home/user/wq/nero_act_4090_shadow_recovery_v9(2)/nero_act_4090_shadow_recovery_v9');root=Path('/home/user/wq/nero_act_v9_real_shadow_20261014')
assert not root.exists();root.mkdir()
manifest=json.loads((p/'package_manifest.json').read_text());checked=[]
for name,digest in manifest['files_sha256'].items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
 checked.append(name)
bundle=torch.load(p/'model/policy.ckpt',map_location='cpu',weights_only=False)
config=bundle['config']; declared=json.loads((p/'model/training_config.json').read_text())
assert config==declared and config['state_dim']==14 and config['action_dim']==8 and config['coordinate_conditioning']
assert config['camera_names']==['camera_0'] and config['chunk_size']==100 and config['pretrained_backbone'] is False
assert bundle.get('hardware_deployment_allowed') is False
with (p/'model/dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
for name,dim in [('qpos_mean',14),('qpos_std',14),('action_mean',8),('action_std',8)]:
 array=np.asarray(stats[name]);assert array.shape==(dim,) and np.isfinite(array).all()
 if name.endswith('std'):assert np.all(array>0)
measurements_path=Path('/home/user/wq/nero_readonly_handoff_20261014/real_measurements.json');measurements=json.loads(measurements_path.read_text());coords=measurements['source_xyz_mm']+measurements['destination_xyz_mm']
assert len(coords)==6 and np.isfinite(coords).all();assert measurements['paired_capture_directory']=='capture_new'
capture=measurements_path.parent/'capture_new/observations.npz'
with np.load(capture,allow_pickle=False) as d:
 names=d['joint_names'].tolist();state=d['state'][:,[names.index(n) for n in [f'joint{i}' for i in range(1,8)]+['gripper']]]
 rgb=d['rgb'];times=d['sample_monotonic_s'];assert state.shape==(101,8) and rgb.shape==(101,360,640,3) and np.all(np.diff(times)>0)
summary=json.loads((p/'model/safety_summary.json').read_text());lo=np.array(summary['train_qpos_min']);hi=np.array(summary['train_qpos_max']);outside=np.flatnonzero((state[0,:7]<lo-.15)|(state[0,:7]>hi+.15))
full=np.r_[state[0],np.array(coords)/1000];normalized=(full-np.asarray(stats['qpos_mean']))/np.asarray(stats['qpos_std'])
comparison=[{'joint':f'J{i+1}','recorded_rad':float(state[0,i]),'train_min_rad':float(lo[i]),'train_max_rad':float(hi[i]),'initial_margin_rad':.15,'outside_initial_range':bool(i in outside)} for i in range(7)]
result={
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'session_date_client':'2026-10-14',
 'package_path':str(p),'manifest_files_verified':len(checked),'package_sha256_check_passed':True,
 'checkpoint_sha256':manifest['files_sha256']['model/policy.ckpt'],'checkpoint_epoch':bundle.get('epoch'),
 'hardware_deployment_allowed':False,'checkpoint_config_matches_supplied_training_config':True,
 'state_dim':14,'action_dim':8,'camera_names':['camera_0'],'image_size':[90,160],'chunk_size':100,
 'joint_residual_anchor':config.get('joint_residual_anchor'),'normalization_dimensions_verified':True,
 'coordinate_input_mm':coords,'coordinate_input_semantics':'Nero-base source/destination cup geometric centres; mm converted to m in shadow loader',
 'coordinate_provenance':measurements['coordinate_provenance'],
 'independent_coordinate_survey_verified':False,'capture_matches_coordinate_scene':True,
 'capture_source':str(capture),'capture_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),
 'capture_timestamp_scope':'previous saved actual robot/scene record, not current live observation',
 'package_measurement_templates_not_used':True,'package_synthetic_smoke_data_not_used_as_real_input':True,
 'joint_training_range_comparison':comparison,'joints_outside_demonstration_range':[f'J{i+1}' for i in outside],
 'first_model_state_14':full.tolist(),'first_normalized_model_state_14':normalized.tolist(),
 'raw_gripper_feedback_mm':float(state[0,7]*1000),'raw_state_not_clipped':True,
 'no_safety_limits_or_model_deployment_flags_changed':True,'motion_sent':False,
 'planned_sample_stride':10,'planned_policy_queries':11,'model_loaded_for_metadata_only':True}
(root/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(root/'measurements_used.json').write_text(json.dumps(measurements,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'results_directory':str(root),'verified_package_files':len(checked),'checkpoint_epoch':result['checkpoint_epoch'],'outside_training_range':result['joints_outside_demonstration_range'],'first_normalized_model_state':normalized.tolist(),'motion_sent':False},indent=2))
