from pathlib import Path
import json,pickle,hashlib,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/home/user/wq/nero_act_v9_real_shadow_20261014');p=Path('/home/user/wq/nero_act_4090_shadow_recovery_v9(2)/nero_act_4090_shadow_recovery_v9')
pre=json.loads((root/'preflight.json').read_text());execution=json.loads((root/'execution.json').read_text());r=json.loads((root/'shadow_recorded/report.json').read_text());rows=json.loads((root/'shadow_recorded/trace.json').read_text());summary=json.loads((p/'model/safety_summary.json').read_text())
assert execution['exit_code']==1 and r['initial_state_error'] and len(rows)==11 and r['inferred_observations']==11
assert r['checkpoint_sha256']==pre['checkpoint_sha256'] and r['motion_sent'] is False and r['robot_sdk_connected'] is False
with np.load(root/'shadow_recorded/predictions.npz',allow_pickle=False) as d:
 chunks=d['chunks'].copy();indices=d['sample_indices'].copy();assert chunks.shape==(11,100,8) and np.isfinite(chunks).all();assert indices.tolist()==list(range(0,101,10))
raw=np.asarray([row['first_predicted_action'] for row in rows]);safe=np.asarray([row['preview_safe_action'] for row in rows]);state=np.asarray([row['state'] for row in rows]);ms=np.asarray([row['inference_and_preprocessing_ms'] for row in rows]);normalized=np.asarray([row['normalized_state'] for row in rows])
assert np.allclose(chunks[:,0,:],raw) and np.isfinite(normalized).all() and np.max(np.abs(safe[:,:7]-state[:,:7]))<=.0075+1e-6
assert all(row['safety_error'] is None for row in rows)
with (p/'model/dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
coords=np.asarray(pre['coordinate_input_mm'])/1000
model_state=np.c_[state,np.repeat(coords[None],len(state),axis=0)]
assert np.allclose(normalized,(model_state-stats['qpos_mean'])/stats['qpos_std'],rtol=1e-5,atol=1e-4)
coord_names=['source_x','source_y','source_z','destination_x','destination_y','destination_z']
coordinate_comparison=[{'feature':name,'retained_visual_input_m':float(coords[i]),'training_mean_m':float(np.asarray(stats['qpos_mean'])[i+8]),'training_std_m':float(np.asarray(stats['qpos_std'])[i+8]),'normalized_input':float(normalized[0,i+8])} for i,name in enumerate(coord_names)]
fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
train_lo=np.asarray(summary['train_qpos_min']);train_hi=np.asarray(summary['train_qpos_max']);ys=np.arange(7)
for i in range(7):
 ax.hlines(i,train_lo[i],train_hi[i],color='#367baa',linewidth=9,label='Training qpos range' if i==0 else None)
 ax.hlines(i,train_lo[i]-.15,train_hi[i]+.15,color='#b9d2e2',linewidth=3,label='Initial-state range including 0.15 rad margin' if i==0 else None)
ax.scatter(state[0,:7],ys,color='#c64439',s=48,label='Recorded real qpos',zorder=5)
ax.set(yticks=ys,yticklabels=[f'J{i}' for i in range(1,8)],xlabel='Joint angle (rad)',title='Real initial state differs from v9 demonstration poses');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8,loc='lower right');fig.savefig(root/'joint_domain_comparison.png',dpi=170);plt.close(fig)
report={
 'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'recorded real-input shadow only; motion remains blocked',
 'model_package_verified_files':pre['manifest_files_verified'],'checkpoint_sha256':r['checkpoint_sha256'],
 'GPU':execution['GPU'],'capture_samples':101,'policy_queries':11,'prediction_shape':[11,100,8],
 'model_loading_and_normalization_verified':True,'finite_predictions':True,
 'shadow_exit_code':1,'shadow_passed':False,'initial_state_error':r['initial_state_error'],
 'joints_outside_demonstration_initial_range':pre['joints_outside_demonstration_range'],
 'joint_comparison':pre['joint_training_range_comparison'],'task_coordinate_training_comparison':coordinate_comparison,
 'task_coordinate_source_is_visual_estimate_not_independent_survey':True,
 'coordinate_geometry_scene_matches_saved_images':True,'recording_is_live_now':False,
 'first_query_ms':float(ms[0]),'all_queries_median_ms':float(np.median(ms)),
 'remaining_10_queries_p95_ms':float(np.percentile(ms[1:],95)),
 'remaining_10_queries_max_ms':float(ms[1:].max()),'timing_note':'No dedicated warmup occurred; overall reported P95 mixes first-query overhead with later inference. Saved-frame preprocessing/inference timings exclude live camera capture and actuator scheduling.',
 'single_step_filter_errors':r['safety_errors'],'single_step_filter_error_count_is_not_deployment_acceptance':True,
 'delta_clamps_per_query':[row['safety_info']['delta_clamps'] for row in rows],
 'training_range_anomalies_per_query':[row['safety_info']['range_anomalies'] for row in rows],
 'first_step_raw_joint_delta_max_rad':np.max(np.abs(raw[:,:7]-state[:,:7]),axis=0).tolist(),
 'safe_first_step_delta_cap_rad':.0075,
 'first_predicted_gripper_width_range_mm':[float(raw[:,7].min()*1000),float(raw[:,7].max()*1000)],
 'recorded_gripper_width_mm':float(state[0,7]*1000),
 'gripper_interpretation':'Model first action suggests reducing total opening from99.832mm toabout29.65mm. This is an offline prediction; no grasp-stage or physical contact validity has been established, so no gripper command was sent.',
 'fixed_destination_training_limitation':'Protocol trains a fixed place goal; destination features do not establish arbitrary place-point generalization.',
 'deployment_blockers':[
   'Checkpoint hardware_deployment_allowed=false and simulation acceptance false (4/8 nominal,0/4 recovery)',
   'Five real joints outside demonstration initial range, even with retained0.15rad margin',
   'Real source/destination XY differ substantially from training; current synthetic view/task geometry not reconstructed from real scene',
   'Old run_act_real scheduler does not implement matching2Hz query/20Hz first10-step asynchronous execution',
   'Raw feedback99.832mm exceeds separate V9 capture-audit98mm bound; no raw data clipping or threshold alteration used'],
 'next_simulation_alignment_inputs':{
    'initial_q_rad':state[0,:7].tolist(),'initial_gripper_total_width_m':float(state[0,7]),
    'source_cup_centre_nero_base_mm':pre['coordinate_input_mm'][:3],
    'destination_cup_centre_nero_base_mm':pre['coordinate_input_mm'][3:],
    'nominal_table_z_nero_base_mm':-10,'cup_diameter_mm':75,'cup_height_mm':95,'cup_empty_mass_g':50,
    'camera_calibration_file':'calibration/camera_to_base.json','camera_transform_translation_unit':'mm',
    'policy_image_orientation':'upright RGB; captured images already rotated180deg; avoid another rotation',
    'model_image_size':[90,160],'action_order':'joint1..joint7 radians + gripper total width metres',
    'scene_parameter_provenance':'measurements_used.json','request':'Reconstruct real starting pose, fixed ZED observation and task geometry in simulation, export corresponding demonstrations and independently evaluate retrained policy before physical grasp tests.'},
 'motion_sent':False,'robot_sdk_connected':False,'camera_connected':False,
 'limits_or_deployment_flags_modified':False,'accepted_for_autonomous_real_grasp':False,
 'no_new_numeric_measurements_requested':True}
(root/'domain_alignment_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(root/'calibration').mkdir();shutil=None
import shutil
for name in ['camera_to_base.json','tcp_offset.json']:
 shutil.copy2(Path('/home/user/wq/nero_readonly_handoff_20261014/calibration')/name,root/'calibration'/name)
shutil.copy2('/home/user/wq/nero_act_v9_shadow_input_tests_4090.log',root/'shadow_input_tests_4090.log')
print(json.dumps({'shadow_passed':False,'outside_demonstration_joints':report['joints_outside_demonstration_initial_range'],'GPU':report['GPU'],'first_query_ms':report['first_query_ms'],'later_10_queries_p95_ms':report['remaining_10_queries_p95_ms'],'first_gripper_prediction_mm':report['first_predicted_gripper_width_range_mm'],'motion_sent':False},indent=2))
