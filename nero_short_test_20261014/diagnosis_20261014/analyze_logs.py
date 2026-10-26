"""Offline comparison of the short probe with existing historical control logs."""
from pathlib import Path
import csv,json,hashlib,math,datetime,shutil,tarfile,tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent; root=out.parent; w=root.parent
paths=[root/'logs/act_joint_stream_20261014_224752.csv',*sorted(Path('/home/user/projects/nero/logs').glob('act_joint_stream_*.csv'))]
comparison=[]; raw=[]
for p in paths:
 rows=list(csv.DictReader(p.open())); t=np.array([float(r['timestamp_s']) for r in rows]); cmd=np.array([json.loads(r['command_q_target_rad']) for r in rows]); q=np.array([json.loads(r['feedback_qpos_rad']) for r in rows]); j=int(rows[0]['joint_index']); assert j==0
 assert q.shape==cmd.shape==(len(rows),7) and np.isfinite(q).all() and np.isfinite(cmd).all() and (np.diff(t)>0).all()
 delta=np.abs(q[:,j]-q[0,j]); significant=np.flatnonzero(delta>1.5*math.radians(.001)); first_any=np.flatnonzero(delta>0)
 changed=np.any(q!=q[0],axis=1);changed_idx=np.flatnonzero(changed)
 period=float(rows[0]['duration_s']);amplitude=float(rows[0]['amplitude_rad'])
 record={'file':p.name,'source_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'samples':len(rows),'joint_index':j,'frequency_hz':float(rows[0]['requested_frequency_hz']),'sine_period_s':period,'amplitude_rad':amplitude,'command_peak_to_peak_rad':float(np.ptp(cmd[:,j])),'feedback_peak_to_peak_rad':float(np.ptp(q[:,j])),'first_any_J1_change_s':float(t[first_any[0]]) if len(first_any) else None,'first_J1_change_gt_1_5_encoder_steps_s':float(t[significant[0]]) if len(significant) else None,'first_any_seven_joint_change_s':float(t[changed_idx[0]]) if len(changed_idx) else None,'first_half_second_frames':int((t<.5).sum()),'first_half_second_J1_feedback_span_rad':float(np.ptp(q[t<.5,j])),'ideal_peak_target_velocity_rad_s':amplitude*2*math.pi/period,'ideal_peak_target_acceleration_rad_s2':amplitude*(2*math.pi/period)**2,'controller_speed_percent_documented_in_CSV':False}
 comparison.append(record);raw.append((p,t,cmd,q))
assert all(record['first_half_second_J1_feedback_span_rad']==0 for record in comparison)
assert len(comparison)==7
(out/'historical_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
with (out/'historical_comparison.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
p,t,cmd,q=raw[0];axes[0].plot(t,1000*(cmd[:,0]-q[0,0]),'o-',label='Recorded target',color='#2979b9');axes[0].plot(t,1000*(q[:,0]-q[0,0]),'s-',label='Recorded feedback',color='#c74439');axes[0].set(xlabel='Time from stream start (s)',ylabel='J1 relative angle (mrad)',title='Current test: only 0.5 seconds recorded',xlim=(-.01,.51));axes[0].legend();axes[0].grid(alpha=.2)
for p,ts,targets,qs in raw[1:]:
 mask=ts<=1.25;axes[1].plot(ts[mask],1000*(qs[mask,0]-qs[0,0]),'.-',label=p.stem.replace('act_joint_stream_',''),alpha=.8)
axes[1].axvspan(0,.5,color='#c74439',alpha=.12,label='Current 0.5 s window');axes[1].set(xlabel='Time from stream start (s)',ylabel='Historical J1 change (mrad)',title='Six historical runs: changes start after 0.7-0.8 s',xlim=(0,1.25));axes[1].legend(fontsize=7,ncol=2);axes[1].grid(alpha=.2)
fig.savefig(out/'j1_response_comparison.png',dpi=180);plt.close(fig)
replay=json.loads((out/'offline_reproduction_result.json').read_text());old=json.loads((root/'short_test_result_4090.json').read_text())
report={
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'日志、历史对照和现有SDK离线复现；本轮没有连接机器人SDK、发送CAN帧、使能或追加运动。',
 'summary':'目前最有证据支持的是本轮观察窗口过短；同时已证实原脚本缺少目标跟随与反馈新鲜度验收，静止反馈也能退出0。日志不足以唯一确定本次实际不动的物理原因。',
 'root_cause_confidence':{
   'stationary_feedback_can_pass_original_script':'已离线复现的软件验收缺陷',
   'stale_feedback_can_pass_getter':'已离线复现的软件新鲜度校验缺陷；不等于已证明本次是旧缓存',
   'short_window_hides_initial_response':'历史对照强支持的首要解释；不是本次已测定的固定延迟',
   'follower_is_wrong_mode':'不能据旧版实现判断；V120按SDK设计使用Follower，normal_mode是空操作',
   'actual_physical_motion_failure_cause_uniquely_established':False},
 'current_run':{
   'sample_count':10,'host_loop_hz':old['observed_host_loop_hz'],'recorded_time_range_s':[float(raw[0][1][0]),float(raw[0][1][-1])],
   'sine_period_and_stream_duration_s':.5,'speed_percent_commanded':5,'J1_feedback_unique_values':1,
   'target_peak_to_peak_rad':old['J1_command_peak_to_peak_rad'],'feedback_peak_to_peak_rad':0,
   'last_logged_target_relative_to_first_feedback_rad':float(raw[0][2][-1,0]-raw[0][3][0,0]),
   'after_stream':'发送q0并按0.01rad宽松条件等待回位；CSV不含回位反馈时间序列',
   'no_ACT_or_camera_or_task_coordinate_used':True},
 'historical_comparison':{
   'historical_log_count':6,'first_half_second_stationary_in_all_historical_logs':True,
   'historical_observed_first_significant_J1_change_s':[r['first_J1_change_gt_1_5_encoder_steps_s'] for r in comparison[1:]],
   'historical_significant_change_rule':'反馈偏离首帧大于1.5个0.001度编码单位，约2.62e-5rad',
   'historical_log_amplitude_rad':.02,'historical_sine_period_s':5,
   'current_to_historical_ideal_peak_velocity_ratio':2.5,'current_to_historical_ideal_peak_acceleration_ratio':25,
   'interpretation':'0.5秒发送结束后立即回位，早于历史0.70–0.80秒首次明显变化记录。小幅度不等于慢轨迹；本次周期缩短10倍，要求速度2.5倍、加速度25倍，同时速度限制为5%。这支持启动/规划或低速响应被过短窗口截断，但不能给出本次物理延迟定值。',
   'limitations':'历史姿态、幅度、周期与本轮不同，CSV未记控制器速度、控制器状态、命令接受时间或反馈设备时间戳。观察到的首次角度变化不是纯CAN传输时延。'},
 'offline_encoding_replay':{
   'source':'offline_reproduction_result.json','all_10_targets_form_4_seven_joint_control_frames':True,
   'target_frames_CAN_IDs':['0x155','0x156','0x157','0x170'],
   'J1_decoded_target_peak_to_peak_rad':replay['decoded_J1_target_peak_to_peak_rad'],
   'maximum_roundoff_rad':replay['maximum_encoding_roundoff_rad'],
   'TX_throttling':'SDK默认仅抑制0x151的100ms内重复模式帧；变动的J1目标帧仍通过，离线复现每步4个关节目标帧。',
   'conclusion':'未发现rad/deg换算、关节维度、J1索引或量化归零问题。仅证实编码路径；没有本次实时TX抓包，不能把虚拟输出当作实机接受证明。'},
 'software_validation_findings':[
   {'finding':'静止反馈通过发送测试','evidence':'原execute_stream只因反馈TimeoutError设置row.error；没有目标跟随误差或进展失败判据。离线模拟10步反馈全静止仍返回并写出无errors CSV。','source':'nero_act_source/src/act/test_joint_stream.py:185'},
   {'finding':'未检查反馈更新','evidence':'原get_arm_qpos只检查shape、有限值；timestamp=1.0、hz=0的缓存仍直接返回。实机CSV没有timestamp，无法恢复本轮新鲜度。','source':'nero_act_source/src/act/robot_wrapper.py:95'},
   {'finding':'回位容差大于测试幅度','evidence':'RETURN_TOLERANCE_RAD=0.01rad，测试幅度0.005rad；静止在q0即可通过。回位等待无日志化时间序列。','source':'nero_act_source/src/act/test_joint_stream.py:163'},
   {'finding':'没有实机命令接受或状态跟踪证据','evidence':'CSV只有目标、读回值和主机函数耗时；没有逐步arm_status/motion_status、SDK时间戳、CAN TX确认。0.4ms move_j耗时不是实体运动响应时间。','source':'logs/act_joint_stream_20261014_224752.csv'}],
 'firmware_specific_mode_correction':{
   'prior_mode_candidate_revised':True,'actual_profile':'NeroFW.V120; Driver inherits V112/V111/default',
   'normal_mode_resolves_to':replay['method_resolution']['set_normal_mode'],
   'normal_mode_sent_frames_in_offline_replay':0,
   'corrected_conclusion':'V112+取消旧normal_mode；V120使用Leader/Follower是SDK设计。不应把set_follower_mode替换成无效的set_normal_mode作为修复。旧默认类写CAN-push字段也不能证明V120实机推送已关闭。',
   'physical_feedback_evidence':'测试后2秒被动接收6196帧，其中J1/J2帧400个；控制器CAN_CTRL/MOVE_J、arm_status=0、err_code=0。仅证明测试后反馈正常，不补证测试期间反馈。'},
 'remaining_unknowns':['本次每个目标是否被控制器接受','发送期间SDK读到的数据时间戳是否更新','当前姿态和5%速度下的真实启动/规划延迟','0.5秒后是否曾出现短暂运动，原CSV未覆盖该阶段'],
 'next_steps_proposed_not_executed':[
   '先完善离线可验证的反馈时间戳/序列更新校验、状态日志与目标跟随结果；保持静止反馈应得到未验证/失败结果。',
   '下一次经授权的同幅度低速测试，把命令发送时段与观测/保持时段分开，覆盖历史起动窗口并记录回位阶段；同步被动抓取真实TX和RX。',
   '根据实机记录决定控制周期与块执行方式，不以本次退出码0进入ACT部署。'],
 'no_production_source_modified':True,'new_motion_performed':False,'physical_ROOT_cause_retest_done':False,
 'references':{'history_data':'historical_comparison.json','history_table':'historical_comparison.csv','plot':'j1_response_comparison.png','offline_reproduction':'offline_reproduction.py','offline_reproduction_results':'offline_reproduction_result.json'},
 'preserved_original_reports':'原short_test_result_4090.json和原回传包保留，不覆盖；本报告补充V120继承链与历史起动窗口分析。'}
(out/'diagnosis_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
# Include exact versions of the inheritance code inspected; these are evidence, not a patch.
(out/'sdk_inheritance_evidence').mkdir(exist_ok=True)
for version in ['v111','v112','v120']:
 source=Path('/home/user/nero/lib/python3.12/site-packages/pyAgxArm/protocols/can_protocol/drivers/nero/versions')/version/'driver.py'
 shutil.copy2(source,out/'sdk_inheritance_evidence'/f'driver_{version}.py')
for item in report['software_validation_findings']:assert item['finding']
print(json.dumps({'diagnosis_report':str(out/'diagnosis_report.json'),'historical_runs':6,'all_historical_J1_static_first_half_second':True,'earliest_historical_significant_change_s':min(r['first_J1_change_gt_1_5_encoder_steps_s'] for r in comparison[1:]),'latest_historical_significant_change_s':max(r['first_J1_change_gt_1_5_encoder_steps_s'] for r in comparison[1:]),'new_motion_performed':False},ensure_ascii=False,indent=2))
