from pathlib import Path
import json,hashlib,collections,math,datetime,shutil,tarfile,tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/home/user/wq/nero_response_probe_v2_20261014'); probe=root/'logs/probe_001'; w=root.parent
trace=json.loads((probe/'trace.json').read_text()); declared=json.loads((probe/'report.json').read_text()); frames=[json.loads(line) for line in (probe/'can_frames.jsonl').open()];session=json.loads((root/'execution_session_4090.json').read_text());log=root/session['terminal_log']
assert session['test_exit_code']==0 and 'COMMAND_EXIT_CODE="0"' in log.read_text()
assert len(frames)==declared['can_frames']==23266 and len(trace)==121
assert dict(collections.Counter(r['phase'] for r in trace))=={'baseline':1,'ramp':20,'hold':40,'return':60}
commands=[r for r in trace if r.get('command_sent')];assert len(commands)==120
q0=np.asarray(trace[0]['feedback']['q']);q=np.asarray([r['feedback']['q'] for r in commands]);targets=np.asarray([r['command'] for r in commands]);times=np.asarray([r['feedback']['host_monotonic_s'] for r in commands])
assert q0.shape==(7,) and q.shape==targets.shape==(120,7) and np.isfinite(q).all() and np.isfinite(targets).all()
assert np.max(np.abs(targets[:,1:]-q0[1:]))==0
assert np.all(targets[:,0]>=q0[0]-1e-10) and np.max(targets[:,0]-q0[0])<=.005+1e-10
for name in ['joint','status']:
 stamps=np.array([r['feedback'][name+'_sdk_timestamp'] for r in trace]);assert np.isfinite(stamps).all() and np.all(np.diff(stamps)>0)
for row in trace:
 status=row['feedback']['status'];assert status['ctrl_mode']==1 and status['mode_feedback']==1 and status['arm_status']==status['err_code']==0
endpoint=q0.copy();endpoint[0]+=.005
holds=[r for r in commands if r['phase']=='hold'];returns=[r for r in commands if r['phase']=='return']
hold_errors=[float(np.max(np.abs(np.array(r['feedback']['q'])-endpoint))) for r in holds[-5:]]
return_errors=[float(np.max(np.abs(np.array(r['feedback']['q'])-q0))) for r in returns[-5:]]
progress=float(max(r['feedback']['q'][0]-q0[0] for r in holds))
assert progress>=.004 and max(hold_errors)<=.001 and max(return_errors)<=.001
assert declared['passed'] and declared['outbound_tracking_passed'] and declared['return_tracking_passed']
assert abs(progress-declared['positive_progress_rad'])<1e-10
hz=float(1/np.diff(times).mean());assert abs(hz-declared['observed_loop_hz'])<1e-8
rx_j1=[];status_counter=collections.Counter();by_direction={}
for label,flag in [('local_TX_echo',True),('RX',False)]:
 by_direction[label]=dict(collections.Counter(hex(f['can_id_with_flags']&0x1fffffff) for f in frames if f['local_tx_loopback']==flag))
for frame in frames:
 if frame['local_tx_loopback']:continue
 cid=frame['can_id_with_flags']&0x1fffffff;data=bytes.fromhex(frame['data_hex'])
 assert len(data)==frame['dlc']<=8
 if cid==0x2a5:
  rx_j1.append((frame['host_monotonic_s'],math.radians(int.from_bytes(data[:4],'big',signed=True)*.001)))
 if cid==0x2a1:
  status_counter[frame['data_hex']]+=1
  assert data[0]==1 and data[1]==0 and data[2]==1 and int.from_bytes(data[6:8],'big')==0
origin=trace[0]['feedback']['host_monotonic_s'];raw_t=np.array([x[0]-origin for x in rx_j1]);raw_q=np.array([x[1] for x in rx_j1]);elapsed=times-origin
mask=(raw_t>=0)&(raw_t<=elapsed[-1]+.1)
assert np.ptp(raw_q[mask])>.004
last_raw=raw_q[(raw_t>=elapsed[-1]-.25)&(raw_t<=elapsed[-1]+.1)]
assert len(last_raw)>20 and max(np.abs(last_raw-q0[0]))<.001
half_mdeg=math.radians(.0005)
# Match each copied SDK sample to independently received J1 frames near its host-read time.
match_deltas=[]
for row in commands:
 t=row['feedback']['host_monotonic_s']-origin;near=np.abs(raw_t-t)<=.03
 assert near.any()
 match_deltas.append(float(np.min(np.abs(raw_q[near]-row['feedback']['q'][0]))))
assert max(match_deltas)<half_mdeg
meaningful=[r['feedback']['host_monotonic_s']-origin for r in commands if r['feedback']['q'][0]-q0[0]>=.0005]
fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
axes[0].plot(raw_t[mask],1000*(raw_q[mask]-q0[0]),color='#d14940',linewidth=1.2,label='Independent CAN RX J1')
axes[0].plot(elapsed,1000*(targets[:,0]-q0[0]),'--',color='#252525',label='Logged command')
axes[0].plot(elapsed,1000*(q[:,0]-q0[0]),'.',color='#2675b8',markersize=4,label='Fresh SDK feedback')
axes[0].set(ylabel='J1 relative angle (mrad)',title='Nero v2: 1 s ramp, 2 s hold, 1 s return, 2 s observation');axes[0].legend(loc='upper right');axes[0].grid(alpha=.2)
axes[1].plot(elapsed,1000*np.abs(q[:,0]-targets[:,0]),label='J1 instantaneous target error',color='#2675b8');axes[1].axhline(1,ls='--',color='#d14940',label='Endpoint acceptance: 1 mrad');axes[1].set(xlabel='Time after baseline sample (s)',ylabel='Absolute error (mrad)');axes[1].legend();axes[1].grid(alpha=.2)
for ax in axes:
 for t in [1,3,4]:ax.axvline(t,color='#777777',alpha=.4,linewidth=.8)
 ax.set_xlim(0,elapsed[-1]+.1)
fig.savefig(probe/'response_trace_review.png',dpi=180);plt.close(fig)
result={
 'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'Independent offline review of one authorized J1 response probe; no ACT/V9 execution',
 'accepted_for_this_J1_probe':True,'ready_for_v9_execution':False,
 'trace_rows':len(trace),'commands_sent_logged':len(commands),'phase_counts':dict(collections.Counter(r['phase'] for r in trace)),
 'observed_command_loop_hz':hz,'sampled_command_duration_s':float(times[-1]-times[0]),
 'positive_J1_progress_rad':progress,'positive_J1_progress_deg':math.degrees(progress),
 'final_five_hold_max_seven_joint_error_rad':max(hold_errors),'final_five_hold_errors_rad':hold_errors,
 'final_five_return_max_seven_joint_error_rad':max(return_errors),'final_five_return_errors_rad':return_errors,
 'joint_and_status_SDK_timestamps_strictly_advanced':True,'all_recorded_controller_status_CAN_MOVE_J_no_fault':True,
 'first_recorded_J1_progress_at_least_0_0005_rad_elapsed_s':float(meaningful[0]),
 'progress_timing_scope':'First recorded position threshold crossing, not a measured command acceptance/physical latency; not comparable to old1.5-encoder-step noise threshold.',
 'CAN_evidence':{
   'frames':len(frames),'frame_counts_by_direction':by_direction,'independent_RX_J1_motion_peak_to_peak_rad':float(np.ptp(raw_q[mask])),
   'last_observed_RX_J1_max_distance_from_start_rad':float(max(np.abs(last_raw-q0[0]))),
   'all_SDK_J1_samples_have_matching_nearby_independent_RX':True,
   'match_window_s':.03,'maximum_nearby_angle_matching_difference_rad':max(match_deltas),
   'status_payload_counts':dict(status_counter),
   'local_TX_echo_frames':sum(f['local_tx_loopback'] for f in frames),
   'live_CAN_TX_capture_complete':False,
   'missing_TX_explanation':'Installed SDK config factory defaults local_loopback=False; observer received RX only. No individual target CAN TX evidence was recorded. This is not evidence that no command was sent.',
   'acknowledgement_and_response_latency_measured':False},
 'interpretation':'同一SDK链路在v2低速缓升与保持窗口内完成了J1小幅跟随及回位；新鲜SDK状态和独立CAN RX支持本轮响应判据。旧0.5秒测试没有响应的原因仍不能唯一归因，因为v2同时改变了时长和目标曲线，且旧日志缺少反馈时间戳。',
 'limitations':['只测试J1及当前姿态，其他关节只有保持观测，不代表逐轴运动验收。','CAN捕获为RX，SDK默认local_loopback=False，缺少同期TX与控制器目标接受证据。','SDK时间戳推进不是七轴硬件同步或绝对传感器年龄证明。','未运行ACT；没有验证V9块执行或抓杯任务。'],
 'controller_speed_commanded_percent':5,'controller_speed_restored':False,'ROS_driver_auto_restored':False,
 'gripper_commands_requested':False,'ACT_policy_used':False,'additional_motion_after_probe':False,
 'raw_sources_sha256':{name:hashlib.sha256((probe/name).read_bytes()).hexdigest() for name in ['trace.json','report.json','can_frames.jsonl']}}
(probe/'independent_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['accepted_for_this_J1_probe','commands_sent_logged','observed_command_loop_hz','positive_J1_progress_rad','final_five_hold_max_seven_joint_error_rad','final_five_return_max_seven_joint_error_rad','first_recorded_J1_progress_at_least_0_0005_rad_elapsed_s']},indent=2))
print(json.dumps({'all_SDK_J1_samples_match_independent_RX':True,'TX_echo_frames':0,'archive_pending':True},indent=2))
