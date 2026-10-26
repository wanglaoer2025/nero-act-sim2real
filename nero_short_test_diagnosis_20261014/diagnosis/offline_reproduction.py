"""Replay installed SDK encoding and probe validation with all hardware paths blocked."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import collections, contextlib, csv, hashlib, inspect, io, json, math, socket, sys, threading, warnings
import numpy as np

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'nero_act_source'))
def denied(*args,**kwargs):raise AssertionError('Hardware use forbidden in offline diagnosis')
with patch.object(socket,'socket',side_effect=denied):
 import can,pyAgxArm
 from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v120.driver import Driver
 from pyAgxArm.protocols.can_protocol.drivers.core.arm_driver_abstract import ArmDriverAbstract
 from pyAgxArm.protocols.can_protocol.drivers.core.driver_context import DriverContext
 from pyAgxArm.protocols.can_protocol.comms import CommsFactory
 from src.act.robot_wrapper import NeroRobotWrapper
 from src.act import test_joint_stream
 with patch.object(can,'Bus',side_effect=denied),patch.object(CommsFactory,'create_comm',side_effect=denied),patch.object(pyAgxArm.AgxArmFactory,'create_arm',side_effect=denied),patch.object(ArmDriverAbstract,'__init__',side_effect=denied),patch.object(ArmDriverAbstract,'connect',side_effect=denied):
  class FakeFPS:
   def get_fps(self,*a):return 200
   def add_variable(self,*a):pass
   def increment(self,*a):pass
  sent=[]; ctx=DriverContext.__new__(DriverContext)
  ctx._tx_repeat_lock=threading.Lock();ctx._tx_repeat_min_interval={0x151:.1};ctx._tx_last_payload={};ctx._tx_last_stamp={};ctx.fps=FakeFPS()
  ctx.get_comm=lambda:SimpleNamespace(send=lambda frame:sent.append({'can_id':hex(frame.arbitration_id),'data_hex':bytes(frame.data).hex(),'time_s':clock[0]}))
  robot=Driver.__new__(Driver);robot._ctx=ctx;robot._parser=Driver._Parser(ctx.fps,config={});robot._msg_mode=robot._MSG_ModeCtrl();robot._auto_set_motion_mode_enabled=True;robot._joint_limits_enabled=False;robot._config={}
  clock=[1000.0];rows=list(csv.DictReader((root/'logs/act_joint_stream_20261014_224752.csv').open()));encoded=[]
  with patch('time.monotonic',side_effect=lambda:clock[0]):
   robot.set_follower_mode();robot.set_speed_percent(5);clock[0]+=.1
   initial_frames=list(sent)
   for row in rows:
    clock[0]=1000.1+float(row['timestamp_s']);before=len(sent);target=json.loads(row['command_q_target_rad']);robot.move_j(target);group=sent[before:]
    by_id={f['can_id']:bytes.fromhex(f['data_hex']) for f in group}
    assert all(cid in by_id for cid in ['0x155','0x156','0x157','0x170'])
    decoded=[]
    for cid in ['0x155','0x156','0x157','0x170']:
     parts=[(0,4)] if cid=='0x170' else [(0,4),(4,8)]
     decoded.extend(math.radians(int.from_bytes(by_id[cid][a:b],'big',signed=True)*.001) for a,b in parts)
    assert np.max(np.abs(np.array(target)-decoded))<=math.radians(.001)/2+1e-12
    encoded.append({'step':int(row['step']),'qtarget_rad':target,'decoded_target_rad':decoded,'frames':group})
   pre_normal=len(sent)
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always');robot.set_normal_mode()
   normal_sent=len(sent)-pre_normal
   assert normal_sent==0 and caught
  method_resolution={name:{'module':getattr(Driver,name).__module__,'file':inspect.getfile(getattr(Driver,name)),'line':inspect.getsourcelines(getattr(Driver,name))[1]} for name in ['set_normal_mode','set_follower_mode','_maybe_set_motion_mode','move_j','get_joint_angles']}
  cached=SimpleNamespace(msg=json.loads(rows[0]['feedback_qpos_rad']),timestamp=1.0,hz=0.0)
  wrapper=NeroRobotWrapper.__new__(NeroRobotWrapper);wrapper.connected=True;wrapper.robot=SimpleNamespace(get_joint_angles=lambda:cached);wrapper.feedback_timeout_s=.15;wrapper.poll_interval_s=.01
  old_sample=wrapper.get_arm_qpos();assert np.array_equal(old_sample,np.asarray(cached.msg,dtype=np.float32))
  q0=old_sample.copy();virtual=[0.0];commands=[];fake=SimpleNamespace(send_arm_qpos=lambda q:commands.append(q.copy()),get_arm_qpos=lambda:q0.copy())
  args=SimpleNamespace(frequency=20.,duration=.5,amplitude_rad=.005,joint_index=0,log_dir=Path(__file__).resolve().parent/'offline_replay_logs')
  with patch('time.monotonic',side_effect=lambda:virtual[0]),patch('time.sleep',side_effect=lambda dt:virtual.__setitem__(0,virtual[0]+dt)),contextlib.redirect_stdout(io.StringIO()) as output:
   replay_log=test_joint_stream.execute_stream(fake,q0,args)
  replay_rows=list(csv.DictReader(replay_log.open()));assert len(replay_rows)==10
  assert all(not row['error'] for row in replay_rows)
  assert np.ptp([json.loads(row['feedback_qpos_rad'])[0] for row in replay_rows])==0
  assert np.ptp([json.loads(row['command_q_target_rad'])[0] for row in replay_rows])>.009
  assert len(commands)==11
  result={
    'scope':'installed SDK packing and static-feedback acceptance reproduced entirely offline; not a hardware test',
    'hardware_constructor_and_CAN_socket_guards_enabled':True,'hardware_commands_sent':False,
    'method_resolution':method_resolution,
    'normal_mode_sent_frame_count':normal_sent,'normal_mode_warning':[str(w.message) for w in caught],
    'follower_and_speed_initial_virtual_frames':initial_frames,
    'CSV_target_encoding_verified_steps':len(encoded),'joint_target_frames_per_step':4,
    'decoded_J1_target_peak_to_peak_rad':float(np.ptp([item['decoded_target_rad'][0] for item in encoded])),
    'maximum_encoding_roundoff_rad':float(max(np.max(np.abs(np.array(e['qtarget_rad'])-e['decoded_target_rad'])) for e in encoded)),
    'changing_J1_targets_survive_TX_throttle':True,
    'warning_scope':'Frames are synthetic SDK output into a fake comm, not captured CAN TX from the real test.',
    'frame_replay':encoded,
    'stale_timestamp_sample_accepted_by_original_wrapper':True,
    'cached_sample_timestamp':cached.timestamp,'cached_sample_hz':cached.hz,
    'stationary_feedback_stream_returns_normally':True,'stationary_feedback_replay_frames':len(replay_rows),
    'offline_return_also_accepts_unchanged_q0':True,'offline_total_targets_including_return':len(commands),
    'offline_replay_csv':str(replay_log.relative_to(root)),
    'conclusion':'Coding/validation defects reproduced; exact physical reason for absent motion in prior test cannot be uniquely reconstructed without contemporaneous TX/status/feedback timestamps.'}
  dest=Path(__file__).with_name('offline_reproduction_result.json');dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps({key:result[key] for key in ['hardware_commands_sent','normal_mode_sent_frame_count','CSV_target_encoding_verified_steps','decoded_J1_target_peak_to_peak_rad','maximum_encoding_roundoff_rad','changing_J1_targets_survive_TX_throttle','stale_timestamp_sample_accepted_by_original_wrapper','stationary_feedback_stream_returns_normally']},indent=2))
