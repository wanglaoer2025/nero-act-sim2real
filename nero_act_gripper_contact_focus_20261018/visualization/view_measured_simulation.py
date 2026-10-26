"""Read-only desktop replay of measured ACT simulation states. No ROS/SDK/CAN."""
import argparse,hashlib,json,os,sys,time
from collections import deque
from pathlib import Path
import numpy as np
BASE=Path('/home/user/wq')
CODE=BASE/'nero_act_gripper_contact_focus_20261018'
sys.path.insert(0,str(CODE))
import mujoco
from simulation.scene import build_scene
from simulation.snapshot import restore,SPEC
from simulation.nero_domain_randomization import NeroServo,disturbance_torque
ROOT=Path(__file__).resolve().parent
CASES=[('run26 ORIGINAL SCENE - FULL ONLINE ACT SUCCESS (recorded simulation)',CODE/'evaluation/run26_nominal/cup_+0_+0')]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(index):
 label,folder=CASES[index];config=json.loads((folder/'scene_configuration.json').read_text())
 result=json.loads((folder/'result.json').read_text())
 assert result['real_hardware_tested'] is False
 paths=[folder/name for name in ['scene_configuration.json','physical_snapshots.json','rollout.npz','result.json']]
 bound={str(p):digest(p) for p in paths};cache=ROOT/f'record_{index}.npz';proof=ROOT/f'record_{index}_proof.json'
 if cache.exists() and proof.exists():
  report=json.loads(proof.read_text())
  if report['source_sha256']==bound and report['cache_sha256']==digest(cache):return config,result,np.load(cache)['physics']
 saved=json.loads((folder/'physical_snapshots.json').read_text())['snapshots'][0]
 trace=np.load(folder/'rollout.npz');states=trace['states'];targets=trace['sent_targets']
 assert states.shape==targets.shape and states.shape[1]==8 and len(states)>0
 model=build_scene(config);data=mujoco.MjData(model);servo=NeroServo(model,data,config);restore(model,data,servo,saved)
 values=[];maximum=0.
 for i,(state,target) in enumerate(zip(states,targets)):
  actual=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]];maximum=max(maximum,float(np.max(abs(actual-state))))
  if maximum>1e-7:raise ValueError(f'Physics reconstruction differs from measured record at{i}: {maximum}')
  value=np.empty(mujoco.mj_stateSize(model,SPEC));mujoco.mj_getState(model,data,value,SPEC);values.append(value)
  data.qfrc_applied[:7]=disturbance_torque(config,i);servo.command(target[:7],target[7])
  for _ in range(25):servo.apply();mujoco.mj_step(model,data)
  mujoco.mj_forward(model,data)
 value=np.empty(mujoco.mj_stateSize(model,SPEC));mujoco.mj_getState(model,data,value,SPEC);values.append(value)
 cup=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cup')
 cup_error=float(np.max(abs(data.xpos[cup]-np.array(result['last_observation']['cup_xyz']))))
 if cup_error>1e-7:raise ValueError('Reconstructed cup differs from recorded outcome')
 physics=np.array(values);np.savez_compressed(cache,physics=physics)
 report=dict(label=label,source=str(folder),source_sha256=bound,cache_sha256=digest(cache),
  states=len(physics),maximum_robot_state_error=maximum,final_cup_error=cup_error,
  record_success=result['success'],no_new_policy_inference=True,hardware_connected=False,
  display_only=True,physics_reconstructed_then_displayed_without_stepping=True)
 proof.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 return config,result,physics

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');a=parser.parse_args()
 prepared=[prepare(i) for i in range(len(CASES))]
 if a.verify_only:return
 import glfw
 import mujoco.viewer
 events=deque();index=0
 def key(code):events.append(code)
 while True:
  config,result,physics=prepared[index];label,folder=CASES[index]
  model=build_scene(config);data=mujoco.MjData(model);NeroServo(model,data,config)
  frame=0;paused=False;finished=False;switch=False;next_time=time.monotonic()
  mujoco.mj_setState(model,data,physics[0],SPEC);mujoco.mj_forward(model,data)
  with mujoco.viewer.launch_passive(model,data,key_callback=key,show_left_ui=False,show_right_ui=False) as view:
   with view.lock():
    view.cam.type=mujoco.mjtCamera.mjCAMERA_FREE;view.cam.lookat[:]=[-.30,-.07,.27]
    view.cam.distance=1.25;view.cam.azimuth=125;view.cam.elevation=-25
   (ROOT/'viewer_status.json').write_text(json.dumps(dict(pid=os.getpid(),opened=True,case=label,
    source=str(folder),hardware_connected=False,mode='recorded_simulation_state_replay'),indent=2))
   print(json.dumps(dict(gui_open=True,pid=os.getpid(),case=label)),flush=True)
   while view.is_running():
    while events:
     code=events.popleft()
     if code==glfw.KEY_SPACE:paused=not paused;next_time=time.monotonic()
     elif code==glfw.KEY_R:frame=0;finished=False;paused=False;next_time=time.monotonic()
     elif code==glfw.KEY_N:switch=True
    if switch:break
    now=time.monotonic()
    if not paused and not finished and now>=next_time:
     frame=min(frame+1,len(physics)-1);next_time+=.05
     if frame==len(physics)-1:finished=True;paused=True
    with view.lock():
     mujoco.mj_setState(model,data,physics[frame],SPEC);mujoco.mj_forward(model,data)
    state='END - '+('SUCCESS' if result['success'] else 'FAILED') if finished else 'PAUSED' if paused else 'PLAYING'
    error=result.get('error') or 'Full task completed'
    view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
     'ACT SIMULATION RECORD | NOT REAL ROBOT\n'+label+'\n'+state+f' | {frame*.05:.1f}s / {(len(physics)-1)*.05:.1f}s\n'+error,
     '\nSPACE pause/resume | R replay | N next record\nMouse drag: rotate | wheel: zoom | Esc: close\nObserved result retained; no automatic continuation'))
    view.sync();time.sleep(.01)
  if not switch:break
  index=(index+1)%len(CASES)
 (ROOT/'viewer_status.json').write_text(json.dumps(dict(pid=os.getpid(),opened=False,hardware_connected=False),indent=2))

if __name__=='__main__':main()
