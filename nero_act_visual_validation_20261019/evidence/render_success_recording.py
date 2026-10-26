"""Render recorded ACT query snapshots only; no integration steps or robot APIs."""
import json,math
from pathlib import Path
import cv2,mujoco,numpy as np
from simulation.scene import build_scene
root=Path(__file__).resolve().parents[1]
case=root/'results/frozen_320_suite16/case_00/heldout_xy_0'
scene=json.loads((case/'scene_configuration.json').read_text())
result=json.loads((case/'result.json').read_text())
assert result['success'] is True
saved=json.loads((case/'physical_snapshots.json').read_text())['snapshots']
trace=np.load(case/'rollout.npz');times=trace['query_simulation_time_s'];steps=trace['query_steps']
model=build_scene(scene);data=mujoco.MjData(model)
output=root/'evidence/ACT_visual15_full_success_simulation_2x.mp4'
writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,360))
assert writer.isOpened()
frames=0;last_rgb=None
with mujoco.Renderer(model,height=360,width=640) as renderer:
 for i,snapshot in enumerate(saved):
  data.qpos[:]=snapshot['qpos'];data.qvel[:]=snapshot['qvel'];mujoco.mj_forward(model,data)
  renderer.update_scene(data,camera='calibrated_nero_camera');renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
  rgb=renderer.render().copy();last_rgb=rgb
  bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
  events=[e['kind'] for e in result['simulated_events'] if e['physics_step']<=steps[i]]
  phase=events[-1] if events else 'APPROACH'
  cv2.putText(bgr,'SIMULATION | Recorded ACT query states | 2x',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1,cv2.LINE_AA)
  cv2.putText(bgr,f't={times[i]:.2f}s  phase={phase}  query={i}',(8,37),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1,cv2.LINE_AA)
  duration=times[i+1]-times[i] if i+1<len(times) else .1
  for _ in range(max(1,round(duration*10))):writer.write(bgr);frames+=1
writer.release()
cv2.imwrite(str(root/'evidence/ACT_visual15_success_final.png'),cv2.cvtColor(last_rgb,cv2.COLOR_RGB2BGR))
probe=cv2.VideoCapture(str(output));count=int(probe.get(cv2.CAP_PROP_FRAME_COUNT));ok,_=probe.read();probe.release()
assert count==frames and ok
report=dict(video=str(output),frames=frames,fps=20,playback_speed=2,recorded_queries=len(saved),
 simulated_case='heldout_xy_0',full_task_success=True,source=str(case),
 max_lift_m=result['max_lift_m'],transfer_m=result['maximum_horizontal_transfer_m'],
 stable_released_frames=result['stable_released_frames'],
 display_scope='recorded query snapshots; gaps held for display; no newly generated control or physical integration',
 hardware_commands_sent=False)
(root/'evidence/success_video.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
