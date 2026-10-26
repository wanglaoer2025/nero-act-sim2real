"""Render exact saved MuJoCo query states, without generating new robot actions."""
import json
from fractions import Fraction
from pathlib import Path
import av,cv2,mujoco,numpy as np
from simulation.scene import build_scene
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/run15_simulation_operation_01'
CASE=OUT/'last_site_geometry_prepared_open_0'
result=json.loads((CASE/'result.json').read_text())
assert result['success'] and not result['real_hardware_tested']
record=json.loads((CASE/'physical_snapshots.json').read_text());snapshots=record['snapshots']
assert len(snapshots)>20 and len(snapshots)==len(record['query_steps'])
times=np.array([s['physics_state'][0] for s in snapshots]);assert np.all(np.diff(times)>0)
config=json.loads((CASE/'scene_configuration.json').read_text());model=build_scene(config);data=mujoco.MjData(model)
video=OUT/'ACT_full_pick_place_SIMULATION.mp4';assert not video.exists()
with av.open(str(video),'w') as container,mujoco.Renderer(model,height=360,width=640) as renderer:
 stream=container.add_stream('libx264',rate=2);stream.width=640;stream.height=400;stream.pix_fmt='yuv420p'
 stream.time_base=Fraction(1,1000);stream.codec_context.time_base=Fraction(1,1000)
 stream.options={'crf':'20','preset':'veryfast'}
 for index,(snapshot,t) in enumerate(zip(snapshots,times)):
  data.qpos[:]=snapshot['qpos'];data.qvel[:]=snapshot['qvel'];mujoco.mj_forward(model,data)
  renderer.update_scene(data,camera='calibrated_nero_camera')
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
  rgb=renderer.render().copy();frame_array=np.vstack([np.zeros((40,640,3),dtype=np.uint8),rgb])
  cv2.putText(frame_array,'SIMULATION | Online ACT | Known development scene | NOT HARDWARE',
   (6,15),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
  cv2.putText(frame_array,f'Query snapshots | sim time {t-times[0]:.1f}s | no expert task targets',
   (6,32),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
  frame=av.VideoFrame.from_ndarray(frame_array,format='rgb24');frame.pts=round((t-times[0])*1000);frame.time_base=Fraction(1,1000)
  for packet in stream.encode(frame):container.mux(packet)
 for packet in stream.encode():container.mux(packet)
with av.open(str(video)) as container:
 decoded=sum(1 for _ in container.decode(video=0))
assert decoded==len(snapshots)
report=dict(video=str(video),frames=decoded,format='h264_mp4',query_snapshot_timestamps_preserved=True,
 simulation_duration_s=float(times[-1]-times[0]),full_task_result_verified=True,
 development_scene_repeat=True,independent_acceptance=False,real_hardware_tested=False,
 max_lift_m=result['max_lift_m'],maximum_horizontal_transfer_m=result['maximum_horizontal_transfer_m'])
(OUT/'video_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
