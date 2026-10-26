"""Render completed case9006 query snapshots; no integration or control commands."""
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from simulation.scene import build_scene

case = ROOT/'run_01/9006'
scene = json.loads((case/'scene_configuration.json').read_text())
result = json.loads((case/'result.json').read_text())
assert result['success'] and result['error'] is None
saved = json.loads((case/'physical_snapshots.json').read_text())['snapshots']
trace = np.load(case/'rollout.npz')
steps, times = trace['query_steps'], trace['query_simulation_time_s']
assert len(saved)==len(steps)==len(times)
model = build_scene(scene)
data = mujoco.MjData(model)
video = ROOT/'case9006_full_flow_simulation_2x.mp4'
writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,360))
assert writer.isOpened()
frames = 0
event_steps = {e['kind']:e['physics_step'] for e in result['simulated_events']}
selected = {0:'START'}
for event,step in event_steps.items():
    selected[min(int(np.searchsorted(steps,step+5)),len(saved)-1)] = event
selected[len(saved)-1] = 'END QUERY'
panels = []
with mujoco.Renderer(model,height=360,width=640) as renderer:
    for i,snapshot in enumerate(saved):
        data.qpos[:]=snapshot['qpos']
        data.qvel[:]=snapshot['qvel']
        mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera='calibrated_nero_camera')
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        bgr=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        events=[e['kind'] for e in result['simulated_events'] if e['physics_step']<=steps[i]]
        stage=events[-1] if events else 'APPROACH'
        cv2.rectangle(bgr,(0,0),(640,42),(25,25,25),-1)
        cv2.putText(bgr,'SIMULATION | recorded ACT query states | 2x',(8,16),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(bgr,f'case9006  t={times[i]:.2f}s  last event={stage}',(8,34),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
        duration=times[i+1]-times[i] if i+1<len(times) else .1
        for _ in range(max(1,round(duration*10))):
            writer.write(bgr)
            frames+=1
        if i in selected:
            panel=bgr.copy()
            cv2.putText(panel,selected[i],(10,350),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
            panels.append(panel)
writer.release()
assert len(panels)==6
sheet=np.vstack([np.hstack(panels[i:i+3]) for i in (0,3)])
cv2.imwrite(str(ROOT/'case9006_contact_sheet.jpg'),sheet)
probe=cv2.VideoCapture(str(video))
count=int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
ok,_=probe.read()
probe.release()
assert count==frames and ok
report=dict(complete=True,case_id=9006,source=str(case),recorded_queries=len(saved),
    video=str(video),frames=frames,fps=20,nominal_playback_speed=2,
    timing='Query intervals rounded to20fps; gaps held; presentation is not real-time execution.',
    camera='same configured calibrated_nero_camera, nominal calibration candidate',
    display_scope='recorded query snapshots only; final100-frame stability measured in original measurements',
    newly_integrated_physics_steps=0,hardware_commands_sent=False)
(ROOT/'recording.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
