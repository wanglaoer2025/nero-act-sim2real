import sys,os,json
from pathlib import Path
protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
sys.path.insert(0,'/home/user/wq/nero_act_visual_feedback_20261019')
from executor.runtime import SceneObserver
scene=json.loads(Path(sys.argv[1]).read_text());sensor=SceneObserver(scene,Path(sys.argv[2]))
from ultralytics import YOLO
sensor.model=YOLO('/home/user/projects/nero/yolov8n.pt')
sensor.cup_id=next(i for i,n in sensor.model.names.items() if n=='cup')
sensor.args.weights='/home/user/projects/nero/yolov8n.pt'
print(json.dumps(dict(type='ready')),file=protocol,flush=True)
for line in sys.stdin:
 request=json.loads(line)
 if request.get('stop'):break
 try:
  r=sensor.measure(request)
  response=dict(type='geometry',index=request['index'],cup=r['cup_xyz'].tolist(),target=r['target_xyz'].tolist(),camera_monotonic_s=r['camera_monotonic_s'],frame=r['frame'],fit_rmse_px=float(r['fit_rmse_px']))
 except Exception as exc:response=dict(type='error',error=repr(exc))
 print(json.dumps(response),file=protocol,flush=True)
