import os
os.environ['MUJOCO_GL']='egl'
import sys,json,time
from pathlib import Path
import h5py,cv2,numpy as np,mujoco
ROOT=Path('/home/user/wq/nero_act_transfer_v2_20261025')
sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
from simulation.teacher import run_teacher
from observer import current_cup_component
from vendor.cylinder import NominalImageEstimator
from scipy.optimize import least_squares
OUT=ROOT/'evidence/cup_association_diagnosis'
OUT.mkdir(exist_ok=False)
source=ROOT/'data/v2_enclosed_missing_01/train_02_02.hdf5'
with h5py.File(source) as f: scene=json.loads(f.attrs['scene_config'])
profile=dict(camera=scene['camera_calibration'],table_top_z_m=scene['table_top_z_m'],cup=scene['cup'])
cylinder=NominalImageEstimator(profile,(720,1280))
renderer=None;count=0;rows=[];saved=set();start=time.monotonic()
def capture(model,data,action,stage):
 global renderer,count
 n=count;count+=1
 if n not in [0,1454,1455,1475,1500,1600,1700,2000,2300,2600,2900]:return
 if renderer is None:
  model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
  renderer=mujoco.Renderer(model,height=720,width=1280)
 renderer.update_scene(data,camera='calibrated_nero_camera')
 renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
 rgb=renderer.render().copy();hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 raw=((hsv[:,:,0]>=15)&(hsv[:,:,0]<=45)&(hsv[:,:,1]>85)&(hsv[:,:,2]>30)).astype('uint8')
 mask=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 num,labels,stats,_=cv2.connectedComponentsWithStats(mask)
 ids=[i for i in range(1,num) if stats[i,4]>=15]
 retained=np.isin(labels,ids);y,x=np.nonzero(retained)
 bounds=np.array([min(x)-.5,min(y)-.5,max(x)+.5,max(y)+.5])
 initial=cylinder.ray_at_table([(min(x)+max(x))/2,max(y)]);initial[2]=scene['table_top_z_m']
 def residual(bottom):
  pixels=cylinder.project(cylinder.surface+bottom)
  return np.r_[pixels.min(0),pixels.max(0)]-bounds
 fit=least_squares(residual,np.clip(initial,[-.9,-.6,-.03],[-.15,.4,.42]),bounds=([-.9,-.6,-.03],[-.15,.4,.42]),max_nfev=30,diff_step=1e-4)
 hull=cylinder.cup_hull(fit.x)
 pts=np.c_[x,y].astype('float32')
 inside=np.array([cv2.pointPolygonTest(hull,tuple(p),True) for p in pts])
 try:current_cup_component(raw);error=None
 except ValueError as exc:error=str(exc)
 cup=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cup')
 truth=data.xpos[cup]-[0,0,scene['cup']['height_m']/2]
 row=dict(frame=n,stage=stage,error=error,components=[stats[i].tolist() for i in ids],
  bbox_rmse_px=float(np.sqrt(np.mean(fit.fun**2))),fit_bottom=fit.x.tolist(),
  truth_diagnostic_error_mm=float(np.linalg.norm(fit.x-truth)*1000),
  outside_hull_pixels=int((inside<0).sum()),outside_hull_max_px=float(max(0,-inside.min())))
 rows.append(row)
 cv2.imwrite(str(OUT/f'frame_{n}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
 cv2.imwrite(str(OUT/f'mask_{n}.png'),raw*255)
 overlay=rgb.copy();cv2.polylines(overlay,[np.rint(hull).astype('int32')],True,(255,0,255),2)
 cv2.imwrite(str(OUT/f'fit_{n}.png'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
 print(json.dumps(row),flush=True)
result=run_teacher(scene,capture);result.pop('trace',None)
if renderer:renderer.close()
(OUT/'result.json').write_text(json.dumps(dict(source=str(source),rows=rows,teacher_result=result,elapsed_s=time.monotonic()-start,hardware_commands_sent=False),indent=2)+'\n')
print('complete',result['success'],flush=True)
