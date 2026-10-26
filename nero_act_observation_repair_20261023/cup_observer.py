"""Current-frame visible cup silhouette fitting; nominal offline candidate only."""
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'nero_act_contact_escape_fix_20261023'))
from card_edges import CardEdgesEstimator


class VisibleBoundaryCupEstimator(CardEdgesEstimator):
    def measure(self,rgb):
        seed,target,legacy=super().measure(rgb)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        cup,_=self.component((hsv[:,:,0]>=15)&(hsv[:,:,0]<=45)&(hsv[:,:,1]>85)&(hsv[:,:,2]>30),15)
        background=(~cup)&(hsv[:,:,2]>=85)
        points=[]
        # Pixel-edge coordinates distinguish a visible outline from a dark finger boundary.
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            adjacent=np.roll(background,(-dy,-dx),axis=(0,1))
            y,x=np.nonzero(cup&adjacent)
            points.extend(np.c_[x+dx*.5,y+dy*.5].tolist())
        points=np.asarray(points,float)
        if len(points)<12 or min(np.ptp(points,axis=0))<8:
            raise ValueError('Visible cup boundary has insufficient geometric support')
        def residual(bottom):
            hull=self.cup_hull(bottom).astype(float)
            start=hull;end=np.roll(hull,-1,axis=0);delta=end-start
            offset=points[:,None,:]-start[None]
            ratio=np.clip(np.sum(offset*delta[None],axis=2)/np.maximum(np.sum(delta**2,axis=1),1e-12)[None],0,1)
            distance=offset-ratio[:,:,None]*delta[None]
            return np.sqrt(np.min(np.sum(distance**2,axis=2),axis=1)+1e-12)
        fit=least_squares(residual,seed,bounds=([-.9,-.6,-.03],[-.15,.4,.42]),
            loss='soft_l1',f_scale=.5,diff_step=1e-4,max_nfev=100)
        singular=np.linalg.svd(fit.jac,compute_uv=False)
        if not fit.success or singular[-1]<=0 or singular[0]/singular[-1]>100:
            raise ValueError('Visible cup boundary pose is underconstrained')
        sigma=np.sqrt(np.diag(np.linalg.inv(fit.jac.T@fit.jac)))*.5
        p95=float(np.percentile(fit.fun,95))
        if p95>1.25 or max(sigma)>.003:
            raise ValueError('Visible cup boundary does not support a precise nominal pose')
        return fit.x,target,dict(legacy,method='current_RGB_visible_boundary_cylinder',
            legacy_bbox_cup_position_m=seed.tolist(),visible_boundary_points=len(points),
            visible_boundary_p95_px=p95,nominal_linearized_std_m=sigma.tolist(),
            truth_used_as_observer_input=False,runtime_qualified=False,error_bound_m=None)
