"""Current-image nominal card geometry; partial visibility never uses cached targets."""
import cv2
import numpy as np

from training.legacy_visual_data import NominalImageEstimator


class CardEdgesEstimator(NominalImageEstimator):
    # Nominal source model:100mm square, base-axis aligned; not independent real metrology.
    card_side_m=.1
    edge_tolerance_m=.006

    def component(self,mask,minimum):
        if minimum!=12:
            return NominalImageEstimator.component(mask,minimum)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        accepted=[]
        clouds=[]
        for index in range(1,count):
            if stats[index,cv2.CC_STAT_AREA]<minimum:
                continue
            y,x=np.nonzero(labels==index)
            pixels=np.c_[x,y]
            raw=self.size-1-((pixels+.5)/self.scale-.5)
            rays=np.c_[raw,np.ones(len(raw))]@np.linalg.inv(self.K).T@self.transform[:3,:3].T
            with np.errstate(divide='ignore',invalid='ignore'):
                depths=(self.profile['table_top_z_m']-self.transform[2,3])/rays[:,2]
                points=self.transform[:3,3]+rays*depths[:,None]
            inside=np.isfinite(points).all(1)&(depths>0)&(points[:,0]>=-.65)&(points[:,0]<=0)&(points[:,1]>=-.3)&(points[:,1]<=.3)
            if not inside.any():
                continue
            if not inside.all() or min(x)<=0 or max(x)>=self.width-1 or min(y)<=0 or max(y)>=self.height-1:
                raise ValueError('Blue object crosses workspace or image boundary')
            accepted.append(index)
            clouds.append(points)
        if not accepted:
            raise ValueError('No currently visible card geometry')
        points=np.concatenate(clouds)[:,:2]
        low,high=points.min(0),points.max(0)
        if np.any(abs((high-low)-self.card_side_m)>self.edge_tolerance_m):
            raise ValueError('Visible blue geometry does not span the nominal square card')
        corners=np.array([[low[0],low[1]],[low[0],high[1]],[high[0],low[1]],[high[0],high[1]]])
        distances=np.array([np.linalg.norm(points-corner,axis=1).min() for corner in corners])
        visible=int(np.sum(distances<=self.edge_tolerance_m))
        if visible<3:
            raise ValueError('At least three current visible card corners required')
        selected=np.isin(labels,accepted)
        polygon=np.array([[low[0],low[1]],[high[0],low[1]],
                          [high[0],high[1]],[low[0],high[1]]])
        world=np.c_[polygon,np.full(4,self.profile['table_top_z_m'])]
        interior=np.zeros(mask.shape,np.uint8)
        cv2.fillConvexPoly(interior,np.rint(self.project(world)).astype(np.int32),1)
        # Exclude one raster boundary pixel; interior gaps need a visible occluder.
        interior=cv2.erode(interior,np.ones((3,3),np.uint8)).astype(bool)
        hsv=self.current_hsv
        cup=(hsv[:,:,0]>=15)&(hsv[:,:,0]<=45)&(hsv[:,:,1]>85)&(hsv[:,:,2]>30)
        occluder=cup|(hsv[:,:,2]<85)
        unexplained=float(np.sum(interior&~selected&~occluder)/max(1,interior.sum()))
        if not interior.any() or unexplained>.1:
            raise ValueError('Missing card pixels are not explained by a current visible occluder')
        y,x=np.nonzero(selected)
        self.card_diagnostic=dict(target_visible_fragments=len(accepted),target_visible_corners=visible,
            target_size_fit_m=(high-low).tolist(),target_current_frame_only=True,
            target_shape='nominal100mm_square_axis_aligned',real_geometry_qualified=False,
            target_unexplained_interior_fraction=unexplained)
        return selected,np.array([x.mean(),y.mean()])

    def measure(self,rgb):
        self.card_diagnostic=None
        self.current_hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        cup,target,diagnostic=super().measure(rgb)
        if self.card_diagnostic is None:
            raise ValueError('No current card measurement')
        return cup,target,dict(diagnostic,**self.card_diagnostic)
