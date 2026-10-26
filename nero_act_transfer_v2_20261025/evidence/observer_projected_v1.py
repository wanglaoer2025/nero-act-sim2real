"""One image-domain card observer for offline training and timing integration.

Registration never authorizes use under full occlusion. Metric accuracy remains
nominal until independent real geometry validation; no truth input is accepted.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from vendor.cylinder import NominalImageEstimator


def current_cup_component(mask, cylinder=None):
    connected=cv2.morphologyEx(np.asarray(mask,np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    try:
        # Preserve every previously accepted input/output exactly.
        return NominalImageEstimator.component(connected,15)[0]
    except ValueError:
        count,labels,stats,_=cv2.connectedComponentsWithStats(connected)
        ids=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=15]
        retained=np.isin(labels,ids)
        enclosing=[]
        for i in ids:
            y,x=np.nonzero(labels==i)
            hull=cv2.convexHull(np.c_[x,y].astype('int32'))
            envelope=np.zeros_like(connected)
            cv2.fillConvexPoly(envelope,hull,1)
            if not (retained&~envelope.astype(bool)).any():enclosing.append(i)
        # Every retained fragment must lie inside ONE observed body's hull;
        # a second exterior object is never discarded or merged by distance.
        if len(enclosing)==1:return retained
        if cylinder is None or len(ids)<2:
            raise ValueError('Unique visible cup and target required')
        # A finger can split the visible outline itself. Associate fragments
        # against the projected complete cup, never the largest blob alone.
        y,x=np.nonzero(retained)
        if min(x)==0 or max(x)==cylinder.width-1 or min(y)==0 or max(y)==cylinder.height-1:
            raise ValueError('Cup clipped by image boundary')
        bounds=np.array([min(x)-.5,min(y)-.5,max(x)+.5,max(y)+.5])
        initial=cylinder.ray_at_table([(min(x)+max(x))/2,max(y)])
        initial[2]=cylinder.profile['table_top_z_m']
        def residual(bottom):
            pixels=cylinder.project(cylinder.surface+bottom)
            return np.r_[pixels.min(0),pixels.max(0)]-bounds
        fit=least_squares(residual,np.clip(initial,[-.9,-.6,-.03],[-.15,.4,.42]),
                          bounds=([-.9,-.6,-.03],[-.15,.4,.42]),max_nfev=30,diff_step=1e-4)
        if not fit.success or np.sqrt(np.mean(fit.fun**2))>2.5:
            raise ValueError('Fragments inconsistent with one nominal cup')
        hull=cylinder.cup_hull(fit.x)
        outline=cv2.convexHull(np.c_[x,y].astype('float32')).reshape(-1,2)
        if any(cv2.pointPolygonTest(hull,tuple(point),True)<-2.5 for point in outline):
            raise ValueError('Exterior colour candidate outside projected cup')
        return retained


class RegisteredCardObserver:
    def __init__(self, profile, image_size):
        self.profile = profile
        self.height, self.width = image_size
        camera = profile['camera']
        self.K = np.asarray(camera['intrinsic_matrix'], float)
        self.T = np.asarray(camera['transform_base_from_raw_optical_m'], float)
        self.native = np.asarray(camera['resolution_wh'], float)
        self.scale = np.array([self.width, self.height])/self.native
        if (self.K.shape != (3,3) or self.T.shape != (4,4)
                or not np.isfinite(np.r_[self.K.ravel(),self.T.ravel(),self.scale]).all()
                or np.any(self.scale <= 0)):
            raise ValueError('Finite paired optical calibration required')
        self.registered = False
        self.last_frame = -1
        self.last_stamp = -np.inf

    def _masks(self, rgb):
        if rgb.dtype != np.uint8 or rgb.shape != (self.height,self.width,3):
            raise ValueError('RGB must match declared calibration image dimensions')
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        blue = ((hsv[:,:,0]>=95)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>90)&(hsv[:,:,2]>25))
        cup = ((hsv[:,:,0]>=15)&(hsv[:,:,0]<=45)&(hsv[:,:,1]>85)&(hsv[:,:,2]>30))
        occluder = cup | (hsv[:,:,2]<85)
        return hsv, blue, occluder

    def _table_points(self, pixels):
        raw=self.native-1-((np.asarray(pixels)+.5)/self.scale-.5)
        rays=np.c_[raw,np.ones(len(raw))]@np.linalg.inv(self.K).T@self.T[:3,:3].T
        depth=(self.profile['table_top_z_m']-self.T[2,3])/rays[:,2]
        points=self.T[:3,3]+rays*depth[:,None]
        if np.any(depth<=0) or not np.isfinite(points).all():
            raise ValueError('Invalid calibrated table intersection')
        return points[:,:2]

    def _blue_points(self, blue):
        y,x=np.nonzero(blue)
        if len(x)<12 or min(x)==0 or max(x)==self.width-1 or min(y)==0 or max(y)==self.height-1:
            raise ValueError('No unclipped current card geometry')
        return self._table_points(np.c_[x,y])

    def register(self, frames):
        if len(frames)<3:
            raise ValueError('Three agreeing registration frames required')
        quads=[]
        for rgb in frames:
            _,blue,_=self._masks(rgb)
            count,_,stats,_=cv2.connectedComponentsWithStats(blue.astype('uint8'))
            if sum(s[cv2.CC_STAT_AREA]>=12 for s in stats[1:])!=1:
                raise ValueError('One full card required for registration')
            points=self._blue_points(blue)
            rect=cv2.minAreaRect(points.astype('float32'))
            if np.any(np.abs(np.asarray(rect[1])-.1)>.006):
                raise ValueError('Registered card must span nominal100mm square')
            quad=cv2.boxPoints(rect).astype(float)
            quads.append(quad)
        centres=np.array([q.mean(0) for q in quads])
        if np.max(np.linalg.norm(centres-centres[0],axis=1))>.002:
            raise ValueError('Card moved during registration')
        self.quad=quads[-1]
        self.centre=self.quad.mean(0)
        self.ref_rgb=frames[-1].copy()
        self.ref_gray=cv2.cvtColor(self.ref_rgb,cv2.COLOR_RGB2GRAY)
        hsv,blue,occluder=self._masks(self.ref_rgb)
        mask=((hsv[:,:,1]<65)&~occluder&~blue).astype('uint8')*255
        mask[:,int(.36*self.width):int(.6*self.width)]=0
        self.background=cv2.goodFeaturesToTrack(self.ref_gray,80,.01,5,mask=mask)
        if self.background is None or len(self.background)<4:
            raise ValueError('Insufficient independent background evidence')
        self.registered=True
        self.last_frame=-1
        self.last_stamp=-np.inf

    def _background_check(self,rgb):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        q,ok,_=cv2.calcOpticalFlowPyrLK(self.ref_gray,gray,self.background,None)
        back,ok2,_=cv2.calcOpticalFlowPyrLK(gray,self.ref_gray,q,None)
        hsv,_,_=self._masks(rgb)
        xy=np.rint(q.reshape(-1,2)).astype(int)
        inside=(xy[:,0]>=0)&(xy[:,0]<self.width)&(xy[:,1]>=0)&(xy[:,1]<self.height)
        keep=inside&ok.ravel().astype(bool)&ok2.ravel().astype(bool)
        ix=np.flatnonzero(inside)
        keep[ix]&=(hsv[xy[ix,1],xy[ix,0],1]<65)&(hsv[xy[ix,1],xy[ix,0],2]>75)
        keep&=np.linalg.norm(back-self.background,axis=2).ravel()<.7
        if keep.sum()<4:
            raise ValueError('Background occluded or tracking degenerate')
        shifts=np.linalg.norm(q-self.background,axis=2).ravel()[keep]
        if np.percentile(shifts,90)>1.2:
            raise ValueError('Camera/background moved')
        return float(np.percentile(shifts,90)),int(keep.sum())

    def measure(self,rgb,stamp,frame_id):
        if not self.registered:
            raise ValueError('Card registration required')
        if (not np.isfinite(stamp) or stamp<=self.last_stamp or frame_id<=self.last_frame):
            raise ValueError('New increasing frame and exposure required')
        self._masks(rgb)
        background_shift,count=self._background_check(rgb)
        _,blue,occluder=self._masks(rgb)
        points=self._blue_points(blue)
        # Work with EXTERNAL visible card boundaries, not arbitrary interior
        # blue pixels or the edge caused by a yellow/dark occluder.
        interior=cv2.erode(blue.astype('uint8'),np.ones((3,3),np.uint8))
        hidden=cv2.dilate(occluder.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
        boundary=blue&~interior.astype(bool)&~hidden
        y,x=np.nonzero(boundary)
        if len(x)<8:
            raise ValueError('No visible external card edges')
        border=self._table_points(np.c_[x,y])
        normals=[];offsets=[];supported=[]
        for i in range(4):
            a,b=self.quad[i],self.quad[(i+1)%4]
            u=(b-a)/np.linalg.norm(b-a)
            n=np.array([-u[1],u[0]])
            distance=(points-a)@n
            # minAreaRect gives a cyclic convex quad, interior is to its left.
            if distance.min()<-.006:
                raise ValueError('Card moved outside registered envelope or extra blue object')
            along=(border-a)@u
            group=border[(abs((border-a)@n)<.004)&(along>=-.004)&(along<=.104)]
            if len(group)<6 or np.ptp((group-a)@u)<.024:
                continue
            vx,vy,_,_=cv2.fitLine(group.astype('float32'),cv2.DIST_L2,0,.01,.01).ravel()
            direction=np.array([vx,vy])
            if abs(direction@u)<np.cos(np.deg2rad(2.)):
                # An occluder can contaminate ONE candidate edge. It supplies
                # no pose evidence; the remaining current edges must still
                # independently constrain both translation axes and a corner.
                continue
            normals.append(n);offsets.append(float(np.median((group-a)@n)));supported.append(i)
        A=np.asarray(normals)
        if len(A)<2 or np.linalg.matrix_rank(A,tol=.1)<2:
            raise ValueError('At least two nonparallel current card edges required')
        translation=np.linalg.lstsq(A,np.asarray(offsets),rcond=None)[0]
        if np.linalg.norm(translation)>.004:
            raise ValueError('Card moved from registered target')
        visible=sum(np.min(np.linalg.norm(border-(q+translation),axis=1))<.004 for q in self.quad)
        if not visible:
            raise ValueError('Current identifiable card corner required')
        self.last_frame,self.last_stamp=int(frame_id),float(stamp)
        return dict(target_base_m=[*(self.centre+translation),self.profile['table_top_z_m']],
                    frame_id=int(frame_id),current_exposure_s=float(stamp),
                    source='registered_card_constrained_by_current_geometry',
                    current_supported_edges=len(supported),current_visible_corners=int(visible),
                    registration_nominal_side_m=.1,background_shift_p90_px=background_shift,
                    background_features=count,real_geometry_qualified=False,
                    metric_error_bound_m=None,valid=True)


class UnifiedObserver:
    def __init__(self,profile,image_size,*,tcp_from_q):
        self.card=RegisteredCardObserver(profile,image_size)
        self.cylinder=NominalImageEstimator(profile,image_size)
        self.tcp_from_q=tcp_from_q

    def register(self,frames):
        self.card.register(frames)

    def measure(self,rgb,stamp,frame_id,state8):
        from contract import physical_state
        state=physical_state(state8)
        card=self.card.measure(rgb,stamp,frame_id)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        mask=current_cup_component((hsv[:,:,0]>=15)&(hsv[:,:,0]<=45)&
                 (hsv[:,:,1]>85)&(hsv[:,:,2]>30),self.cylinder)
        y,x=np.nonzero(mask)
        if min(x)==0 or max(x)==self.cylinder.width-1 or min(y)==0 or max(y)==self.cylinder.height-1:
            raise ValueError('Cup clipped by image boundary')
        bounds=np.array([min(x)-.5,min(y)-.5,max(x)+.5,max(y)+.5])
        initial=self.cylinder.ray_at_table([(min(x)+max(x))/2,max(y)])
        initial[2]=self.cylinder.profile['table_top_z_m']
        def residual(bottom):
            pixels=self.cylinder.project(self.cylinder.surface+bottom)
            return np.r_[pixels.min(0),pixels.max(0)]-bounds
        fit=least_squares(residual,np.clip(initial,[-.9,-.6,-.03],[-.15,.4,.42]),
                          bounds=([-.9,-.6,-.03],[-.15,.4,.42]),max_nfev=30,diff_step=1e-4)
        if not fit.success or np.sqrt(np.mean(fit.fun**2))>2.5:
            raise ValueError('Current cup silhouette inconsistent with nominal cylinder')
        tcp=np.asarray(self.tcp_from_q(state[:7]),float)
        if tcp.shape!=(3,) or not np.isfinite(tcp).all():
            raise ValueError('Finite joint-derived TCP required')
        return dict(relative_m=np.r_[fit.x-tcp,np.asarray(card['target_base_m'])-tcp],
                    cup_bottom_base_m=fit.x,card=card,frame_id=frame_id,
                    source='current_RGB_cup_and_current_constrained_card',
                    truth_used_as_input=False,real_geometry_qualified=False)
