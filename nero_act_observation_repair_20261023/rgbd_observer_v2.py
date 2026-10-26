"""Review fix of sealed rgbd_observer.py: absolute ray-depth residual gate."""
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'nero_act_contact_escape_fix_20261023'))
from card_edges import CardEdgesEstimator
from rgbd_geometry import world_points_from_depth


class RgbdCupEstimator(CardEdgesEstimator):
    def measure_camera_xyz(self,rgb,xyz_camera,confidence):
        xyz=np.asarray(xyz_camera,float)
        if xyz.shape!=(self.height,self.width,3):
            raise ValueError('Native pixel-paired SDK optical XYZ required')
        # Arrays are already indexed upright; SDK vector axes remain optical.
        world=xyz@self.transform[:3,:3].T+self.transform[:3,3]
        return self._measure_world(rgb,world,confidence,np.isfinite(xyz).all(2)&(xyz[:,:,2]>.05)&(xyz[:,:,2]<3))

    def component(self,mask,minimum):
        if minimum!=15:
            return super().component(mask,minimum)
        count,labels,stats,centres=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        accepted=[]
        for i in range(1,count):
            x,y,w,h,area=stats[i]
            if area>=500 and x>0 and y>0 and x+w<self.width and y+h<self.height:
                accepted.append(i)
        if len(accepted)!=1:
            raise ValueError('Unique complete current yellow cup component required')
        return labels==accepted[0],centres[accepted[0]]

    def measure_depth(self,rgb,depth_m,confidence):
        depth=np.asarray(depth_m,float)
        if depth.shape!=(self.height,self.width):
            raise ValueError('Pixel-paired metric camera-Z depth required')
        world=world_points_from_depth(depth,self.profile['camera'])
        return self._measure_world(rgb,world,confidence,np.isfinite(depth)&(depth>.05)&(depth<3))

    def _measure_world(self,rgb,world,confidence,valid_depth):
        if rgb.dtype!=np.uint8 or rgb.shape!=(self.height,self.width,3):
            raise ValueError('Current native RGB required')
        confidence=np.asarray(confidence,float)
        if confidence.shape!=(self.height,self.width):
            raise ValueError('Pixel-paired SDK confidence required')
        self.current_hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        hsv=self.current_hsv
        cup,_=self.component((hsv[:,:,0]>=15)&(hsv[:,:,0]<=45)&(hsv[:,:,1]>85)&(hsv[:,:,2]>30),15)
        y,x=np.nonzero(cup)
        if min(x)<=0 or max(x)>=self.width-1 or min(y)<=0 or max(y)>=self.height-1:
            raise ValueError('Cup clipped by image boundary')
        card,_=self.component((hsv[:,:,0]>=95)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>90)&(hsv[:,:,2]>25),12)
        cy,cx=np.nonzero(card)
        targets=np.array([self.ray_at_table(pixel) for pixel in np.c_[cx,cy]])
        target=(targets.min(0)+targets.max(0))/2
        interior=cv2.erode(cup.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        valid=(interior&valid_depth&np.isfinite(confidence)&(confidence>=0)&(confidence<=50)
            &np.isfinite(world).all(2))
        cloud=world[valid]
        if len(cloud)<100:
            raise ValueError('Insufficient current high-confidence cup depth; no RGB fallback')
        cloud=cloud[np.linspace(0,len(cloud)-1,min(512,len(cloud))).astype(int)]
        background=(~cup)&(hsv[:,:,2]>=85)
        edges=[]
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            adjacent=np.roll(background,(-dy,-dx),axis=(0,1))
            ey,ex=np.nonzero(cup&adjacent)
            edges.extend(np.c_[ex+dx*.5,ey+dy*.5].tolist())
        edges=np.asarray(edges,float)
        if len(edges)<12 or min(np.ptp(edges,axis=0))<8:
            raise ValueError('Visible cup outline lacks geometric support')
        edges=edges[np.linspace(0,len(edges)-1,min(384,len(edges))).astype(int)]
        radius=self.radius;inner=radius-float(self.profile['cup']['wall_thickness_m'])
        bottom_thickness=float(self.profile['cup'].get('bottom_thickness_m',.002))
        height=self.cup_height

        def distances(p):
            rotation=Rotation.from_rotvec([p[3],p[4],0]).as_matrix()
            local=(cloud-p[:3])@rotation
            r=np.linalg.norm(local[:,:2],axis=1);z=local[:,2]
            # Finite exterior/interior walls, rim annulus and the two base faces.
            surfaces=np.array([
                np.hypot(r-radius,z-np.clip(z,0,height)),
                np.hypot(r-inner,z-np.clip(z,bottom_thickness,height)),
                np.hypot(z-height,r-np.clip(r,inner,radius)),
                np.hypot(z-bottom_thickness,np.maximum(0,r-inner)),
                np.hypot(z,np.maximum(0,r-radius))])
            # Nearest surface alone aliases the visible outer wall with the hidden
            # inner wall. Enforce the first visible intersection along each camera ray.
            origin=(self.transform[:3,3]-p[:3])@rotation
            directions=local-origin
            observed_length=np.linalg.norm(directions,axis=1)
            directions=directions/observed_length[:,None]
            a=np.sum(directions[:,:2]**2,axis=1)
            b=2*np.sum(origin[:2]*directions[:,:2],axis=1)
            hits=[]
            for wall,zmin in [(radius,0),(inner,bottom_thickness)]:
                c=np.sum(origin[:2]**2)-wall**2
                discriminant=b*b-4*a*c
                for sign in [-1,1]:
                    t=(-b+sign*np.sqrt(np.maximum(0,discriminant)))/(2*np.maximum(a,1e-12))
                    hit_z=origin[2]+t*directions[:,2]
                    hits.append(np.where((discriminant>=0)&(t>0)&(hit_z>=zmin)&(hit_z<=height),t,np.inf))
            for plane,rmin,rmax in [(0,0,radius),(bottom_thickness,0,inner),(height,inner,radius)]:
                dz=directions[:,2]
                t=(plane-origin[2])/np.where(abs(dz)>1e-12,dz,1e-12)
                hit_r=np.linalg.norm(origin[:2]+t[:,None]*directions[:,:2],axis=1)
                hits.append(np.where((abs(dz)>1e-12)&(t>0)&(hit_r>=rmin)&(hit_r<=rmax),t,np.inf))
            first=np.min(hits,axis=0)
            depth_residual=np.where(np.isfinite(first),first-observed_length,surfaces.min(0))
            hull=cv2.convexHull(self.project(self.surface@rotation.T+p[:3]).astype(np.float32)).reshape(-1,2).astype(float)
            delta=np.roll(hull,-1,axis=0)-hull
            offset=edges[:,None,:]-hull[None]
            ratio=np.clip(np.sum(offset*delta[None],axis=2)/np.maximum(np.sum(delta**2,axis=1),1e-12)[None],0,1)
            image_residual=np.sqrt(np.min(np.sum((offset-ratio[:,:,None]*delta[None])**2,axis=2),axis=1)+1e-12)
            return depth_residual,image_residual

        def residual(p):
            depth,image=distances(p)
            return np.r_[depth/.001,image/.5]

        initial=self.ray_at_table([(min(x)+max(x))/2,max(y)])
        initial[2]=float(np.min(cloud[:,2]))-bottom_thickness
        # A current point-cloud circle gives an independent XY seed; no table-Z constraint.
        circle=np.linalg.lstsq(np.c_[2*cloud[:,:2],np.ones(len(cloud))],np.sum(cloud[:,:2]**2,axis=1),rcond=None)[0]
        seeds=[np.r_[initial,0.,0.],np.r_[circle[:2],initial[2],0.,0.]]
        low=np.array([-.9,-.6,-.03,-.2,-.2]);high=np.array([-.15,.4,.42,.2,.2])
        fits=[least_squares(residual,np.clip(seed,low+1e-8,high-1e-8),bounds=(low,high),
            loss='soft_l1',f_scale=1.,diff_step=1e-3,max_nfev=120) for seed in seeds]
        fit=min(fits,key=lambda f:f.cost)
        singular=np.linalg.svd(fit.jac,compute_uv=False)
        if not fit.success or singular[-1]<1e-5:
            raise ValueError('Current RGB-D cup pose is underconstrained')
        sigma=np.sqrt(np.diag(np.linalg.inv(fit.jac.T@fit.jac)))
        depth_error,image_error=distances(fit.x)
        dp95=float(np.percentile(abs(depth_error),95));ip95=float(np.percentile(image_error,95))
        if dp95>.002 or ip95>1.5 or max(sigma[:3])>.002 or np.any(abs(fit.x[3:])>.199):
            raise ValueError(f'Nominal RGB-D fit rejected: depth_p95={dp95:.6f}m outline_p95={ip95:.3f}px')
        return fit.x[:3],target,dict(self.card_diagnostic,
            method='current_RGB_depth_hollow_cylinder',depth_points=len(cloud),visible_boundary_points=len(edges),
            depth_surface_p95_m=dp95,depth_surface_residual_kind='absolute_ray_length',visible_boundary_p95_px=ip95,
            nominal_linearized_std=sigma.tolist(),tilt_rotvec_xy_rad=fit.x[3:].tolist(),
            truth_used_as_observer_input=False,runtime_qualified=False,error_bound_m=None,
            filled_depth_allowed=False,hardware_deployment_allowed=False)
