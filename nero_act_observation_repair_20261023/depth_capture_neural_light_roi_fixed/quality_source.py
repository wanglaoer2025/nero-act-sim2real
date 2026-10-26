"""Read-only surface depth diagnostics; no object-centre or calibration claim."""
import numpy as np
import cv2


def cup_colour_roi(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=((hsv[:,:,0]>=15)&(hsv[:,:,0]<=45)&(hsv[:,:,1]>85)&(hsv[:,:,2]>30)).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    accepted=[];clipped=0
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        if area<500:continue
        if x<=0 or y<=0 or x+w>=mask.shape[1] or y+h>=mask.shape[0]:
            clipped+=1;continue
        accepted.append(i)
    report=dict(valid=len(accepted)==1,components=len(accepted),clipped_components_rejected=clipped,
        method='unique_complete_current_yellow_component_not_semantic_identity')
    if len(accepted)!=1:return np.zeros(mask.shape,bool),report
    selected=cv2.erode((labels==accepted[0]).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    return selected,report


def upright_arrays(rgb,xyz,confidence):
    rgb,xyz,confidence=[np.asarray(x) for x in [rgb,xyz,confidence]]
    if rgb.ndim!=3 or rgb.shape[-1]!=3 or xyz.shape!=rgb.shape or confidence.shape!=rgb.shape[:2]:
        raise ValueError('Pixel-paired RGB/XYZ/confidence arrays required')
    # Rotate the pixel indexing only;XYZ vectors retain the SDK optical axes.
    return tuple(x[::-1,::-1].copy() for x in [rgb,xyz,confidence])


def valid_points(mask,xyz,confidence):
    mask,xyz,confidence=[np.asarray(x) for x in [mask,xyz,confidence]]
    if mask.ndim!=2 or xyz.shape!=(*mask.shape,3) or confidence.shape!=mask.shape:
        raise ValueError('Pixel-paired surface mask and XYZ/confidence required')
    return (mask.astype(bool)&np.isfinite(xyz).all(2)&(xyz[:,:,2]>0)
        &np.isfinite(confidence)&(confidence>=0)&(confidence<=50))


def surface_quality(mask,xyz,confidence):
    valid=valid_points(mask,xyz,confidence)
    total=int(np.count_nonzero(mask));count=int(valid.sum())
    return dict(valid=count>0,roi_pixels=total,valid_points=count,
        valid_fraction=count/max(1,total),confidence_maximum=50,
        median_surface_camera_xyz_m=np.median(xyz[valid],axis=0).tolist() if count else None,
        measurement_kind='visible_surface_points_not_object_centre',
        coordinate_frame='ZED_IMAGE_left_optical_metres',absolute_accuracy_verified=False,
        hardware_deployment_allowed=False)


def sequence_quality(samples):
    if len(samples)<2:
        raise ValueError('At least two independent captures required for repeatability')
    reports=[surface_quality(x['mask'],x['xyz'],x['confidence']) for x in samples]
    centres=[]
    for x in samples:
        y,u=np.nonzero(x['mask'])
        centres.append(np.array([u.mean(),y.mean()]) if len(u) else [np.nan,np.nan])
    centres=np.asarray(centres)
    span=float(np.linalg.norm(np.ptp(centres,axis=0))) if np.isfinite(centres).all() else None
    shared=np.logical_and.reduce([valid_points(x['mask'],x['xyz'],x['confidence']) for x in samples])
    static=span is not None and span<=2 and int(shared.sum())>=16
    jitter=None
    if static:
        depths=np.stack([x['xyz'][:,:,2][shared] for x in samples])
        jitter=float(np.percentile(np.std(depths,axis=0),95)*1000)
    return dict(samples=len(samples),frame_reports=reports,
        static_repeatability_valid=bool(static),roi_centre_span_px=span,
        shared_valid_pixels=int(shared.sum()),per_pixel_depth_std_p95_mm=jitter,
        absolute_accuracy_verified=False,independent_geometric_reference_present=False,
        camera_robot_clock_alignment_verified=False,hardware_deployment_allowed=False)
