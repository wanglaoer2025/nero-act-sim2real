"""Audit saved real observations and measurements; never connect or command hardware."""
import argparse
import json
from pathlib import Path
import numpy as np

NAMES=[f'joint{i}' for i in range(1,8)]+['gripper']


def audit(capture,measurements):
    blockers=[];details={};points={}
    try:
        report=json.loads((capture/'report.json').read_text())
        if not isinstance(report,dict):raise ValueError('capture report must be a JSON object')
        if report.get('mode')!='read_only_live_capture':blockers.append('capture is not declared live read-only capture')
        for key,expected in [('motion_sent',False),('robot_sdk_connected',False),('image_already_upright',True),
                             ('rgb_order','RGB'),('arm_unit','rad'),('gripper_unit','m total width')]:
            if report.get(key)!=expected:blockers.append(f'capture metadata mismatch: {key}')
        if report.get('failure'):blockers.append('capture reported a runtime failure')
        with np.load(capture/'observations.npz',allow_pickle=False) as data:
            names=data['joint_names'].tolist();q=data['state'];rgb=data['rgb'];times=data['sample_monotonic_s']
            if len(names)!=8 or len(set(names))!=8 or set(names)!=set(NAMES):raise ValueError('invalid joint names')
            if q.ndim!=2 or q.shape[1]!=8 or not np.isfinite(q).all():raise ValueError('invalid finite [T,8] state')
            if rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[-1]!=3 or len(rgb)!=len(q):raise ValueError('invalid uint8 RGB images')
            if len(q)<2 or times.shape!=(len(q),) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('invalid strictly increasing timestamps')
            if report.get('samples')!=len(q):blockers.append('capture report sample count mismatch')
            gripper=q[:,names.index('gripper')]
            if np.any(gripper<-.003) or np.any(gripper>.098):blockers.append('gripper feedback incompatible with 0-95 mm total opening')
            if 'joint_receipt_age_s' not in data:raise ValueError('joint receipt ages absent')
            ages=data['joint_receipt_age_s']
            if ages.shape!=(len(q),) or not np.isfinite(ages).all() or np.any(ages<0) or np.max(ages)>.2:
                blockers.append('missing or stale host-receipt feedback pairing')
            details={'samples':len(q),'image_shape':list(rgb.shape),'achieved_sample_hz':float(1/np.mean(np.diff(times))),
                     'maximum_host_feedback_age_s':float(np.max(ages)),'hardware_synchronized':False}
    except (OSError,ValueError,KeyError,TypeError) as error:
        blockers.append(f'observation file validation: {error}')
    if measurements.get('coordinate_frame')!='nero_base' or measurements.get('unit')!='mm':
        blockers.append('measurements require nero_base XYZ in mm')
    for key in ['source_xyz_mm','destination_xyz_mm']:
        try:point=np.asarray(measurements.get(key),dtype=float)
        except (ValueError,TypeError):point=np.array([])
        if point.shape!=(3,) or not np.isfinite(point).all():blockers.append(f'unmeasured or invalid {key}')
        else:points[key]=point.tolist()
    table=measurements.get('table_top_z_mm');height=measurements.get('cup_height_mm')
    if not isinstance(table,(int,float)) or isinstance(table,bool) or not np.isfinite(table):
        blockers.append('table height unmeasured')
    elif not isinstance(height,(int,float)) or isinstance(height,bool) or not np.isfinite(height) or height<=0:
        blockers.append('cup height invalid')
    else:
        for key,point in points.items():
            if abs(point[2]-(table+height/2))>5:blockers.append(f'{key} is inconsistent with table and geometric cup centre')
    return {'mode':'saved_observation_readiness_audit','ready_for_recorded_shadow':not blockers,
            'shadow_blockers':blockers,'capture_details':details,'task_coordinates_mm':points,
            'ready_for_real_motion':False,'motion_sent':False,
            'real_motion_pending':['independent closed-loop policy acceptance','real camera appearance/geometry alignment',
                                   'physical TCP verification','measured actuator response and single CAN owner'],
            'tcp_verified_on_robot':measurements.get('tcp_verified_on_robot') is True,
            'camera_independent_points_verified':measurements.get('camera_to_base_verified_on_independent_points') is True,
            'scope':'file audit only; passing permits recorded-input checking, never motion',
            'real_transfer_verified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--measurements',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=audit(args.capture,json.loads(args.measurements.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return int(not report['ready_for_recorded_shadow'])


if __name__=='__main__':raise SystemExit(main())
