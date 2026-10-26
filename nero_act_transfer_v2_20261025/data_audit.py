"""Read-only schema/numeric audit. Declared teacher success is not re-simulated."""
import argparse
from collections import Counter
import json
from pathlib import Path
import h5py
import numpy as np

STAGES={'approach','descend','close','lift','carry','place','release','retreat'}
SCHEMA='nero_act_transfer_v2_15d'


def audit_episode(path):
    path=Path(path)
    reasons=[]
    result=dict(path=str(path),main15_ready=False,physical_labels_reusable=False,
                errors=reasons,physical_success_independently_rechecked=False)
    try:
        with h5py.File(path,'r') as f:
            synthetic=bool(f.attrs.get('synthetic',False) or f.attrs.get('sim',False))
            result['synthetic']=synthetic
            n=len(f['action']);result['frames']=n
            a=f['action'][:]
            q=f['observations/qpos'][:]
            w=f['observations/gripper_state'][:].reshape(-1)
            t=f['timestamp'][:]
            if (a.shape!=(n,8) or q.shape!=(n,7) or len(w)!=n
                    or not np.isfinite(np.r_[a.ravel(),q.ravel(),w,t]).all()
                    or np.any(np.diff(t)<=0)):
                reasons.append('invalid_numeric_shape_values_or_time')
            if np.any((w<0)|(w>.1)) or np.any((a[:,7]<0)|(a[:,7]>.0950001)):
                reasons.append('gripper_feedback_over100mm_or_command_over95mm')
            semantics=str(f.attrs.get('action_semantics',''))
            result['action_semantics']=semantics
            proxy=('future_measured' in semantics or 'feedback' in semantics
                   or not synthetic and 'submitted' not in semantics)
            if proxy:reasons.append('proxy_or_unproven_submitted_actions')
            complete=bool(f.attrs.get('full_task_verified',False) and f.attrs.get('teacher_success',False))
            stage_values=[]
            if 'simulation/stage' in f:
                stage_values=f['simulation/stage'].asstr()[:].tolist()
                result['stage_counts']=dict(Counter(stage_values))
                complete=complete and STAGES.issubset(stage_values)
            elif synthetic:
                complete=False
            else:
                complete=bool(f.attrs.get('full_task_verified',False))
            result['declared_complete_task']=complete
            if not complete:reasons.append('missing_complete_task_evidence')
            result['physical_labels_reusable']=synthetic and complete and not reasons
            required=['observations/visual_relative_m','observations/visual_valid',
                      'observations/frame_id','observations/camera_monotonic_s',
                      'observations/feedback_monotonic_s','action_time_monotonic_s']
            missing=[k for k in required if k not in f]
            result['missing_fields']=missing
            if missing:reasons.append('missing_visual_or_command_clock_fields')
            if f.attrs.get('observation_schema','')!=SCHEMA:
                reasons.append('observer_not_shared_v2_schema')
            if ('action_acknowledged' not in f or f['action_acknowledged'].shape!=(n,)
                    or f['action_acknowledged'].dtype.kind!='b' or not f['action_acknowledged'][:].all()):
                reasons.append('submission_ACK_unavailable')
            if not synthetic:
                if not f.attrs.get('exposure_clock_verified',False):
                    reasons.append('real_exposure_clock_unverified')
                if 'calibration_sha256' not in f.attrs:
                    reasons.append('real_calibration_binding_unavailable')
            if not missing:
                times=f['action_time_monotonic_s'][:]
                exposed=f['observations/camera_monotonic_s'][:]
                feedback=f['observations/feedback_monotonic_s'][:]
                ids=f['observations/frame_id'][:]
                if (any(x.shape!=(n,) for x in [times,exposed,feedback,ids])
                        or not np.isfinite(np.r_[times,exposed,feedback]).all()
                        or np.any(np.diff(times)<=0) or np.any(np.diff(exposed)<=0)
                        or np.any(np.diff(ids)<=0) or np.any(times<exposed)
                        or np.max(abs(feedback-exposed))>.05):
                    reasons.append('invalid_causal_or_paired_times')
                valid=f['observations/visual_valid'][:]
                geo=f['observations/visual_relative_m'][:]
                result['visual_valid_frames']=int(valid.sum())
                if valid.shape!=(n,) or geo.shape!=(n,6) or not valid.all() or not np.isfinite(geo).all():
                    reasons.append('not_complete_same_observer_visibility')
                if 'initial_previous_gripper_target_m' not in f.attrs:
                    reasons.append('missing_initial_submitted_context')
                elif np.all(np.diff(times)>0):
                    idx=np.searchsorted(times,exposed,side='left')-1
                    initial=float(f.attrs['initial_previous_gripper_target_m'])
                    prev=np.where(idx>=0,a[np.maximum(idx,0),7],initial)
                    delta=a[:2,7]-prev[0]
                    result['first_query_gripper_increment_mm']=(delta*1000).tolist()
                    if not 0<=initial<=.0950001:
                        reasons.append('invalid_initial_submitted_context')
                    explicit=(f.attrs.get('initial_previous_acknowledged',False)
                        and bool(f.attrs.get('initial_previous_source',''))
                        and np.isfinite(f.attrs.get('initial_previous_time_monotonic_s',np.nan))
                        and f.attrs['initial_previous_time_monotonic_s']<exposed[0])
                    # Existing new-collector episodes have an auditable simulated
                    # servo initialization, not a fabricated physical ACK.
                    initialization=False
                    if synthetic and 'scene_config' in f.attrs:
                        scene=json.loads(f.attrs['scene_config'])
                        initialization=(f.attrs.get('clock_domain','')=='synthetic_20Hz_exposure_before_command'
                            and not scene.get('continuation') and not scene.get('initial_simulation_snapshot')
                            and exposed[0]==0 and abs(initial-w[0])<1e-7
                            and abs(initial-scene['initial_robot_state']['gripper_total_width_m'])<1e-7)
                    if not (explicit or initialization):
                        reasons.append('unproven_initial_submitted_context')
                    result['initial_context_provenance']=('explicit_ACK_before_exposure' if explicit else
                        'simulated_servo_initialization_at_t0_before_first_exposure' if initialization else None)
                    for point in range(2):
                        length=n-point
                        bad=np.flatnonzero(abs(a[point:,7]-prev[:length])>.0010001)
                        if len(bad):
                            reasons.append('gripper_query_increment_over_1mm')
                            result['invalid_prefix']=dict(query_frame=int(bad[0]),point=point,
                                increment_m=float(a[bad[0]+point,7]-prev[bad[0]]))
                            break
            image_names=list(f.get('observations/images',{}))
            if not image_names:reasons.append('missing_RGB')
            for name in image_names:
                ds=f['observations/images/'+name]
                if ds.is_virtual:
                    for src in ds.virtual_sources():
                        source=Path(src.file_name)
                        if not source.is_absolute():source=path.parent/source
                        if not source.exists():reasons.append('missing_VDS_image_source')
            if n and image_names:
                # Actual pixels are sampled; full-frame visibility is audited separately.
                first=f['observations/images/'+image_names[0]][0]
                result['first_RGB_nonzero']=bool(first.any())
            result['main15_ready']=not reasons
    except (OSError,KeyError,ValueError,IndexError) as exc:
        reasons.append(type(exc).__name__+': '+str(exc))
    return result


def existing_inventory(root):
    root=Path(root)
    record=json.loads((root/'nero_act_root_cause_audit_20261025/training_readonly_rechecks.json').read_text())
    paths=set()
    def walk(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k=='path' and isinstance(v,str) and v.endswith(('.hdf5','.h5')):paths.add(v)
                else:walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    walk(record['uniform_data'])
    # Include the newer complete/recovery manifests without duplicating raw images.
    fresh=json.loads((root/'nero_act_new_handeye_policy_repair_20261023/data/manifest.json').read_text())
    for row in fresh['episodes']:paths.add(row['path'])
    real=json.loads((root/'nero_act_real_data_adapter_20261024/data/manifest.json').read_text())
    sim_rows=[audit_episode(p) for p in sorted(paths)]
    real_rows=[audit_episode(r['source_path']) for r in real['episodes']]
    return dict(complete=True,simulation_episodes=len(sim_rows),real_episodes=len(real_rows),
                simulation_frames=sum(r.get('frames',0) for r in sim_rows),
                real_frames=sum(r.get('frames',0) for r in real_rows),
                simulation_main15_ready=sum(r['main15_ready'] for r in sim_rows),
                real_main15_ready=sum(r['main15_ready'] for r in real_rows),
                simulation_physical_labels_reusable=sum(r['physical_labels_reusable'] for r in sim_rows),
                simulation=sim_rows,real=real_rows,hardware_commands_sent=False,
                scope='Numeric/schema audit; sparse RGB and old declared outcomes are not new physical acceptance')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=existing_inventory(Path(__file__).resolve().parent.parent)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ['simulation','real']}))


if __name__=='__main__':main()
