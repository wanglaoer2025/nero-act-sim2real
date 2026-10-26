"""Dense replay rendering of an existing PHYSICAL TEACHER, not an ACT success test."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import h5py
import numpy as np


def summarize_visibility(expected,rows):
    bad=[r for r in rows if not r['valid']]
    ids=[r['frame'] for r in rows]
    return dict(expected_frames=expected,audited_frames=len(rows),
                rejected_frames=len(bad),failure_reasons=dict(Counter(r['reason'] for r in bad)),
                stage_frames=dict(Counter(r['stage'] for r in rows)),
                stage_rejected=dict(Counter(r['stage'] for r in bad)),
                complete_visibility=(len(rows)==expected and ids==list(range(expected)) and not bad))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    import mujoco
    from observer import UnifiedObserver
    sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
    from simulation.scene import build_scene
    started=time.monotonic()
    with h5py.File(args.source,'r') as f:
        scene=json.loads(f.attrs['scene_config'])
        m=build_scene(scene);d=mujoco.MjData(m);fk=mujoco.MjData(m)
        m.vis.global_.offwidth=args.width;m.vis.global_.offheight=args.height
        site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
        def tcp(q):
            fk.qpos[:7]=q;mujoco.mj_forward(m,fk)
            return fk.site_xpos[site].copy()
        profile=dict(camera=scene['camera_calibration'],table_top_z_m=scene['table_top_z_m'],cup=scene['cup'])
        observer=UnifiedObserver(profile,(args.height,args.width),tcp_from_q=tcp)
        stages=f['simulation/stage'].asstr()[:]
        q=f['simulation/qpos'][:];v=f['simulation/qvel'][:];stamps=f['timestamp'][:]
        states=np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]]
        cup_truth=f['observations/object_xyz'][:]
        def render(t,renderer):
            d.qpos[:]=q[t];d.qvel[:]=v[t];d.time=float(stamps[t]);mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera='calibrated_nero_camera')
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            return renderer.render().copy()
        rows=[];errors=[];examples=Counter()
        with mujoco.Renderer(m,height=args.height,width=args.width) as renderer:
            first=[render(t,renderer) for t in range(3)]
            observer.register(first)
            for t in range(len(stamps)):
                rgb=render(t,renderer)
                row=dict(frame=t,stage=str(stages[t]),exposure_s=float(stamps[t]))
                try:
                    observed=observer.measure(rgb,float(stamps[t]),t,states[t])
                    row.update(valid=True,card=observed['card'])
                    # Diagnostic ONLY, never observer input or data gate.
                    truth=cup_truth[t]-[0,0,scene['cup']['height_m']/2]
                    error=float(np.linalg.norm(observed['cup_bottom_base_m']-truth))
                    errors.append(error)
                except ValueError as exc:
                    row.update(valid=False,reason=str(exc))
                    examples[str(exc)]+=1
                    if examples[str(exc)]<=1:
                        import cv2
                        cv2.imwrite(str(args.output/f'rejected_{t}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                rows.append(row)
                if t%100==0:
                    progress=dict(frame=t,total=len(stamps),rejected=sum(not r['valid'] for r in rows),elapsed_s=time.monotonic()-started)
                    (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                    print(json.dumps(progress),flush=True)
    result=summarize_visibility(len(stamps),rows)
    result.update(source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  dense_rerender=True,image_size_hw=[args.height,args.width],
                  policy_training_started=False,ACT_closed_loop_tested=False,hardware_commands_sent=False,
                  runtime_geometry_qualified=False,elapsed_s=time.monotonic()-started,
                  cup_truth_diagnostic_m={'median':float(np.median(errors)),'p95':float(np.percentile(errors,95)),
                                          'max':float(max(errors))} if errors else None)
    (args.output/'frames.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
