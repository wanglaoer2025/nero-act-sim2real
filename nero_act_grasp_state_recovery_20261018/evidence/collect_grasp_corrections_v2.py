"""Offline expert labels at ACT-visited states; no runtime teacher takeover."""
import copy, hashlib, json, time
from pathlib import Path
import mujoco
from simulation.scene import build_scene
from simulation.collect import collect_episode, verify_replay
from simulation.refresh_observations import derive_episode, require_fresh_inputs
from training.recoveries import validate_corrections

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'evaluation/run19_grasp_sources'
RAW=ROOT/'training_data/grasp_raw_v2'
OUT=ROOT/'training_data/grasp_fresh_v2'
CASES=[('validation42_grasp','val')]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    RAW.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False)
    prior=json.loads((ROOT/'training_data/grasp_fresh_v1/manifest.failed.json').read_text())
    assert len(prior['episodes'])==2 and all(r['split']=='train' and r['teacher_success'] and r['replay']['passed'] for r in prior['episodes'])
    for row in prior['episodes']:
        assert sha(Path(row['path']))==row['sha256']
    manifest=dict(complete=False,episodes=prior['episodes'],rejected=[],
        previous_rejected_batch=str(ROOT/'training_data/grasp_fresh_v1/manifest.failed.json'),
        excluded_sources=prior['rejected'],source_evaluation=str(SOURCE),
        source_protocol_sha256=sha(SOURCE/'protocol.json'),
        source_checkpoint_sha256=json.loads((SOURCE/'protocol.json').read_text())['checkpoint_sha256'],
        original_16_cases_now_development_only=True,role='partial_descent_alignment_recoveries')
    for case,split in CASES:
        origin=SOURCE/case
        original=json.loads((origin/'scene_configuration.json').read_text())
        record=json.loads((origin/'physical_snapshots.json').read_text())
        result_source=json.loads((origin/'result.json').read_text())
        first_grasp=min([e['physics_step'] for e in result_source['simulated_events'] if e['kind']=='GRASP'] or [float('inf')])
        model=build_scene(original);data=mujoco.MjData(model)
        site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
        heights=[]
        for snap in record['snapshots']:
            data.qpos[:]=snap['qpos'];mujoco.mj_forward(model,data)
            heights.append(float(data.site_xpos[site][2]))
        eligible=[i for i,s in enumerate(record['snapshots'])
            if record['query_steps'][i]<first_grasp and s['qpos'][7]-s['qpos'][8]>.09
            and sum((s['qpos'][9+k]-original['pick_xy_m'][k])**2 for k in [0,1])<.005**2]
        if not eligible:raise ValueError('No original pre-grasp open source states')
        chosen=set()
        for height in [.060,.055]:
            index=3001+len(manifest['episodes']);started=time.monotonic()
            try:
                j=min(eligible,key=lambda i:abs(heights[i]-height))
                if abs(heights[j]-height)>.0025:raise ValueError('No matching ACT-visited correction height')
                if j in chosen:raise ValueError('Two requested heights selected the same source state')
                chosen.add(j)
                snap=record['snapshots'][j];config=copy.deepcopy(original)
                config.update(continuation='descend_to_retreat',initial_simulation_snapshot=snap,
                    initial_robot_state=dict(qpos_rad=snap['qpos'][:7],gripper_total_width_m=snap['qpos'][7]-snap['qpos'][8]),
                    joint_speed_limit_rad_s=.02,teacher_command_lead_limit_rad=.015,
                    gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
                    correction_source=dict(origin=str(origin),query_step=record['query_steps'][j],split=split,
                        source_group='run19_low_grasp_'+case,role='partial_not_full',
                        reason='recover ACT-visited pose drift before cup contact; restore original grasp rotation'))
                config['dwell_seconds'].update(close=8,release=8)
                path=RAW/f'episode_{index}.hdf5'
                path.with_suffix('.scene.json').write_text(json.dumps(config,indent=2))
                result=collect_episode(config,path);replay=verify_replay(path)
                assert result['success'] and not result['full_task_verified'] and replay['passed']
                assert result['physical_success'] and result['max_lift_m']>.06 and result['maximum_horizontal_transfer_m']>.1
                row=dict(id=index,path=str(path),sha256=sha(path),status='valid',split=split,
                    source_group=config['correction_source']['source_group'],teacher_success=True,
                    full_task_verified=False,correction_verified=True,task='descend_to_retreat_correction',
                    origin=str(origin),query_step=record['query_steps'][j],snapshot_sha256=sha(origin/'physical_snapshots.json'),
                    scene_sha256=sha(origin/'scene_configuration.json'),replay=replay,
                    prefix_contract_audit=result['prefix_contract_audit'])
                derived=derive_episode(path,OUT/path.name,expected_source_sha256=row['sha256'])
                row.update(path=derived['target'],sha256=derived['target_sha256'],observation_derivation=derived)
                manifest['episodes'].append(row)
                print(json.dumps(dict(id=index,case=case,split=split,start_height=heights[j],success=True,
                    lift_m=result['max_lift_m'],transfer_m=result['maximum_horizontal_transfer_m'],
                    replay_error=replay['maximum_qpos_error'],elapsed_s=time.monotonic()-started)),flush=True)
            except Exception as exc:
                manifest['rejected'].append(dict(case=case,height=height,error=str(exc)))
                (OUT/'manifest.failed.json').write_text(json.dumps(manifest,indent=2));raise
    manifest['complete']=len(manifest['episodes'])==4 and not manifest['rejected']
    assert manifest['complete']
    validate_corrections(manifest,[])
    require_fresh_inputs([Path(r['path']) for r in manifest['episodes']])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(collection_complete=True,accepted=4,train=2,val=2,no_hardware=True)),flush=True)

if __name__=='__main__':main()
