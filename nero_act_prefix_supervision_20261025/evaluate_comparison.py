"""Tiered diagnostic and unchanged full-task physics acceptance."""
import copy
import json
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np
import torch
import h5py
from experiment import BASE,PrefixPolicy
from metrics import representative_dataset,deployment_metrics
from evaluate import development_cases,load_runner,full_task_passed,evaluate_development

ROOT=Path(__file__).resolve().parent


class BoundedPolicy:
    def __init__(self,inner,limit=300):
        self.inner=inner;self.limit=limit;self.queries=0

    def __getattr__(self,key):return getattr(self.inner,key)

    def __call__(self,*args,**kwargs):
        if self.queries>=self.limit:
            raise RuntimeError(f'Diagnostic{self.limit}query cap; not full-task acceptance')
        self.queries+=1
        return self.inner(*args,**kwargs)


def approach_metrics(scene,result,measurements):
    goal=np.r_[scene['pick_xy_m'],scene['table_top_z_m']+scene['grasp_tcp_height_m']+scene['lift_height_m']]
    tcp=np.asarray([r['tcp_xyz'] for r in measurements],float)
    distance=np.linalg.norm(tcp-goal,axis=1)
    error=result.get('error')
    diagnostic_cap=bool(error and error.startswith('RuntimeError: Diagnostic')
                        and error.endswith('query cap; not full-task acceptance'))
    safe=(result.get('execution_contract_failures',1)==0
          and result.get('unexpected_arm_cup_contact_frames',1)==0
          and (error is None or diagnostic_cap))
    reached=bool(distance.min()<=.01)
    return dict(initial_distance_m=float(distance[0]),minimum_distance_m=float(distance.min()),
        final_distance_m=float(distance[-1]),distance_reduction_m=float(distance[0]-distance[-1]),
        endpoint_tcp_m=tcp[-1].tolist(),tcp_height_range_m=float(np.ptp(tcp[:,2])),
        approach_reached=reached,eligible_for_full_nominal=bool(reached and safe),
        error=error,diagnostic_cap_reached=diagnostic_cap,simulated_events=result.get('simulated_events',[]),
        full_task_test=False,full_task_passed=False,hardware_commands_sent=False)


def load_policy(path):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    policy=PrefixPolicy(cp['stats'],variant=cp.get('variant','A'),pretrained=False).cuda().eval()
    policy.load_state_dict(cp['model'],strict=True);policy.config=dict(cp['config'])
    return policy,cp


def scene_cases(manifest):
    with h5py.File(manifest['source_scene']) as f:base=json.loads(f.attrs['scene_config'])
    cases=[]
    for row in development_cases()[:3]:
        scene=copy.deepcopy(base)
        scene['pick_xy_m']=(np.asarray(base['pick_xy_m'])+np.asarray(row['offset_mm'])/1000).tolist()
        rng=np.random.default_rng(row['seed'])
        scene['initial_robot_state']['qpos_rad']=(np.asarray(base['initial_robot_state']['qpos_rad'])+
                                                rng.uniform(-.0002,.0002,7)).tolist()
        scenario=dict(seed=row['seed'],query_stride=2,servo_delay_steps=0,
                      observation_age_s=.05,jitter=False,force=False)
        cases.append((row['id'],scene,scenario))
    return cases


def evaluate(binding):
    from run import save_json,verify_bindings
    verify_bindings(binding)
    out=ROOT/'evaluation';out.mkdir(exist_ok=False)
    manifest=json.loads((BASE/'data/v2_contour_01/manifest.json').read_text())
    runner=load_runner();cases=scene_cases(manifest)
    records={}
    for name,path in [('source5K',BASE/'training/formal/policy_5000.ckpt'),
                      ('A',ROOT/'training/A/policy_6000.ckpt'),('B',ROOT/'training/B/policy_6000.ckpt')]:
        policy,cp=load_policy(path)
        folder=out/name;folder.mkdir()
        dataset,splits=representative_dataset(manifest,cp['stats'])
        metrics=deployment_metrics(policy,dataset,splits)
        save_json(folder/'deployment_metrics.json',metrics)
        record=dict(checkpoint=str(path),deployment_metrics=metrics,approach=[],full_nominal=[])
        records[name]=record
        for case,scene,scenario in cases:
            result=runner(scene,BoundedPolicy(policy),cp['stats'],scenario,
                          folder/('short_'+case),runtime_profile='history_common_v1')
            measurements=json.loads((folder/('short_'+case)/'measurements.json').read_text())
            measured=approach_metrics(scene,result,measurements)
            record['approach'].append(dict(id=case,**measured))
            save_json(out/'progress.json',dict(stage='approach',records=records,hardware_commands_sent=False))
            print(json.dumps(dict(stage='short_approach',variant=name,case=case,**measured)),flush=True)
        if all(r['eligible_for_full_nominal'] for r in record['approach']):
            for case,scene,scenario in cases:
                result=runner(scene,policy,cp['stats'],scenario,folder/('full_'+case),runtime_profile='history_common_v1')
                record['full_nominal'].append(dict(id=case,passed=full_task_passed(result),result=result))
                save_json(out/'progress.json',dict(stage='full_nominal',records=records,hardware_commands_sent=False))
        record['full_nominal_passed']=(len(record['full_nominal'])==3 and all(r['passed'] for r in record['full_nominal']))
        del policy,cp,dataset;torch.cuda.empty_cache()
    eligible=[name for name in ['A','B'] if records[name]['full_nominal_passed']]
    selected=min(eligible,key=lambda n:records[n]['deployment_metrics']['records']['dev']['deployment_prefix2_MAE_rad']) if eligible else None
    full_development=None
    if selected:
        policy,cp=load_policy(records[selected]['checkpoint'])
        full_development=evaluate_development(policy,cp['stats'],manifest,out/'selected_full44')
    verify_bindings(binding)
    a,b=records['A']['deployment_metrics']['records'],records['B']['deployment_metrics']['records']
    comparison=dict(approach_prefix_error_A_rad=a['approach']['deployment_prefix2_MAE_rad'],
        approach_prefix_error_B_rad=b['approach']['deployment_prefix2_MAE_rad'],
        approach_prefix_relative_change_B_over_A=b['approach']['deployment_prefix2_MAE_rad']/a['approach']['deployment_prefix2_MAE_rad']-1,
        dev_prefix_relative_change_B_over_A=b['dev']['deployment_prefix2_MAE_rad']/a['dev']['deployment_prefix2_MAE_rad']-1,
        complete_task_repaired=bool(full_development and full_development['adaptation_gate_passed']))
    result=dict(complete=True,additional_optimizer_updates=2000,records=records,comparison=comparison,
        selected_for_full44=selected,full_development=full_development,
        independent_holdout_consumed=False,hardware_commands_sent=False,
        next_action='report_result_no_automatic_extra_training_or_hardware')
    save_json(ROOT/'results.json',result)
    save_json(ROOT/'execution.json',dict(stage='complete',pid=os.getpid(),hardware_commands_sent=False))
    lines=['# 执行前缀监督对照结果','',
        '源5K、A/B同初始模型与AdamW状态，各新增1000步；仅B七轴损失改变。',
        '短测为600执行点诊断，不计完整抓放成功。','',
        '|模型|接近前2点MAE(rad)|开发前2点MAE(rad)|短测到达高位接近目标|完整中心通过|',
        '|---|---:|---:|---:|---:|']
    for name,r in records.items():
        m=r['deployment_metrics']['records'];short=sum(t['approach_reached'] for t in r['approach'])
        full=sum(t['passed'] for t in r['full_nominal'])
        lines.append(f"|{name}|{m['approach']['deployment_prefix2_MAE_rad']:.8f}|{m['dev']['deployment_prefix2_MAE_rad']:.8f}|{short}/3|{full}/{len(r['full_nominal'])}|")
    lines.extend(['',f"完整44项候选：{selected or '无；接近或完整中心条件未通过'}。",
        '完整流程是否修复：'+str(comparison['complete_task_repaired'])+'。',
        '未消耗独立40项测试，未发送硬件命令；完整门槛未通过时不自动增加训练或切换结构。'])
    (ROOT/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stage='complete',comparison=comparison,selected=selected)),flush=True)
