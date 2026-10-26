"""Protected tiered physics for the frozen-core arm-coordinate comparison."""
import copy,json,os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,torch,h5py
from experiment import BASE,ArmPolicy
from support import save_json,verify_bindings,verify_snapshots
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
    ds=cp.get('delta_stats',json.loads((ROOT/'delta_stats.json').read_text()))
    p=ArmPolicy(cp['stats'],ds,variant=cp.get('arm_variant','A'),pretrained=False).cuda().eval()
    if 'arm_variant' in cp:
        p.load_state_dict(cp['model'],strict=True);p.config=dict(cp['config'])
    else:p.load_source(cp)
    return p,cp


def scene_cases(manifest):
    with h5py.File(manifest['source_scene']) as f:base=json.loads(f.attrs['scene_config'])
    result=[]
    for case in development_cases()[:3]:
        scene=copy.deepcopy(base);rng=np.random.default_rng(case['seed'])
        scene['initial_robot_state']['qpos_rad']=(np.asarray(base['initial_robot_state']['qpos_rad'])+rng.uniform(-.0002,.0002,7)).tolist()
        scenario=dict(seed=case['seed'],query_stride=2,servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False)
        result.append((case['id'],scene,scenario))
    return result


def evaluate_comparison(binding):
    verify_bindings(binding);proof=json.loads((ROOT/'data_proof.json').read_text());verify_snapshots(proof)
    output=ROOT/'evaluation';output.mkdir(exist_ok=False)
    manifest=json.loads((BASE/'data/v2_contour_01/manifest.json').read_text());runner=load_runner();cases=scene_cases(manifest)
    records={}
    for name,path in [('source20K',BASE/'training/formal/policy_20000.ckpt'),('A',ROOT/'training/A/policy.ckpt'),('B',ROOT/'training/B/policy.ckpt')]:
        p,cp=load_policy(path);folder=output/name;folder.mkdir()
        dataset,splits=representative_dataset(manifest,cp['stats']);metrics=deployment_metrics(p,dataset,splits)
        save_json(folder/'deployment_metrics.json',metrics)
        record=dict(checkpoint=str(path),deployment_metrics=metrics,approach=[],full_nominal=[]);records[name]=record
        for case,scene,scenario in cases:
            result=runner(scene,BoundedPolicy(p),cp['stats'],scenario,folder/('short_'+case),runtime_profile='history_common_v1')
            measurements=json.loads((folder/('short_'+case)/'measurements.json').read_text())
            measured=approach_metrics(scene,result,measurements);record['approach'].append(dict(id=case,**measured))
            save_json(output/'progress.json',dict(stage='short_approach',records=records,hardware_commands_sent=False))
            print(json.dumps(dict(stage='short_approach',variant=name,case=case,**measured)),flush=True)
        if all(row['eligible_for_full_nominal'] for row in record['approach']):
            for case,scene,scenario in cases:
                result=runner(scene,p,cp['stats'],scenario,folder/('full_'+case),runtime_profile='history_common_v1')
                record['full_nominal'].append(dict(id=case,passed=full_task_passed(result),result=result))
                save_json(output/'progress.json',dict(stage='full_nominal',records=records,hardware_commands_sent=False))
        record['full_nominal_passed']=len(record['full_nominal'])==3 and all(t['passed'] for t in record['full_nominal'])
        del p,cp,dataset;torch.cuda.empty_cache()
    eligible=[v for v in ['A','B'] if records[v]['full_nominal_passed']]
    selected=min(eligible,key=lambda v:records[v]['deployment_metrics']['records']['dev']['deployment_prefix2_MAE_rad']) if eligible else None
    full=None
    if selected:
        p,cp=load_policy(records[selected]['checkpoint']);full=evaluate_development(p,cp['stats'],manifest,output/'selected_full44')
    verify_bindings(binding);verify_snapshots(proof)
    result=dict(complete=True,additional_updates=2000,records=records,selected_for_full44=selected,full_development=full,
        independent_holdout_consumed=False,hardware_commands_sent=False,
        complete_task_acceptance_passed=bool(full and full['adaptation_gate_passed']))
    save_json(ROOT/'results.json',result)
    save_json(ROOT/'execution.json',dict(stage='complete',pid=os.getpid(),hardware_commands_sent=False))
    lines=['# 七轴增量与尺度对照结果','','原20K已完成，四个检查点均0/44。新A/B各1000步，只有七轴新读出训练；核心、视觉、CVAE和夹爪保持冻结。',
        'A/B仅物理初始行为等价，头参数不同，两组均用全新优化器；不是旧优化器的精确续训。','',
        '|模型|接近前2点MAE(rad)|开发前2点MAE(rad)|短测到达目标|完整中心验收|',
        '|---|---:|---:|---:|---|']
    for name,r in records.items():
        m=r['deployment_metrics']['records'];short=sum(t['approach_reached'] for t in r['approach'])
        full_text=f"{sum(t['passed'] for t in r['full_nominal'])}/3" if r['full_nominal'] else '未进入'
        lines.append(f"|{name}|{m['approach']['deployment_prefix2_MAE_rad']:.8f}|{m['dev']['deployment_prefix2_MAE_rad']:.8f}|{short}/3|{full_text}|")
    lines.extend(['',f"完整44项候选：{selected or '无'}。",'独立40项未使用，真机未操作。短600点预算诊断不能当作完整任务失败/成功。',
        '限定读出实验的失败不能排除可训练主干的增量策略；不自动追加训练或扩大冻结范围。'])
    (ROOT/'results.md').write_text('\n'.join(lines)+'\n')


