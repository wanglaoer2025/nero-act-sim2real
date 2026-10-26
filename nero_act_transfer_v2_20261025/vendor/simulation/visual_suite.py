"""Parallel independent simulation processes, aggregated only after all16 finish."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from simulation.visual_evaluation import aggregate_cases, file_hashes, require_files, evaluation_dependencies,full_task_passed

ROOT=Path(__file__).resolve().parents[1]


def case_command(training_run,output,index,*,protocol_path,development=False,nominal_summary=None):
    if not 0<=index<16:raise ValueError('Expected one of16 cases')
    command=[sys.executable,'-u','-m','simulation.visual_evaluation','--training-run',str(training_run),
        '--output',str(output),'--case-index',str(index),'--protocol',str(protocol_path)]
    if development:command.append('--development-regression')
    if nominal_summary is not None:command+=['--nominal-summary',str(nominal_summary)]
    return command


def validate_child_summary(summary,child_bindings,parent_bindings,name):
    if (child_bindings!=parent_bindings or not parent_bindings or summary.get('complete') is not True
            or summary.get('finalized') is not True or summary.get('total')!=1
            or len(summary.get('results',[]))!=1 or summary['results'][0].get('scenario',{}).get('name')!=name
            or summary.get('passed') not in (0,1)
            or bool(summary['passed'])!=full_task_passed(summary['results'][0])):
        raise ValueError('Child case identity, physical outcome, finalization or frozen dependencies changed')
    return dict(name=name,passed=summary['passed']==1,result=summary['results'][0],provenance=summary.get('provenance',{}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--development-regression',action='store_true')
    p.add_argument('--protocol',type=Path)
    p.add_argument('--nominal-summary',type=Path)
    a=p.parse_args()
    if not 1<=a.workers<=4:p.error('1..4 independent simulation workers required')
    declared_protocol=Path(a.protocol or ROOT/'evaluation/protocol.json').resolve()
    protocol=json.loads((ROOT/'evidence/baseline_development_protocol.json' if a.development_regression else declared_protocol).read_text())
    names=([c['name'] for c in protocol['resolved_cases']] if 'resolved_cases' in protocol
           else [c['id'] for c in protocol['cases']])
    dependencies=evaluation_dependencies(ROOT,a.training_run,protocol_path=declared_protocol)
    if a.development_regression:
        config=json.loads((a.training_run/'config.json').read_text())
        if config.get('history_runtime_profile')=='history_common_v1':
            from executor.history_qualification import require_development_gate
            dependencies+=require_development_gate(ROOT,a.training_run,nominal_summary=a.nominal_summary)
    bindings=file_hashes(dependencies)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'bindings.json').write_text(json.dumps(bindings,indent=2))
    records=[]
    def run(index):
        output=a.output/f'case_{index:02d}'
        command=case_command(a.training_run,output,index,protocol_path=declared_protocol,development=a.development_regression,nominal_summary=a.nominal_summary)
        with (a.output/f'case_{index:02d}.log').open('x') as log:
            result=subprocess.run(command,cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Case {index} process failed; inspect case log')
        summary=json.loads((output/'summary.json').read_text())
        child=json.loads((output/'protocol.json').read_text())
        record=validate_child_summary(summary,child['dependencies'],bindings,names[index])
        record['summary']=str(output/'summary.json');return record
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending=[pool.submit(run,i) for i in range(16)]
        for future in as_completed(pending):
            record=future.result();records.append(record)
            (a.output/'status.json').write_text(json.dumps(dict(completed=len(records),total=16,
                passed=sum(r['passed'] for r in records),last=record['name']),indent=2))
            print(json.dumps(dict(completed=len(records),name=record['name'],passed=record['passed'])),flush=True)
    require_files(bindings)
    records.sort(key=lambda r:names.index(r['name']))
    summary=aggregate_cases(records,names)
    summary.update(assessment_scope='retained_nominal_simulation_only_not_real_deployment_qualification',
        assessment_role='paired_baseline_development_not_independent' if a.development_regression else 'fresh_independent_parameters_frozen_before_training',
        all_parameters_predeclared_before_training=protocol.get('parameters_frozen_before_optimization') is True,measured_geometry=False,
        live_frontend_qualified=False,real_hardware_tested=False,source_and_model_unchanged=True,
        checkpoint_sha256=bindings[str((a.training_run/'checkpoints/policy_best_val.ckpt').resolve())])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(passed=summary['passed'],total=16,complete=True)),flush=True)


if __name__=='__main__':main()
