"""Parallel independent simulation processes, aggregated only after all16 finish."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from simulation.visual_evaluation import aggregate_cases, file_hashes, require_files, evaluation_dependencies

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=4:p.error('1..4 independent simulation workers required')
    protocol=json.loads((ROOT/'evaluation/protocol.json').read_text())
    names=[c['id'] for c in protocol['cases']]
    dependencies=evaluation_dependencies(ROOT,a.training_run)
    bindings=file_hashes(dependencies)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'bindings.json').write_text(json.dumps(bindings,indent=2))
    records=[]
    def run(index):
        output=a.output/f'case_{index:02d}'
        command=[sys.executable,'-u','-m','simulation.visual_evaluation','--training-run',str(a.training_run),
                 '--output',str(output),'--case-index',str(index)]
        with (a.output/f'case_{index:02d}.log').open('x') as log:
            result=subprocess.run(command,cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Case {index} process failed; inspect case log')
        summary=json.loads((output/'summary.json').read_text())
        child=json.loads((output/'protocol.json').read_text())
        if child['dependencies']!=bindings:
            raise RuntimeError('Child case physical/code/model dependencies differ from suite freeze')
        if summary['complete'] is not True or summary['finalized'] is not True or summary['total']!=1:
            raise RuntimeError('Per-case evaluation did not finalize')
        return dict(name=names[index],passed=summary['passed']==1,result=summary['results'][0],
                    provenance=summary['provenance'],summary=str(output/'summary.json'))
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
        all_parameters_predeclared_before_training=False,measured_geometry=False,
        live_frontend_qualified=False,real_hardware_tested=False,source_and_model_unchanged=True,
        checkpoint_sha256=bindings[str((a.training_run/'checkpoints/policy_best_val.ckpt').resolve())])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(passed=summary['passed'],total=16,complete=True)),flush=True)


if __name__=='__main__':main()
