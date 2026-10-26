"""Continue authorized offline stages; no retraining loop and no robot API."""
import json,argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def next_stage(collection,pilot,formal,revalidation=None):
    if collection.get('rejected'):return 'blocked_data'
    if not collection.get('complete'):return 'collecting'
    if pilot is None:return 'pilot'
    if not pilot.get('complete') or not pilot.get('mechanism_passed'):return 'blocked_pilot'
    if pilot.get('admission_scope')=='mechanism_pilot_only':
        if revalidation is None:return 'revalidate_pilot'
        if not (revalidation.get('complete') and revalidation.get('mechanism_passed')
                and revalidation.get('admission_allowed')):return 'blocked_pilot_revalidation'
    if formal is None:return 'formal'
    return 'awaiting_real_data_or_development_result'


def read(path):
    if not path.exists():return None
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return None


def parse_arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/v2_complete/manifest.json')
    return p.parse_args(argv)


def main():
    args=parse_arguments()
    review=read(ROOT/'evidence/code_review.json')
    if not review or not review.get('implementation_gate_passed'):
        raise RuntimeError('Code review must finish before automated optimization')
    manifest=args.manifest.resolve()
    while True:
        collection=read(manifest) or {'complete':False,'episodes':[],'rejected':[]}
        pilot=read(ROOT/'training/pilot/completion.json')
        formal=read(ROOT/'training/formal/completion.json')
        revalidation=read(ROOT/'training/pilot_full_data_revalidation/completion.json')
        if revalidation is not None:
            revalidation['admission_allowed']=(read(ROOT/'training/pilot_full_data_revalidation/admission.json') or {}).get('allowed')
        stage=next_stage(collection,pilot,formal,revalidation)
        state=dict(stage=stage,accepted=len(collection.get('episodes',[])),
                   rejected=len(collection.get('rejected',[])),hardware_commands_sent=False,
                   expected_total=150,pilot_updates=(read(ROOT/'training/pilot/status.json') or {}).get('updates',0),
                   formal_updates=(read(ROOT/'training/formal/status.json') or {}).get('updates',0))
        (ROOT/'evidence/pipeline_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
        if stage=='collecting':time.sleep(30);continue
        if stage in ['pilot','formal','revalidate_pilot']:
            output=ROOT/'training'/('pilot_full_data_revalidation' if stage=='revalidate_pilot' else stage)
            if output.exists():
                raise RuntimeError('Previous incomplete attempt requires audit, no automatic retry:'+str(output))
            with (ROOT/'evidence'/f'{stage}.log').open('x') as log:
                command=[sys.executable,str(ROOT/('pilot_revalidate.py' if stage=='revalidate_pilot' else 'train.py')),
                         '--manifest',str(manifest),'--output',str(output)]
                if stage!='revalidate_pilot':command+=['--mode',stage]
                if stage=='formal' and revalidation is not None:
                    command+=['--pilot-proof',str(ROOT/'training/pilot_full_data_revalidation/completion.json')]
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    state['child_pid']=child.pid
                    state['formal_updates']=(read(ROOT/'training/formal/status.json') or {}).get('updates',0)
                    temporary=ROOT/'evidence/pipeline_state.tmp'
                    temporary.write_text(json.dumps(state,indent=2)+'\n')
                    temporary.replace(ROOT/'evidence/pipeline_state.json')
                    time.sleep(5)
            if child.returncode:
                state.update(stage='blocked_'+stage,exit_code=child.returncode)
                (ROOT/'evidence/pipeline_state.json').write_text(json.dumps(state,indent=2)+'\n')
                return child.returncode
            continue
        # Real demonstration qualification/real motion and independent40-case
        # acceptance are deliberately NOT inferred from simulation progress.
        return 0 if stage.startswith('awaiting') else 2


if __name__=='__main__':raise SystemExit(main())
