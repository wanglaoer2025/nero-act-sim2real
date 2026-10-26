"""Keep v2's16 verified parent recoveries; regenerate only2 nominal contexts."""
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from simulation.history_teacher import collect_partial
from training.history_recoveries import validate_history_manifest,sha


def main():
    previous=ROOT/'training_data/history_recoveries_v2/manifest.json'
    original=json.loads(previous.read_text())
    rows=copy.deepcopy(original['episodes'])
    if len(rows)!=16 or any(r['source_kind']!='teacher_interruption' or sha(r['path'])!=r['sha256']
                            or sha(r['context_path'])!=r['context_sha256'] for r in rows):
        raise ValueError('Exactly16 unchanged verified parent rows required')
    output=ROOT/'training_data/history_recoveries_v3';output.mkdir(parents=True,exist_ok=False)
    (output/'contexts').mkdir()
    captures=ROOT/'results/original_failure_capture'
    source=json.loads((captures/'runtime_checkpoints.json').read_text())
    scene=json.loads((captures/'scene_configuration.json').read_text())
    scene['generalization']=dict(index='nominal_development',source_group='nominal_diagnostic_train',split='train')
    torch.cuda.init()
    result=dict(schema='history_recovery_v1',complete=False,episodes=rows,base_episodes=original['base_episodes'],rejected=[],
        verified_parent_batch=str(previous),verified_parent_batch_sha256=sha(previous),
        rejected_v2_episode_2016_used=False,generation_driver_sha256=sha(__file__))
    try:
        for role in ['pre_close','hold_to_lift']:
            context=output/'contexts'/('nominal_'+role+'.json')
            context.write_text(json.dumps(source['named'][role],allow_nan=False))
            index=2000+len(rows)
            report=collect_partial(copy.deepcopy(scene),context,role,output/f'episode_{index}.hdf5',
                parent_group='nominal_diagnostic_train',split='train')
            rows.append(dict(report,id=index,parent_episode_id='nominal_development',source_group='nominal_diagnostic_train',
                split='train',role=role,source_kind='ACT_failure',status='valid',label_source='offline_verified_teacher',
                context_path=str(context.resolve()),context_sha256=sha(context)))
            print(json.dumps(dict(completed=len(rows),total=18,role=role,replay=report['replay'])),flush=True)
        split=json.loads((ROOT.parent/'nero_act_generalized_pick_place_20261018/training_data/generalized_v1/split.json').read_text())
        result['complete']=True;result['group_validation']=validate_history_manifest(result,split)
    except BaseException as exc:
        result['complete']=False;result['rejected'].append(dict(error=type(exc).__name__+': '+str(exc)))
        (output/'manifest.json').write_text(json.dumps(result,indent=2));raise
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,accepted=18,groups=result['group_validation'])),flush=True)


if __name__=='__main__':main()
