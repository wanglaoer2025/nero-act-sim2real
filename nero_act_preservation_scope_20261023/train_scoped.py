"""One-variable correction: all trainable scopes, identical data and1000 batches."""
import json
from pathlib import Path
import sys

import numpy as np

from group_scope import preservation_scale

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'nero_act_dense_response_20261023'
sys.path.insert(0, str(SOURCE))
import train_dense as previous
trainer = previous.trainer
original_initializer = trainer.load_initializer
state = dict(groups=None, calls=0, records=[])


def load_initializer():
    policy, stats, groups = original_initializer()
    state['groups'] = groups
    return policy, stats, groups


def prepare_data(policy, stats):
    dataset, batches, manifest, scene, files = previous.prepare_data(policy, stats)
    fixed = np.load(SOURCE/'training/pilot_A/batch_order.npy', allow_pickle=False)
    if not np.array_equal(batches, fixed):
        raise ValueError('The controlled gradient comparison changed sample order')
    audit = json.loads((ROOT/'sampling_audit.json').read_text())
    audit['batches_identical_to_dense_trial'] = True
    (ROOT/'sampling_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    files += [SOURCE/'repair_plan.json', SOURCE/'training/pilot_A/batch_order.npy',
              ROOT/'current_scene.json']
    return dataset, batches, manifest, scene, files


def all_scopes(task, replay, projection_only):
    if state['groups'] is None:
        raise RuntimeError('Declared trainable scopes must be loaded first')
    weight, audit = preservation_scale(task, replay, state['groups'])
    state['calls'] += 1
    if state['calls']==1 or state['calls']%100==0:
        state['records'].append(dict(update=state['calls'], coefficient=weight, **audit))
        (ROOT/'gradient_scope_audit.json').write_text(json.dumps(dict(records=state['records'],
            ratio_limit=.5, maximum_coefficient=.1, scope='all_four_trainable_groups'), indent=2)+'\n')
    return weight, audit


previous.ROOT = ROOT
previous.retained.ROOT = ROOT
trainer.ROOT = ROOT
trainer.load_initializer = load_initializer
trainer.prepare_data = prepare_data
trainer.preservation_scale = all_scopes

if __name__ == '__main__':
    trainer.main()
