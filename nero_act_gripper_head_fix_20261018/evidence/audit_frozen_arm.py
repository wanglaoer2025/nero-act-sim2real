"""Refuse export if head adaptation changed any learned arm/visual tensor."""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'training/run23_gripper_head_only'
    initial = torch.load(folder/'artifacts/rebased_initializer.ckpt', map_location='cpu', weights_only=False)
    reports = []
    for checkpoint in sorted((folder/'checkpoints').glob('policy*.ckpt')):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert payload['config']['train_gripper_head_only'] is True
        assert payload['config']['weight_decay'] == 0.
        before, after = initial['model'], payload['model']
        assert before.keys() == after.keys()
        changed = []
        for name in before:
            if name in ['model.action_head.weight', 'model.action_head.bias']:
                assert torch.equal(before[name][:7], after[name][:7]), name
                if not torch.equal(before[name][7], after[name][7]):
                    changed.append(name)
            else:
                assert torch.equal(before[name], after[name]), name
        assert changed, 'Gripper adaptation did not update its output row'
        reports.append(dict(checkpoint=str(checkpoint), epoch=payload['epoch'],
            only_gripper_row_changed=changed, all_other_weights_and_buffers_identical=True))
    assert len(reports) >= 2
    (ROOT/'evidence/frozen_arm_audit.json').write_text(json.dumps(dict(passed=True,
        checkpoints=reports, comparison='completed checkpoints vs paired rebased initializer',
        scope='identical inputs preserve arm outputs; changed observations still require closed-loop tests',
        hardware_connected=False), indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
