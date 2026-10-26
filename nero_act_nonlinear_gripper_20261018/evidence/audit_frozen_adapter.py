"""Refuse export unless only the new internal gripper adapter was optimized."""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'training/run24_nonlinear_gripper'
    initial = torch.load(folder / 'artifacts/rebased_initializer.ckpt',
                         map_location='cpu', weights_only=False)
    reports = []
    for checkpoint in sorted((folder / 'checkpoints').glob('policy*.ckpt')):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert payload['config']['train_nonlinear_gripper_only'] is True
        before, after = initial['model'], payload['model']
        assert before.keys() == after.keys()
        changed = [key for key in before if not torch.equal(before[key], after[key])]
        assert changed and all(key.startswith('gripper_adapter.') for key in changed), changed
        reports.append(dict(checkpoint=str(checkpoint), epoch=payload['epoch'],
            changed_parameters=changed, all_pretrained_parameters_and_buffers_identical=True))
    assert len(reports) >= 2
    report = dict(passed=True, checkpoints=reports, hardware_connected=False,
        comparison='all saved checkpoints vs paired zero-residual initializer')
    (ROOT / 'evidence/frozen_adapter_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
