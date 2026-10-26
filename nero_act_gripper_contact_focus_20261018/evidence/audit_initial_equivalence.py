"""Compare new zero-residual initialization with the diagnosed parent on actual inputs."""
import json
import pickle
from pathlib import Path

import torch

from executor.model import BoundedActPolicy, physical_actions
from training.command_context import ReferenceDataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    source = ROOT.parent / 'nero_act_cvae_training_fix_20261018/model/run21_cvae_posterior'
    new = ROOT / 'training/run26_contact_focus'
    directories = [(source / 'policy.ckpt', source / 'dataset_stats.pkl'),
        (new / 'artifacts/rebased_initializer.ckpt', new / 'training_only_stats.pkl')]
    predictions = []
    for checkpoint, statistics in directories:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = dict(payload['config'], device='cpu', pretrained_backbone=False)
        with statistics.open('rb') as stream:
            stats = pickle.load(stream)
        policy = BoundedActPolicy(config).eval()
        policy.load_state_dict(payload['model'], strict=True)
        output = []
        for split in ['train', 'val']:
            manifest = ROOT.parent / f'nero_act_gripper_gradient_fix_20261018/training_data/direct_grasp_fresh_v1/{split}_manifest.json'
            path = Path(json.loads(manifest.read_text())['episodes'][0]['path'])
            image, q, _, _, _ = ReferenceDataset([path], config['camera_names'], stats, fixed_start_ts=0)[0]
            with torch.no_grad():
                output.append(physical_actions(policy(q[None], image[None]), stats, bounded_gripper=True))
        predictions.append(output)
    errors = [float((before-after).abs().max()) for before, after in zip(*predictions)]
    report = dict(passed=max(errors)<1e-6, train_and_val_max_physical_action_difference=errors,
        compared='run21 vs run24 zero-residual paired initializer on actual grasp inputs',
        hardware_connected=False)
    (ROOT / 'evidence/parent_equivalence.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
