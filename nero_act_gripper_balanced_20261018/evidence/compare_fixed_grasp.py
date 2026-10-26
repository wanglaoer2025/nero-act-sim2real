"""Record the auxiliary component on fixed actual inputs before/after training."""
import json
import pickle
from pathlib import Path

import torch

from executor.model import BoundedActPolicy, physical_actions
from training.command_context import ReferenceDataset
from training.train import weighted_l1

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'training/run25_balanced_gripper'
    with (folder/'training_only_stats.pkl').open('rb') as stream:
        stats = pickle.load(stream)
    torch.set_num_threads(2)
    records = []
    for label, checkpoint in [('before', folder/'artifacts/rebased_initializer.ckpt'),
                              ('after', ROOT/'model/run25_balanced_gripper/policy.ckpt')]:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = dict(payload['config'], device='cpu', pretrained_backbone=False)
        policy = BoundedActPolicy(config).eval()
        policy.load_state_dict(payload['model'], strict=True)
        for split in ['train', 'val']:
            manifest = json.loads((ROOT.parent/f'nero_act_gripper_gradient_fix_20261018/training_data/direct_grasp_fresh_v1/{split}_manifest.json').read_text())
            sample = ReferenceDataset([Path(manifest['episodes'][0]['path'])],
                                      config['camera_names'], stats, fixed_start_ts=0)[0]
            image, q, target, padding, _ = sample
            with torch.inference_mode():
                out = policy(q[None], image[None], return_unbounded=True)
                bounded = physical_actions(out['a_hat'], stats, bounded_gripper=True)
                raw = out['unbounded_a_hat'][..., 7]*stats['action_std'][7]+stats['action_mean'][7]
                loss = weighted_l1(out['unbounded_a_hat'][..., 7:8], target[None, :20, 7:8], padding[None, :20])
            records.append(dict(model=label, split=split, prior_gripper_loss=float(loss),
                raw_width_mm=[float(raw.min()*1000), float(raw.max()*1000)],
                bounded_width_mm=[float(bounded[..., 7].min()*1000), float(bounded[..., 7].max()*1000)]))
    (ROOT/'evidence/fixed_grasp_readout_comparison.json').write_text(json.dumps(
        dict(records=records, future_labels_only_used_for_reported_loss=True,
             task_success_not_determined_by_this_report=True, hardware_connected=False), indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
