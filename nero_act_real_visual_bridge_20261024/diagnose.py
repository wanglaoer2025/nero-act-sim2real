"""Post-hoc cached-input interventions; never optimizes or changes the protocol."""
import json
from pathlib import Path
import hashlib

import numpy as np
import torch

from bridge import VisualResidual, AuxiliaryHead, auxiliary_loss


ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    run = ROOT / 'run_03'
    cache = torch.load(run/'cache.pt', map_location='cpu', weights_only=False)['real']['val']
    records = {}
    for name in ('A', 'B'):
        path = run/name/'sidecar_last.pt'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = torch.load(path, map_location='cpu', weights_only=False)
        adapter, head = VisualResidual().eval(), AuxiliaryHead().eval()
        adapter.load_state_dict(payload['adapter']); head.load_state_dict(payload['auxiliary_head'])
        def loss(features, state):
            predicted = head(adapter(features), state)
            return float(auxiliary_loss(predicted, cache['measured'], cache['targets'], cache['padding']))
        with torch.no_grad():
            features, state = cache['features'], cache['state']
            baseline = loss(features, state)
            image_values, state_values = [], []
            for seed in range(10):
                order = torch.randperm(len(state), generator=torch.Generator().manual_seed(100+seed))
                image_values.append(loss(features[order], state))
                state_values.append(loss(features, state[order]))
            # Same feature values, permuted spatial positions; no new image labels.
            order = torch.randperm(features.shape[2]*features.shape[3], generator=torch.Generator().manual_seed(77))
            spatial = features.flatten(2)[:, :, order].reshape_as(features)
            original_prediction = head(adapter(features), state)
            spatial_prediction = head(adapter(spatial), state)
            spatial_delta = float((original_prediction-spatial_prediction).abs().max())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        assert spatial_delta < 1e-5, 'Unexpected spatial dependence; inspect the pooling assumption'
        records[name] = dict(original_loss=baseline,
            image_permutation_loss_mean=float(np.mean(image_values)),
            image_permutation_loss_std=float(np.std(image_values)),
            state_permutation_loss_mean=float(np.mean(state_values)),
            state_permutation_loss_std=float(np.std(state_values)),
            final_feature_spatial_permutation_max_prediction_delta=spatial_delta,
            sidecar_sha256=before)
    result = dict(post_hoc_diagnostic=True, variants=records, permutations=10,
        heldout_windows=100, optimizer_updates=0,
        interpretation='Auxiliary global pooling cannot supervise feature-map position; state/image intervention measures this auxiliary head, not all ACT visual competence.',
        original_protocol_changed=False, policy_adopted=False, hardware_commands_sent=False)
    path = ROOT/'input_intervention_diagnostic.json'
    with path.open('x') as stream: stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
