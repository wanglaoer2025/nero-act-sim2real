"""Bounded verification on training-only windows; never exports probe weights."""
import hashlib
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from training.command_context import ReferenceDataset
from training.rebase_initializer import rebase_initializer
from training.train import FullTaskPolicy

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent/'nero_act_grasp_state_recovery_20261018/training/run20_grasp_state_recovery'


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = json.loads((BASE/'config.json').read_text())
    config.update(device='cuda', pretrained_backbone=False, cvae_training=True, kl_weight=.001)
    manifest = json.loads((BASE/'manifest.json').read_text())
    split = json.loads((BASE/'split.json').read_text())
    rows = [r for r in manifest['episodes'] if r['id'] in split['train']]
    paths = [Path(row['path']) for row in rows]
    for row, path in zip(rows, paths):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    with (BASE/'training_only_stats.pkl').open('rb') as stream:
        stats = pickle.load(stream)
    selected = []
    selection = []
    for episode, stage in [(0, 'descend'), (0, 'close'), (0, 'lift'), (0, 'release'),
                           (3001, 'descend'), (3001, 'close'), (3002, 'lift'), (3002, 'release')]:
        index = next(i for i, row in enumerate(rows) if row['id'] == episode)
        with h5py.File(paths[index], 'r') as data:
            start = int(np.flatnonzero(data['simulation/stage'].asstr()[:] == stage)[0])
        selected.append((index, start))
        selection.append(dict(episode=episode, stage=stage, start=start, split='train'))
    dataset = ReferenceDataset(paths, config['camera_names'], stats, sample_starts=selected)
    samples = [dataset[i] for i in range(len(dataset))]
    image = torch.stack([s[0] for s in samples]).cuda()
    q = torch.stack([s[1] for s in samples]).cuda()
    target = torch.stack([s[2][:20] for s in samples]).cuda()
    padding = torch.stack([s[3][:20] for s in samples]).cuda()
    source = torch.load(ROOT.parent/'nero_act_pose_objective_fix_20261018/model/run19_pose_objective/policy.ckpt',
                        map_location='cpu', weights_only=False)
    old = {k: np.array(v, np.float32) for k, v in source['config']['constraint_loss_stats'].items()}
    policy = FullTaskPolicy(config).cuda().eval()
    policy.load_state_dict(rebase_initializer(source['model'], old, stats, config), strict=True)
    control = FullTaskPolicy(dict(config, cvae_training=False, kl_weight=0.)).cuda().eval()
    control.load_state_dict(torch.load(BASE/'artifacts/rebased_initializer.ckpt',
                                      map_location='cpu', weights_only=False)['model'], strict=True)
    with torch.inference_mode():
        initial_difference = float((policy(q, image)-control(q, image)).abs().max())
    assert initial_difference <= 1e-6, initial_difference
    before = policy.model.encoder_action_proj.weight.detach().clone()
    policy.train()
    records = []
    for step in range(8):
        policy.optimizer.zero_grad(set_to_none=True)
        output = policy(q, image, target, padding)
        assert torch.isfinite(output['loss']) and float(output['kl']) > 0
        kl_grad = torch.autograd.grad(output['kl'], policy.model.latent_proj.weight, retain_graph=True)[0]
        assert torch.isfinite(kl_grad).all() and float(kl_grad.abs().max()) > 0
        output['loss'].backward()
        posterior_grad = policy.model.encoder_action_proj.weight.grad
        assert posterior_grad is not None and torch.isfinite(posterior_grad).all()
        assert float(posterior_grad.abs().max()) > 0
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), .1)
        assert torch.isfinite(norm)
        policy.optimizer.step()
        record = dict(step=step+1, loss=float(output['loss'].detach()),
                      l1=float(output['l1'].detach()), kl=float(output['kl'].detach()),
                      kl_only_gradient_max=float(kl_grad.abs().max()),
                      posterior_gradient_max=float(posterior_grad.abs().max()))
        records.append(record)
        print(json.dumps(record), flush=True)
    assert not torch.equal(before, policy.model.encoder_action_proj.weight)
    policy.eval()
    with torch.inference_mode():
        assert torch.isfinite(policy(q, image)).all()
        first = policy(q, image, target, padding)
        second = policy(q, image, target+10., padding)
        assert torch.equal(first['a_hat'], second['a_hat']) and float(first['kl']) == 0
    result = dict(passed=True, steps=records, windows=selection,
                  initial_zero_latent_difference_from_run20=initial_difference,
                  posterior_weights_updated=True, kl_gradient_independently_verified=True,
                  validation_ignores_future_labels=True, weights_saved=False,
                  not_a_task_success_test=True, hardware_connected=False)
    (ROOT/'evidence/real_batch_verification.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
