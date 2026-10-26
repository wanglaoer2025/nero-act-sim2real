import importlib.util
import json
from pathlib import Path
import unittest

import torch


class HeadFinetuneTests(unittest.TestCase):
    def test_training_updates_only_the_existing_action_head_and_preserves_frozen_buffers(self):
        self.assertIsNotNone(importlib.util.find_spec('training.head_finetune'))
        from training.head_finetune import HeadPolicy
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'model/source_run11/training_config.json').read_text())
        config.update(device='cpu', pretrained_backbone=False)
        torch.set_num_threads(2)
        policy = HeadPolicy(config)
        self.assertEqual({n for n, p in policy.named_parameters() if p.requires_grad},
                         {'model.action_head.weight', 'model.action_head.bias'})
        with torch.no_grad():
            policy.model.action_head.weight.zero_()
            policy.model.action_head.bias.zero_()
        before = {n: v.clone() for n, v in policy.state_dict().items()}
        policy.train()
        qpos, image = torch.zeros(1, 8), torch.zeros(1, 1, 3, 90, 160)
        with torch.no_grad():
            target = policy(qpos, image).clone()
        target[:, :, 7] -= .1
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
        loss = policy(qpos, image, target, torch.zeros(1, 20, dtype=torch.bool))['loss']
        loss.backward()
        optimizer.step()
        changed = {n for n, v in policy.state_dict().items() if not torch.equal(before[n], v)}
        self.assertTrue(changed)
        self.assertLessEqual(changed, {'model.action_head.weight', 'model.action_head.bias'})


if __name__ == '__main__':
    unittest.main()
